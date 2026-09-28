package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.hypot

/**
 * Transparent drawing layer stacked above the camera preview.
 *
 * Draws the garment (M3, articulated in M3.5) and the pose skeleton (M1). When no pose is
 * detected it falls back to the standing guide, so the screen is never blank.
 *
 * ## The coordinate problem this class exists to solve
 *
 * MediaPipe returns landmarks normalised 0..1 against the *analysis image* (480x640). This view
 * spans the *preview*, which fills a 1080x2460 screen. PreviewView's FILL_CENTER scales the
 * camera image until it covers the view and crops the overflow — on this device that discards
 * roughly 40% of the image width.
 *
 * So `landmark.x * view.width` is wrong: it stretches a 3:4 image across a 21:9 view.
 * [computeMapping] applies the same scale-and-crop PreviewView does, which is why it must stay
 * in step with the preview's scale type.
 *
 * Paints, matrices and buffers are allocated once. onDraw runs every frame, and allocating there
 * causes GC pauses that surface directly as dropped frames.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private companion object {
        /** Landmarks below this confidence are skipped — usually limbs outside the frame. */
        const val MIN_VISIBILITY = 0.5f

        /** Stride of the packed landmark snapshot: x, y, visibility. */
        const val STRIDE = 3

        // MediaPipe pose landmark indices used to place the torso layer.
        const val LEFT_SHOULDER = 11
        const val RIGHT_SHOULDER = 12
        const val LEFT_HIP = 23
        const val RIGHT_HIP = 24

        /**
         * Length of the perpendicular used as a segment's third control point, in artwork px.
         * Only its direction and scale matter; a larger reach keeps the triangle well conditioned.
         */
        const val PERPENDICULAR_REACH = 64f

        /**
         * Exponential smoothing applied to landmark positions: 1.0 follows the model exactly,
         * lower values lag but steady.
         *
         * Raw landmarks jitter a pixel or two every frame, which is invisible on a thin skeleton
         * line and very visible on a garment the size of a torso — the shirt appears to shiver.
         * The cost is responsiveness: the garment trails fast movement slightly. 0.6 keeps the
         * shiver out without a lag you would notice at arm's length.
         */
        const val SMOOTHING = 0.6f

        /**
         * MediaPipe's 33-point pose topology, as index pairs.
         *
         * Spelled out rather than taken from the library's own constant so the skeleton's shape
         * is readable here and cannot shift under a dependency bump.
         */
        val CONNECTIONS = intArrayOf(
            0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10,
            11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 17, 19,
            12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20,
            11, 12, 11, 23, 12, 24, 23, 24,
            23, 25, 25, 27, 27, 29, 27, 31, 29, 31,
            24, 26, 26, 28, 28, 30, 28, 32, 30, 32
        )
    }

    /**
     * Everything needed to draw one garment, resolved once when it is selected so that onDraw
     * does no name lookups and no allocation. Immutable after construction, so swapping the
     * whole object is the only way the drawing thread ever sees a change.
     */
    private class GarmentRender(garment: Garment, layers: GarmentLayers) {
        val fills: List<Bitmap> = layers.fills
        val outlines: List<Bitmap?> = layers.outlines
        val seams: List<Bitmap?> = layers.seams
        val count = garment.parts.size
        val isTorso = BooleanArray(count) { garment.parts[it].kind == PartKind.TORSO }
        val fromLandmark = IntArray(count) { PoseLandmarks.indexOf(garment.parts[it].from) }
        val toLandmark = IntArray(count) { PoseLandmarks.indexOf(garment.parts[it].to) }
        val fromAnchor = Array(count) { garment.anchor(garment.parts[it].from) }
        val toAnchor = Array(count) { garment.anchor(garment.parts[it].to) }
        val parentIndex = IntArray(count) { i ->
            garment.parts.indexOfFirst { it.name == garment.parts[i].parent }
        }
        val matrices = Array(count) { Matrix() }

        val leftShoulder = garment.anchor("left_shoulder")
        val rightShoulder = garment.anchor("right_shoulder")
        val leftHip = garment.anchor("left_hip")
        val rightHip = garment.anchor("right_hip")
    }

    private val density = resources.displayMetrics.density

    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#5FD3DB")
    }

    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
    }

    private val bracketPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#5FD3DB")
    }

    private val captionPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CCFFFFFF")
        textSize = 12f * density
        textAlign = Paint.Align.CENTER
        letterSpacing = 0.12f
    }

    // Unlike the 90-degree frame rotation, this scales the artwork by an arbitrary factor, so
    // filtering genuinely changes the result rather than just costing time.
    private val garmentPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        isFilterBitmap = true
    }

    private val guide = RectF()
    private val segments = FloatArray(CONNECTIONS.size * 2)
    private val jointRadius = 3.5f * density
    private val mapping = FloatArray(3)

    private val torsoMatrix = Matrix()
    private val torsoInverse = Matrix()
    private val matrixValues = FloatArray(9)
    private val vector = FloatArray(2)
    private val src = FloatArray(6)
    private val dst = FloatArray(6)

    /** Immutable snapshot written by the analysis thread, read by the UI thread. */
    @Volatile private var pose: FloatArray? = null
    @Volatile private var imageWidth = 0
    @Volatile private var imageHeight = 0

    /** Previous smoothed frame. Analysis thread only. */
    private var previous: FloatArray? = null

    @Volatile private var render: GarmentRender? = null

    /**
     * Sets the garment to render, or clears it when either argument is null. [layers] must be in
     * the same order as [Garment.parts].
     */
    fun setGarment(layers: GarmentLayers?, garment: Garment?) {
        render = if (layers != null && garment != null && layers.fills.size == garment.parts.size &&
            layers.outlines.size == garment.parts.size && layers.seams.size == garment.parts.size) {
            GarmentRender(garment, layers)
        } else {
            null
        }
        postInvalidate()
    }

    /**
     * Publishes a new detection. Safe to call from the analysis thread.
     *
     * [imageWidth] and [imageHeight] are the dimensions of the bitmap fed to the model — already
     * upright and mirrored — not the raw sensor frame.
     */
    fun setPose(landmarks: List<NormalizedLandmark>?, imageWidth: Int, imageHeight: Int) {
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight

        val next = landmarks?.takeIf { it.isNotEmpty() }?.let { list ->
            FloatArray(list.size * STRIDE).also { out ->
                list.forEachIndexed { i, lm ->
                    out[i * STRIDE] = lm.x()
                    out[i * STRIDE + 1] = lm.y()
                    out[i * STRIDE + 2] = lm.visibility().orElse(1f)
                }
            }
        }

        if (next == null) {
            previous = null
            pose = null
        } else {
            val prev = previous
            if (prev != null && prev.size == next.size) {
                for (i in next.indices) {
                    next[i] = prev[i] + SMOOTHING * (next[i] - prev[i])
                }
            }
            previous = next
            pose = next
        }
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val snapshot = pose
        if (snapshot == null || imageWidth <= 0 || imageHeight <= 0) {
            drawGuide(canvas)
            return
        }

        computeMapping()
        val wearingGarment = drawGarment(canvas, snapshot)
        drawSkeleton(canvas, snapshot, dimmed = wearingGarment)
    }

    /**
     * Mirrors PreviewView's FILL_CENTER: scale until the image covers the view, centre it, and
     * let the overflow fall outside the bounds. Results land in [mapping] as scale, dx, dy.
     */
    private fun computeMapping() {
        val scale = maxOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        mapping[0] = scale
        mapping[1] = (width - imageWidth * scale) / 2f
        mapping[2] = (height - imageHeight * scale) / 2f
    }

    private fun viewX(p: FloatArray, index: Int) = p[index * STRIDE] * imageWidth * mapping[0] + mapping[1]
    private fun viewY(p: FloatArray, index: Int) = p[index * STRIDE + 1] * imageHeight * mapping[0] + mapping[2]
    private fun visible(p: FloatArray, index: Int) =
        index >= 0 && index * STRIDE + 2 < p.size && p[index * STRIDE + 2] >= MIN_VISIBILITY

    /**
     * Draws the garment. Returns whether anything was drawn.
     *
     * Four passes: every limb outline, then the torso, then every limb fill, then every seam. A
     * limb's outline is its silhouette grown by a few px, so whatever is drawn over it covers it —
     * the torso where a sleeve overlaps the chest, its own fill everywhere inside. What survives is
     * an outline only where a limb is the garment's outer edge. Drawing each limb with its own
     * outline, on top of the torso, put a line across the chest wherever a sleeve overlapped it.
     *
     * That rule cannot tell the chest from the underarm, so it also hid the one edge that should
     * show over the body: a long sleeve's inside edge below the armpit, which made the sleeve fuse
     * into the shirt like a batwing. Seams put exactly that edge back, drawn last.
     *
     * The torso uses M3's affine: both shoulders and the hip midpoint define the triangle, so the
     * garment scales with shoulder width and torso length independently. Each other layer is a
     * segment that follows its own two landmarks — see [segmentMatrix] — and falls back to its
     * parent's transform when those landmarks are not visible, which leaves the part in its rest
     * position on the body instead of letting it vanish.
     *
     * The whole garment still needs the torso's four landmarks. Without them there is nothing to
     * hang the limbs from.
     */
    private fun drawGarment(canvas: Canvas, p: FloatArray): Boolean {
        val r = render ?: return false
        if (!visible(p, LEFT_SHOULDER) || !visible(p, RIGHT_SHOULDER) ||
            !visible(p, LEFT_HIP) || !visible(p, RIGHT_HIP)) return false

        val ls = r.leftShoulder ?: return false
        val rs = r.rightShoulder ?: return false
        val lh = r.leftHip ?: return false
        val rh = r.rightHip ?: return false

        src[0] = ls.x; src[1] = ls.y
        src[2] = rs.x; src[3] = rs.y
        src[4] = (lh.x + rh.x) / 2f; src[5] = (lh.y + rh.y) / 2f

        dst[0] = viewX(p, LEFT_SHOULDER); dst[1] = viewY(p, LEFT_SHOULDER)
        dst[2] = viewX(p, RIGHT_SHOULDER); dst[3] = viewY(p, RIGHT_SHOULDER)
        dst[4] = (viewX(p, LEFT_HIP) + viewX(p, RIGHT_HIP)) / 2f
        dst[5] = (viewY(p, LEFT_HIP) + viewY(p, RIGHT_HIP)) / 2f

        if (!torsoMatrix.setPolyToPoly(src, 0, dst, 0, 3)) return false
        if (!torsoMatrix.invert(torsoInverse)) return false

        // If the torso transform mirrors the artwork, every limb must mirror with it, or a
        // sleeve's outer edge would end up facing the body.
        torsoMatrix.getValues(matrixValues)
        val det = matrixValues[Matrix.MSCALE_X] * matrixValues[Matrix.MSCALE_Y] -
            matrixValues[Matrix.MSKEW_X] * matrixValues[Matrix.MSKEW_Y]
        val handedness = if (det >= 0f) 1f else -1f

        for (i in 0 until r.count) {
            val m = r.matrices[i]
            if (r.isTorso[i]) {
                m.set(torsoMatrix)
            } else if (!segmentMatrix(r, i, p, handedness, m)) {
                val parent = r.parentIndex[i]
                m.set(if (parent >= 0) r.matrices[parent] else torsoMatrix)
            }
        }

        for (i in 0 until r.count) {
            if (!r.isTorso[i]) r.outlines[i]?.let { canvas.drawBitmap(it, r.matrices[i], garmentPaint) }
        }
        for (i in 0 until r.count) {
            if (r.isTorso[i]) canvas.drawBitmap(r.fills[i], r.matrices[i], garmentPaint)
        }
        for (i in 0 until r.count) {
            if (!r.isTorso[i]) canvas.drawBitmap(r.fills[i], r.matrices[i], garmentPaint)
        }
        for (i in 0 until r.count) {
            r.seams[i]?.let { canvas.drawBitmap(it, r.matrices[i], garmentPaint) }
        }
        return true
    }

    /**
     * Builds the transform for one limb layer. Returns false when its landmarks are not usable.
     *
     * Length follows the limb, width follows the body. Along the axis from pivot to far end, the
     * part stretches to the wearer's actual segment length. Across it, the part scales with the
     * torso's stretch *in that same direction*: the torso transform is anisotropic — on the test
     * subject 2.1 screen px per artwork px across the shoulders but 1.3 down the torso — and a
     * sleeve's width runs horizontally when the arm hangs but vertically when it is raised. Using
     * the horizontal scale throughout made a raised sleeve ~60% too thick; reading the scale along
     * the actual perpendicular, k = 1 / |T⁻¹·d|, keeps it the thickness of the shirt it belongs to.
     *
     * The same step-for-step maths lives in tools/preview_garments.py, which is how poses were
     * checked before reaching a phone. Change the two together.
     */
    private fun segmentMatrix(r: GarmentRender, i: Int, p: FloatArray, handedness: Float, out: Matrix): Boolean {
        val from = r.fromLandmark[i]
        val to = r.toLandmark[i]
        if (!visible(p, from) || !visible(p, to)) return false
        val a0 = r.fromAnchor[i] ?: return false
        val a1 = r.toAnchor[i] ?: return false

        val ax = a1.x - a0.x
        val ay = a1.y - a0.y
        val artLength = hypot(ax, ay)

        val v0x = viewX(p, from)
        val v0y = viewY(p, from)
        val vx = viewX(p, to) - v0x
        val vy = viewY(p, to) - v0y
        val viewLength = hypot(vx, vy)
        if (artLength < 1f || viewLength < 1f) return false

        // Unit perpendicular to the limb on screen, turned the same way the torso is.
        val dx = handedness * -vy / viewLength
        val dy = handedness * vx / viewLength

        vector[0] = dx
        vector[1] = dy
        torsoInverse.mapVectors(vector)
        val inverseLength = hypot(vector[0], vector[1])
        if (inverseLength < 1e-6f) return false
        val widthScale = 1f / inverseLength

        src[0] = a0.x; src[1] = a0.y
        src[2] = a1.x; src[3] = a1.y
        src[4] = a0.x - ay / artLength * PERPENDICULAR_REACH
        src[5] = a0.y + ax / artLength * PERPENDICULAR_REACH

        dst[0] = v0x; dst[1] = v0y
        dst[2] = v0x + vx; dst[3] = v0y + vy
        dst[4] = v0x + dx * PERPENDICULAR_REACH * widthScale
        dst[5] = v0y + dy * PERPENDICULAR_REACH * widthScale

        return out.setPolyToPoly(src, 0, dst, 0, 3)
    }

    /**
     * Draws the skeleton. When a garment is on, it is dimmed rather than hidden — seeing the
     * joints through the shirt is how you tell a correct fit from a plausible-looking one.
     */
    private fun drawSkeleton(canvas: Canvas, p: FloatArray, dimmed: Boolean) {
        bonePaint.alpha = if (dimmed) 90 else 255
        jointPaint.alpha = if (dimmed) 120 else 255

        val scale = mapping[0]
        val dx = mapping[1]
        val dy = mapping[2]

        var n = 0
        var i = 0
        while (i < CONNECTIONS.size) {
            val a = CONNECTIONS[i]
            val b = CONNECTIONS[i + 1]
            i += 2
            if (!visible(p, a) || !visible(p, b)) continue

            segments[n++] = p[a * STRIDE] * imageWidth * scale + dx
            segments[n++] = p[a * STRIDE + 1] * imageHeight * scale + dy
            segments[n++] = p[b * STRIDE] * imageWidth * scale + dx
            segments[n++] = p[b * STRIDE + 1] * imageHeight * scale + dy
        }
        if (n > 0) canvas.drawLines(segments, 0, n, bonePaint)

        var j = 0
        while (j * STRIDE + 2 < p.size) {
            if (visible(p, j)) {
                canvas.drawCircle(viewX(p, j), viewY(p, j), jointRadius, jointPaint)
            }
            j++
        }
    }

    private fun drawGuide(canvas: Canvas) {
        val inset = width * 0.14f
        val vInset = height * 0.09f
        guide.set(inset, vInset, width - inset, height - vInset)
        val arm = minOf(guide.width(), guide.height()) * 0.08f

        canvas.drawLines(
            floatArrayOf(
                guide.left, guide.top, guide.left + arm, guide.top,
                guide.left, guide.top, guide.left, guide.top + arm,
                guide.right, guide.top, guide.right - arm, guide.top,
                guide.right, guide.top, guide.right, guide.top + arm,
                guide.left, guide.bottom, guide.left + arm, guide.bottom,
                guide.left, guide.bottom, guide.left, guide.bottom - arm,
                guide.right, guide.bottom, guide.right - arm, guide.bottom,
                guide.right, guide.bottom, guide.right, guide.bottom - arm
            ),
            bracketPaint
        )

        canvas.drawText(
            "STAND WITH FULL BODY IN FRAME",
            guide.centerX(),
            guide.bottom + 22f * density,
            captionPaint
        )
    }
}
