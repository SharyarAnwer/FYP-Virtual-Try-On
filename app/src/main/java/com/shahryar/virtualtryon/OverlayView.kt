package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Transparent drawing layer stacked above the camera preview.
 *
 * Draws the garment (M3, articulated in M3.5) and the pose skeleton (M1). The garment is kept
 * behind the wearer's chin. When no pose is detected it falls back to the standing guide, so the
 * screen is never blank.
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
        const val LEFT_ELBOW = 13
        const val RIGHT_ELBOW = 14

        // Face landmarks used to keep the garment behind the chin.
        const val LEFT_EYE = 2
        const val RIGHT_EYE = 5
        const val LEFT_EAR = 7
        const val RIGHT_EAR = 8
        const val MOUTH_LEFT = 9
        const val MOUTH_RIGHT = 10

        /**
         * How far below the mouth the chin reaches, in eye-to-mouth distances. The pose model has
         * no chin point, so the jaw is estimated as an oval from ear to ear whose lowest point is
         * this far below the mouth. 1.2 reaches the bottom of the tester's beard; a clean-shaven
         * chin sits higher, nearer 0.8, where this hides a little of the collar's back that the
         * neck would actually show. Tuned on one person, like [FIXED_TORSO_EASE].
         */
        const val CHIN_DROP = 1.2f

        /**
         * The jaw oval's half-width, as a fraction of half the distance between the ear landmarks.
         * The pose model puts its ear points inside the face outline, about a third of the way in
         * from the real ears: on device an oval only as wide as them left the collar's corners
         * showing either side of the beard. 1.5 brings it out to the actual jaw.
         */
        const val JAW_WIDTH = 1.5f

        /**
         * Fixed widening of a real garment's torso below the armpits: 18% wider than the
         * shoulder-joint distance alone would make it.
         *
         * Measured on device in a fitted undershirt, the unwidened polo left 12-20 px of the
         * undershirt showing down each side - 13-16% of the shirt's width - because the tester's
         * torso is broader relative to his shoulders than a single ratio assumes. 0.18 covers that
         * with a small margin. It is tuned on one person and is the one number to replace when
         * torso width is measured live (M6).
         */
        const val FIXED_TORSO_EASE = 0.18f

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

    /**
     * Everything needed to draw one mesh garment (M3.6). Bone matrices and vertex buffers are
     * allocated here, once, and reused every frame.
     */
    private class MeshRender(
        val front: Bitmap,
        val sleeves: Bitmap,
        val mesh: GarmentMesh,
        val leftShoulder: Anchor,
        val rightShoulder: Anchor,
        val leftHip: Anchor,
        val rightHip: Anchor,
        val leftElbow: Anchor,
        val rightElbow: Anchor
    ) {
        /** Torso, left sleeve, right sleeve — the order of [GarmentMesh.weights]. */
        val bones = Array(GarmentMesh.BONES) { Matrix() }
        val placed = Array(GarmentMesh.BONES) { FloatArray(mesh.vertexCount * 2) }
        val vertices = FloatArray(mesh.vertexCount * 2)

        /** Widening per grid row. A flat fixed profile for now; a measured one later. */
        val easeAmount = FloatArray(mesh.rows + 1) { FIXED_TORSO_EASE }
        val widened = FloatArray(mesh.vertexCount * 2)

        companion object {
            fun from(garment: Garment, layers: GarmentLayers): MeshRender? {
                return MeshRender(
                    front = layers.meshFront ?: return null,
                    sleeves = layers.meshSleeves ?: return null,
                    mesh = layers.mesh ?: return null,
                    leftShoulder = garment.anchor("left_shoulder") ?: return null,
                    rightShoulder = garment.anchor("right_shoulder") ?: return null,
                    leftHip = garment.anchor("left_hip") ?: return null,
                    rightHip = garment.anchor("right_hip") ?: return null,
                    leftElbow = garment.anchor("left_elbow") ?: return null,
                    rightElbow = garment.anchor("right_elbow") ?: return null
                )
            }
        }
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

    /** The region the chin and jaw cover, in view coordinates; see [computeFace]. */
    private val facePath = Path()
    private val faceOval = RectF()
    private val faceMatrix = Matrix()

    /** Immutable snapshot written by the analysis thread, read by the UI thread. */
    @Volatile private var pose: FloatArray? = null
    @Volatile private var imageWidth = 0
    @Volatile private var imageHeight = 0

    /** Previous smoothed frame. Analysis thread only. */
    private var previous: FloatArray? = null

    @Volatile private var render: GarmentRender? = null

    /**
     * Whether to draw the pose skeleton. Off by default: drawn over a garment, its lines and
     * joints are the clearest possible signal that the shirt is an overlay rather than something
     * worn. It stays available for checking alignment, which is what it is for.
     */
    @Volatile var showSkeleton: Boolean = false
        set(value) {
            field = value
            postInvalidate()
        }
    @Volatile private var meshRender: MeshRender? = null

    /**
     * Sets the garment to render, or clears it when either argument is null. [layers] must be in
     * the same order as [Garment.parts].
     */
    fun setGarment(layers: GarmentLayers?, garment: Garment?) {
        var layered: GarmentRender? = null
        var meshed: MeshRender? = null
        if (layers != null && garment != null) {
            if (garment.render == RenderKind.MESH) {
                meshed = MeshRender.from(garment, layers)
            } else if (layers.fills.size == garment.parts.size &&
                layers.outlines.size == garment.parts.size && layers.seams.size == garment.parts.size) {
                layered = GarmentRender(garment, layers)
            }
        }
        render = layered
        meshRender = meshed
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
        val faceSeen = computeFace(snapshot)
        val checkpoint = canvas.save()
        if (faceSeen) canvas.clipOutPath(facePath)
        val wearingGarment = drawGarment(canvas, snapshot)
        canvas.restoreToCount(checkpoint)
        if (showSkeleton) {
            drawSkeleton(canvas, snapshot, dimmed = wearingGarment)
            if (faceSeen) canvas.drawPath(facePath, bonePaint)
        }
    }

    /**
     * Sets [facePath] to the region the wearer's chin and jaw cover. Returns false when the face
     * is not seen clearly enough to place it, and then nothing is hidden.
     *
     * The garment is drawn over the camera image, so it covers everything it overlaps. A real
     * shirt's collar stands up around the neck, and drawn on top it covered the tester's jaw and
     * beard up to his mouth. On a person the chin is in front of the collar, so the garment is
     * clipped out of this region.
     *
     * The region is an oval centred between the ears, as wide as they are apart, reaching
     * [CHIN_DROP] below the mouth. Its axis runs perpendicular to the line between the ears,
     * which tilts with the head and is a longer, steadier baseline than eyes to mouth. The eyes
     * and mouth only say which way is down and how long the face is.
     */
    private fun computeFace(p: FloatArray): Boolean {
        if (!visible(p, LEFT_EAR) || !visible(p, RIGHT_EAR) || !visible(p, LEFT_EYE) ||
            !visible(p, RIGHT_EYE) || !visible(p, MOUTH_LEFT) || !visible(p, MOUTH_RIGHT)) return false

        val earX = (viewX(p, LEFT_EAR) + viewX(p, RIGHT_EAR)) / 2f
        val earY = (viewY(p, LEFT_EAR) + viewY(p, RIGHT_EAR)) / 2f
        val acrossX = viewX(p, RIGHT_EAR) - viewX(p, LEFT_EAR)
        val acrossY = viewY(p, RIGHT_EAR) - viewY(p, LEFT_EAR)
        val earSpan = hypot(acrossX, acrossY)
        if (earSpan < 1f) return false

        val eyeX = (viewX(p, LEFT_EYE) + viewX(p, RIGHT_EYE)) / 2f
        val eyeY = (viewY(p, LEFT_EYE) + viewY(p, RIGHT_EYE)) / 2f
        val mouthX = (viewX(p, MOUTH_LEFT) + viewX(p, MOUTH_RIGHT)) / 2f
        val mouthY = (viewY(p, MOUTH_LEFT) + viewY(p, MOUTH_RIGHT)) / 2f

        // Down the face: perpendicular to the ear line, on the mouth's side of the eyes.
        var downX = -acrossY / earSpan
        var downY = acrossX / earSpan
        val eyeToMouth = (mouthX - eyeX) * downX + (mouthY - eyeY) * downY
        if (eyeToMouth < 0f) {
            downX = -downX
            downY = -downY
        }
        val faceLength = abs(eyeToMouth)
        if (faceLength < 1f) return false

        val depth = (mouthX - earX) * downX + (mouthY - earY) * downY + CHIN_DROP * faceLength
        if (depth <= 0f) return false
        val halfWidth = earSpan / 2f * JAW_WIDTH

        faceOval.set(-halfWidth, -depth, halfWidth, depth)
        facePath.rewind()
        facePath.addOval(faceOval, Path.Direction.CW)
        faceMatrix.setRotate(Math.toDegrees(atan2(downY, downX).toDouble()).toFloat() - 90f)
        faceMatrix.postTranslate(earX, earY)
        facePath.transform(faceMatrix)
        return true
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
        if (!visible(p, LEFT_SHOULDER) || !visible(p, RIGHT_SHOULDER) ||
            !visible(p, LEFT_HIP) || !visible(p, RIGHT_HIP)) return false
        meshRender?.let { return drawMeshGarment(canvas, p, it) }

        val r = render ?: return false
        val handedness = computeTorso(p, r.leftShoulder ?: return false, r.rightShoulder ?: return false,
            r.leftHip ?: return false, r.rightHip ?: return false)
        if (handedness == 0f) return false

        for (i in 0 until r.count) {
            val m = r.matrices[i]
            if (r.isTorso[i]) {
                m.set(torsoMatrix)
            } else if (!segmentMatrix(r.fromAnchor[i], r.toAnchor[i], r.fromLandmark[i],
                    r.toLandmark[i], p, handedness, m)) {
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
     * Sets [torsoMatrix] and [torsoInverse] from the torso anchors: both shoulders and the hip
     * midpoint, so the garment scales with shoulder width and torso length independently (M3).
     *
     * Returns the transform's handedness — -1 if it mirrors the artwork, which every limb must then
     * copy or a sleeve's outer edge would end up facing the body — or 0 if it is unusable.
     */
    private fun computeTorso(p: FloatArray, ls: Anchor, rs: Anchor, lh: Anchor, rh: Anchor): Float {
        src[0] = ls.x; src[1] = ls.y
        src[2] = rs.x; src[3] = rs.y
        src[4] = (lh.x + rh.x) / 2f; src[5] = (lh.y + rh.y) / 2f

        dst[0] = viewX(p, LEFT_SHOULDER); dst[1] = viewY(p, LEFT_SHOULDER)
        dst[2] = viewX(p, RIGHT_SHOULDER); dst[3] = viewY(p, RIGHT_SHOULDER)
        dst[4] = (viewX(p, LEFT_HIP) + viewX(p, RIGHT_HIP)) / 2f
        dst[5] = (viewY(p, LEFT_HIP) + viewY(p, RIGHT_HIP)) / 2f

        if (!torsoMatrix.setPolyToPoly(src, 0, dst, 0, 3)) return 0f
        if (!torsoMatrix.invert(torsoInverse)) return 0f

        torsoMatrix.getValues(matrixValues)
        val det = matrixValues[Matrix.MSCALE_X] * matrixValues[Matrix.MSCALE_Y] -
            matrixValues[Matrix.MSKEW_X] * matrixValues[Matrix.MSKEW_Y]
        return if (det >= 0f) 1f else -1f
    }

    /**
     * Draws a real garment photo bent over its grid (M3.6). Returns whether anything was drawn.
     *
     * Three bones — torso, left sleeve, right sleeve — are computed exactly as for layered
     * garments. Each grid vertex then moves to the weighted blend of where each bone would put it,
     * so the photo bends smoothly across the armhole instead of breaking at a cut. A sleeve whose
     * elbow is out of sight simply follows the torso, leaving it at rest on the body.
     *
     * Two passes: the whole garment, then only its sleeves again on top. A bent mesh draws its
     * triangles in grid order, so without the second pass the body could land over a sleeve
     * wherever the two overlap.
     *
     * Mirrors render() in tools/preview_mesh.py. Change the two together.
     */
    private fun drawMeshGarment(canvas: Canvas, p: FloatArray, m: MeshRender): Boolean {
        val handedness = computeTorso(p, m.leftShoulder, m.rightShoulder, m.leftHip, m.rightHip)
        if (handedness == 0f) return false

        m.bones[0].set(torsoMatrix)
        if (!segmentMatrix(m.leftShoulder, m.leftElbow, LEFT_SHOULDER, LEFT_ELBOW, p, handedness, m.bones[1])) {
            m.bones[1].set(torsoMatrix)
        }
        if (!segmentMatrix(m.rightShoulder, m.rightElbow, RIGHT_SHOULDER, RIGHT_ELBOW, p, handedness, m.bones[2])) {
            m.bones[2].set(torsoMatrix)
        }

        m.mesh.widen(m.easeAmount, m.widened)
        for (k in 0 until GarmentMesh.BONES) m.bones[k].mapPoints(m.placed[k], m.widened)

        val w = m.mesh.weights
        val t = m.placed[0]
        val l = m.placed[1]
        val r = m.placed[2]
        val out = m.vertices
        for (v in 0 until m.mesh.vertexCount) {
            val wt = w[v * 3]
            val wl = w[v * 3 + 1]
            val wr = w[v * 3 + 2]
            val x = v * 2
            out[x] = wt * t[x] + wl * l[x] + wr * r[x]
            out[x + 1] = wt * t[x + 1] + wl * l[x + 1] + wr * r[x + 1]
        }

        val cols = m.mesh.cols
        val rows = m.mesh.rows
        canvas.drawBitmapMesh(m.front, cols, rows, out, 0, null, 0, garmentPaint)
        canvas.drawBitmapMesh(m.sleeves, cols, rows, out, 0, null, 0, garmentPaint)
        return true
    }

    /**
     * Builds the transform for one limb, from its anchors [a0] -> [a1] on the artwork to its
     * landmarks [from] -> [to] on the body. Returns false when either is not usable.
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
    private fun segmentMatrix(
        a0: Anchor?, a1: Anchor?, from: Int, to: Int, p: FloatArray, handedness: Float, out: Matrix
    ): Boolean {
        if (a0 == null || a1 == null) return false
        if (!visible(p, from) || !visible(p, to)) return false

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
