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

/**
 * Transparent drawing layer stacked above the camera preview.
 *
 * Draws the warped garment (M3) and the pose skeleton (M1). When no pose is detected it falls
 * back to the standing guide, so the screen is never blank.
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

        // MediaPipe pose landmark indices used to place an upper-body garment.
        const val LEFT_SHOULDER = 11
        const val RIGHT_SHOULDER = 12
        const val LEFT_HIP = 23
        const val RIGHT_HIP = 24

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

    private val garmentMatrix = Matrix()
    private val srcTriangle = FloatArray(6)
    private val dstTriangle = FloatArray(6)

    /** Immutable snapshot written by the analysis thread, read by the UI thread. */
    @Volatile private var pose: FloatArray? = null
    @Volatile private var imageWidth = 0
    @Volatile private var imageHeight = 0

    /** Previous smoothed frame. Analysis thread only. */
    private var previous: FloatArray? = null

    @Volatile private var garmentBitmap: Bitmap? = null
    @Volatile private var garment: Garment? = null

    /** Sets the garment to render, or clears it when either argument is null. */
    fun setGarment(bitmap: Bitmap?, garment: Garment?) {
        this.garmentBitmap = bitmap
        this.garment = garment
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
        index * STRIDE + 2 < p.size && p[index * STRIDE + 2] >= MIN_VISIBILITY

    /**
     * Warps the garment onto the body and draws it. Returns whether anything was drawn.
     *
     * Three point pairs, so [Matrix.setPolyToPoly] produces an **affine** transform: the garment
     * scales with shoulder width, scales independently with torso length, rotates with the
     * shoulder line, and shears as the torso leans. Two pairs would give a rigid similarity that
     * ignores torso length; four would give a perspective transform that follows a twist more
     * closely but visibly wobbles as landmarks jitter. Affine is the stable middle, and it is
     * what the proposal specifies.
     *
     * The hip midpoint is used rather than both hips because two shoulders plus one hip point
     * define the triangle exactly — adding the fourth would over-constrain an affine fit.
     */
    private fun drawGarment(canvas: Canvas, p: FloatArray): Boolean {
        val bitmap = garmentBitmap ?: return false
        val g = garment ?: return false

        val required = intArrayOf(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        if (required.any { !visible(p, it) }) return false

        val leftShoulder = g.anchor("left_shoulder") ?: return false
        val rightShoulder = g.anchor("right_shoulder") ?: return false
        val leftHip = g.anchor("left_hip") ?: return false
        val rightHip = g.anchor("right_hip") ?: return false

        srcTriangle[0] = leftShoulder.x
        srcTriangle[1] = leftShoulder.y
        srcTriangle[2] = rightShoulder.x
        srcTriangle[3] = rightShoulder.y
        srcTriangle[4] = (leftHip.x + rightHip.x) / 2f
        srcTriangle[5] = (leftHip.y + rightHip.y) / 2f

        dstTriangle[0] = viewX(p, LEFT_SHOULDER)
        dstTriangle[1] = viewY(p, LEFT_SHOULDER)
        dstTriangle[2] = viewX(p, RIGHT_SHOULDER)
        dstTriangle[3] = viewY(p, RIGHT_SHOULDER)
        dstTriangle[4] = (viewX(p, LEFT_HIP) + viewX(p, RIGHT_HIP)) / 2f
        dstTriangle[5] = (viewY(p, LEFT_HIP) + viewY(p, RIGHT_HIP)) / 2f

        if (!garmentMatrix.setPolyToPoly(srcTriangle, 0, dstTriangle, 0, 3)) return false

        canvas.drawBitmap(bitmap, garmentMatrix, garmentPaint)
        return true
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
