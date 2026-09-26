package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

/**
 * Transparent drawing layer stacked above the camera preview.
 *
 * Draws the pose skeleton from M1, and will carry warped garments from M3. When no pose is
 * detected it falls back to the standing guide, so the screen is never blank.
 *
 * ## The coordinate problem this class exists to solve
 *
 * MediaPipe returns landmarks normalised 0..1 against the *analysis image* (640x480). This view
 * spans the *preview*, which fills a 1080x2460 screen. PreviewView's FILL_CENTER scales the
 * camera image until it covers the view and crops the overflow — on this device that discards
 * roughly 40% of the image width.
 *
 * So `landmark.x * view.width` is wrong: it stretches a 3:4 image across a 21:9 view. The
 * skeleton would sit beside the body rather than on it, and the natural conclusion would be that
 * the model is broken. [mapToView] applies the same scale-and-crop PreviewView does, which is why
 * it must stay in step with the preview's scale type.
 *
 * Paints and the segment buffer are allocated once. onDraw runs every frame, and allocating there
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

        /**
         * MediaPipe's 33-point pose topology, as index pairs.
         *
         * Spelled out rather than taken from the library's own constant so the skeleton's shape
         * is readable here and cannot shift under a dependency bump.
         */
        val CONNECTIONS = intArrayOf(
            // face
            0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10,
            // arms
            11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 17, 19,
            12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20,
            // torso
            11, 12, 11, 23, 12, 24, 23, 24,
            // legs
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
        color = Color.parseColor("#FFFFFF")
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

    private val guide = RectF()
    private val segments = FloatArray(CONNECTIONS.size * 2)
    private val jointRadius = 3.5f * density

    /** Immutable snapshot written by the analysis thread, read by the UI thread. */
    @Volatile private var pose: FloatArray? = null
    @Volatile private var imageWidth = 0
    @Volatile private var imageHeight = 0

    /**
     * Publishes a new detection. Safe to call from the analysis thread.
     *
     * [imageWidth] and [imageHeight] are the dimensions of the bitmap that was fed to the model —
     * already upright and mirrored — not the raw sensor frame.
     */
    fun setPose(landmarks: List<NormalizedLandmark>?, imageWidth: Int, imageHeight: Int) {
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        pose = landmarks?.takeIf { it.isNotEmpty() }?.let { list ->
            FloatArray(list.size * STRIDE).also { out ->
                list.forEachIndexed { i, lm ->
                    out[i * STRIDE] = lm.x()
                    out[i * STRIDE + 1] = lm.y()
                    out[i * STRIDE + 2] = lm.visibility().orElse(1f)
                }
            }
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
        drawSkeleton(canvas, snapshot)
    }

    /**
     * Mirrors PreviewView's FILL_CENTER: scale until the image covers the view, centre it, and
     * let the overflow fall outside the bounds. Returns scale, offsetX, offsetY via [mapping].
     */
    private val mapping = FloatArray(3)

    private fun computeMapping() {
        val scale = maxOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        mapping[0] = scale
        mapping[1] = (width - imageWidth * scale) / 2f
        mapping[2] = (height - imageHeight * scale) / 2f
    }

    private fun drawSkeleton(canvas: Canvas, p: FloatArray) {
        computeMapping()
        val scale = mapping[0]
        val dx = mapping[1]
        val dy = mapping[2]

        var n = 0
        var i = 0
        while (i < CONNECTIONS.size) {
            val a = CONNECTIONS[i]
            val b = CONNECTIONS[i + 1]
            i += 2

            val aBase = a * STRIDE
            val bBase = b * STRIDE
            if (aBase + 2 >= p.size || bBase + 2 >= p.size) continue
            if (p[aBase + 2] < MIN_VISIBILITY || p[bBase + 2] < MIN_VISIBILITY) continue

            segments[n++] = p[aBase] * imageWidth * scale + dx
            segments[n++] = p[aBase + 1] * imageHeight * scale + dy
            segments[n++] = p[bBase] * imageWidth * scale + dx
            segments[n++] = p[bBase + 1] * imageHeight * scale + dy
        }
        if (n > 0) canvas.drawLines(segments, 0, n, bonePaint)

        var j = 0
        while (j * STRIDE + 2 < p.size) {
            if (p[j * STRIDE + 2] >= MIN_VISIBILITY) {
                canvas.drawCircle(
                    p[j * STRIDE] * imageWidth * scale + dx,
                    p[j * STRIDE + 1] * imageHeight * scale + dy,
                    jointRadius,
                    jointPaint
                )
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
