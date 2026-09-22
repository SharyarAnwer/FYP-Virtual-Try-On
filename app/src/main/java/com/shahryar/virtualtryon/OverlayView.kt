package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Transparent drawing layer stacked above the camera preview.
 *
 * M0 draws a standing guide, which serves two purposes: it proves this layer composites
 * correctly over the preview, and it gives the user a target to stand in so the full body
 * lands inside frame.
 *
 * From M1 this is where the pose skeleton is drawn, and from M3 the warped garments.
 *
 * Paint objects are allocated once in [init] and never inside [onDraw]. onDraw runs on every
 * frame, so allocating there causes garbage-collection pauses that show up directly as frame
 * drops — the exact metric this project is judged on.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val density = resources.displayMetrics.density

    private val bracketPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#5FD3DB")
    }

    private val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1f * density
        color = Color.parseColor("#3D5FD3DB")
    }

    private val captionPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CCFFFFFF")
        textSize = 12f * density
        textAlign = Paint.Align.CENTER
        letterSpacing = 0.12f
    }

    private val guide = RectF()

    /** Set false once garments are being rendered and the guide would just be clutter. */
    var showGuide: Boolean = true
        set(value) {
            field = value
            invalidate()
        }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!showGuide) return

        val inset = width * 0.14f
        val vInset = height * 0.09f
        guide.set(inset, vInset, width - inset, height - vInset)

        val arm = minOf(guide.width(), guide.height()) * 0.08f

        // Corner brackets rather than a full rectangle — less of the frame is obscured.
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

        // Vertical centre line — the reference the garment transform will align to from M3.
        canvas.drawLine(guide.centerX(), guide.top + arm, guide.centerX(), guide.bottom - arm, axisPaint)

        canvas.drawText(
            "STAND WITH FULL BODY IN FRAME",
            guide.centerX(),
            guide.bottom + 22f * density,
            captionPaint
        )
    }
}
