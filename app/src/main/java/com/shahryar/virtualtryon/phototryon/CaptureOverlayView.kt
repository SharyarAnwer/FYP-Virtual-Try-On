package com.shahryar.virtualtryon.phototryon

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.os.SystemClock
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import android.util.AttributeSet
import android.view.View

/**
 * What a wearer standing across the room sees during a guided capture (Photo Try-On, stage 1):
 * the whole screen edged and tinted in one colour - red to adjust, amber when nearly there, green
 * to hold still - the instruction in large letters, the countdown as one huge digit, and a row of
 * eight marks along the top that fill in as each view of the turn is taken. Everything is sized to
 * be read from three metres, not from arm's length.
 *
 * Drawn over the camera preview. It draws nothing that follows the body, so the preview being
 * mirrored for the front camera does not matter to it.
 */
class CaptureOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    companion object {
        const val RED = 0xFFE53935.toInt()
        const val AMBER = 0xFFFFB300.toInt()
        const val GREEN = 0xFF43A047.toInt()
        private const val FLASH_MS = 220L
    }

    private val density = resources.displayMetrics.density
    private val edge = 22f * density

    private val edgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = edge }
    private val tintPaint = Paint()
    private val outlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
        color = 0xCCFFFFFF.toInt()
        pathEffect = DashPathEffect(floatArrayOf(14f * density, 10f * density), 0f)
    }
    private val textPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 30f * density
        typeface = Typeface.DEFAULT_BOLD
        setShadowLayer(8f * density, 0f, 2f * density, Color.BLACK)
    }
    private val countPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 220f * density
        typeface = Typeface.DEFAULT_BOLD
        textAlign = Paint.Align.CENTER
        setShadowLayer(14f * density, 0f, 4f * density, Color.BLACK)
    }
    private val markPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val flashPaint = Paint().apply { color = Color.WHITE }
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 34f * density
        strokeCap = Paint.Cap.BUTT
    }
    private val progressPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 96f * density
        typeface = Typeface.DEFAULT_BOLD
        textAlign = Paint.Align.CENTER
        setShadowLayer(12f * density, 0f, 3f * density, Color.BLACK)
    }
    private val pulsePaint = Paint().apply { color = GREEN }
    private var turning = false
    private var pulseAtMs = -FLASH_MS
    private val scratch = RectF()

    private var level: Level? = null
    private var text: CharSequence = ""
    private var layout: StaticLayout? = null
    private var count = 0
    private var views = 0
    private var taken = 0
    private var showOutline = false
    private var flashAtMs = -FLASH_MS

    /**
     * Shows one frame's verdict. [level] null hides the colour (before the capture starts and after
     * it ends). [count] above 0 replaces the instruction with that digit. [taken] has one bit per
     * view that is had, of [views] in all; 0 views hides the row of marks.
     */
    fun show(level: Level?, text: CharSequence, count: Int, taken: Int, views: Int, outline: Boolean, turning: Boolean = false) {
        if (text != this.text) {
            this.text = text
            layout = null
        }
        this.turning = turning
        this.level = level
        this.count = count
        this.taken = taken
        this.views = views
        this.showOutline = outline
        invalidate()
    }

    /** A white flash, as a camera's shutter. */
    fun flash() {
        flashAtMs = SystemClock.uptimeMillis()
        invalidate()
    }

    /** A green flash over the whole screen: a view of the turn was just taken. Seen from the side of the eye. */
    fun pulse() {
        pulseAtMs = SystemClock.uptimeMillis()
        invalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        layout = null
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val colour = when (level) {
            Level.ADJUST -> RED
            Level.NEARLY -> AMBER
            Level.READY -> GREEN
            null -> 0
        }
        if (colour != 0) {
            tintPaint.color = (colour and 0x00FFFFFF) or 0x30000000
            canvas.drawRect(0f, 0f, w, h, tintPaint)
            edgePaint.color = colour
            canvas.drawRect(edge / 2, edge / 2, w - edge / 2, h - edge / 2, edgePaint)
        }

        // Where to stand: a dashed frame the body should fill.
        if (showOutline) {
            scratch.set(w * 0.24f, h * 0.13f, w * 0.76f, h * 0.93f)
            canvas.drawRoundRect(scratch, 40f * density, 40f * density, outlinePaint)
        }

        // One mark per view, filled once it is taken.
        if (views > 0) {
            val top = edge + 12f * density
            val markH = 14f * density
            val gap = 8f * density
            val markW = (w - 2 * edge - gap * (views + 1)) / views
            for (i in 0 until views) {
                val left = edge + gap + i * (markW + gap)
                scratch.set(left, top, left + markW, top + markH)
                markPaint.color = if (taken and (1 shl i) != 0) GREEN else 0xB3000000.toInt()
                canvas.drawRoundRect(scratch, markH / 2, markH / 2, markPaint)
            }
        }

        // During the turn: a large ring of one segment per view, filled as each is taken, and the count inside.
        if (turning && views > 0) {
            val cx = w / 2
            val cy = h * 0.56f
            val r = minOf(w, h) * 0.30f
            scratch.set(cx - r, cy - r, cx + r, cy + r)
            val sweep = 360f / views
            for (i in 0 until views) {
                ringPaint.color = if (taken and (1 shl i) != 0) GREEN else 0x8C000000.toInt()
                canvas.drawArc(scratch, -90f + i * sweep + 3f, sweep - 6f, false, ringPaint)
            }
            val had = Integer.bitCount(taken)
            canvas.drawText("$had/$views", cx, cy - (progressPaint.ascent() + progressPaint.descent()) / 2, progressPaint)
        }

        if (count > 0) {
            canvas.drawText(count.toString(), w / 2, h / 2 - (countPaint.ascent() + countPaint.descent()) / 2, countPaint)
        } else if (text.isNotEmpty()) {
            val room = (w - 2 * edge - 32f * density).toInt().coerceAtLeast(1)
            val l = layout ?: StaticLayout.Builder.obtain(text, 0, text.length, textPaint, room)
                .setAlignment(Layout.Alignment.ALIGN_CENTER).setLineSpacing(0f, 1.05f).build().also { layout = it }
            canvas.save()
            canvas.translate(edge + 16f * density, edge + 44f * density)
            l.draw(canvas)
            canvas.restore()
        }

        val sincePulse = SystemClock.uptimeMillis() - pulseAtMs
        if (sincePulse < FLASH_MS * 2) {
            pulsePaint.alpha = (150 * (1f - sincePulse.toFloat() / (FLASH_MS * 2))).toInt().coerceIn(0, 255)
            canvas.drawRect(0f, 0f, w, h, pulsePaint)
            postInvalidateOnAnimation()
        }

        val sinceFlash = SystemClock.uptimeMillis() - flashAtMs
        if (sinceFlash < FLASH_MS) {
            flashPaint.alpha = (255 * (1f - sinceFlash.toFloat() / FLASH_MS)).toInt().coerceIn(0, 255)
            canvas.drawRect(0f, 0f, w, h, flashPaint)
            postInvalidateOnAnimation()
        }
    }
}
