package com.shahryar.virtualtryon.phototryon

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sqrt

/** Which pixels of a photo are the person, row by row: true is person. */
class Silhouette(val width: Int, val height: Int, val person: BooleanArray) {

    /** The first and last rows with a person in them, or -1 when there is nobody. */
    val top: Int
    val bottom: Int

    init {
        var first = -1
        var last = -1
        for (y in 0 until height) {
            var n = 0
            for (x in 0 until width) if (person[y * width + x]) n++
            if (n >= MIN_ROW_PIXELS) {
                if (first < 0) first = y
                last = y
            }
        }
        top = first
        bottom = last
    }

    /**
     * The unbroken run of person pixels in row [y] that holds column [x], or failing that the one
     * nearest to it: its first and last columns, or null for an empty row. Arms held away from the
     * body are separate runs, so the run through the body's centre line is the torso alone.
     */
    fun run(y: Int, x: Int): IntRange? {
        if (y !in 0 until height) return null
        val row = y * width
        var best: IntRange? = null
        var bestGap = Int.MAX_VALUE
        var start = -1
        for (c in 0..width) {
            val on = c < width && person[row + c]
            if (on && start < 0) start = c
            if (!on && start >= 0) {
                val gap = if (x < start) start - x else if (x > c - 1) x - (c - 1) else 0
                if (gap < bestGap) {
                    bestGap = gap
                    best = start until c
                }
                start = -1
            }
        }
        return best
    }

    private companion object {
        /** Fewer person pixels than this in a row is noise, not the top of a head. */
        const val MIN_ROW_PIXELS = 3
    }
}

/**
 * A wearer's measurements from their photos, centimetres. Lengths come from the front photo;
 * the three girths are round an ellipse as wide as the front photo shows and as deep as the side
 * photo shows.
 *
 * [girthsFromSide] is false when there was no usable side photo and the depths were taken as a
 * usual share of the widths instead: the girths are then rough. [scaleFromFace] is false when the
 * wearer's own eye spacing was not measured close up and an average adult's was assumed: every
 * figure is then only as good as that guess, about one part in twenty either way.
 */
data class Measurements(
    val heightCm: Float,
    val shoulderCm: Float,
    val chestCm: Float,
    val waistCm: Float,
    val hipCm: Float,
    val armCm: Float,
    val legCm: Float,
    val torsoCm: Float,
    val girthsFromSide: Boolean,
    val scaleFromFace: Boolean,
)

/**
 * Turns a front photo and a side photo into [Measurements] (Photo Try-On, stage 2).
 *
 * ## Scale without a tape or a typed height
 *
 * A photo has no scale of its own: someone tall and far away looks the same as someone short and
 * near. Something of known size has to be in it. The iris of the eye is very nearly the same in
 * every adult, [IRIS_MM] across, which is how MediaPipe measures distance from one camera. Across
 * a room an iris is too few pixels to measure, so it is read while the wearer is at the phone,
 * giving their eye spacing in millimetres ([ipdMm]); that spacing is then found in the full-body
 * photo, where it is some 150 pixels wide, and gives millimetres per pixel there.
 *
 * ## What is measured where
 *
 * Height is the silhouette's, top of head to soles. Shoulder, arm, leg and torso lengths are
 * between the pose model's joints. Chest, waist and hips are measured across the silhouette on
 * lines placed between the shoulder line and the hip line, through the body's centre line - arms
 * held away from the body fall outside the run of pixels that line passes through. The side photo
 * gives the depth on the same lines, and width and depth together give the distance round.
 *
 * The wearer's clothes are part of the silhouette: a loose shirt measures bigger than the body.
 */
object BodyMeasure {

    /** The width of an adult's iris, millimetres: 11.7, give or take 0.5. */
    const val IRIS_MM = 11.7f

    /** An average adult's eye spacing, millimetres: used when the wearer's own was not measured. */
    const val AVERAGE_IPD_MM = 63f

    /** Where chest, waist and hips are measured, as shares of the way from the shoulder line to the hip line. */
    const val CHEST_LEVEL = 0.22f
    const val WAIST_LEVEL = 0.68f
    const val HIP_LEVEL = 1.05f

    /** With no side photo, depth is taken as this share of width: usual figures for an adult. */
    const val CHEST_DEPTH = 0.72f
    const val WAIST_DEPTH = 0.82f
    const val HIP_DEPTH = 0.78f

    /** A measured depth outside these shares of the width is not believed. */
    const val MIN_DEPTH = 0.45f
    const val MAX_DEPTH = 1.25f

    /** The shoulders' full width over the distance between the shoulder joints. */
    const val SHOULDER_EDGE = 1.17f

    /** The torso may be at most this many times the joints' spacing wide: wider is an arm touching it. */
    const val MAX_TORSO_SPAN = 1.45f

    private const val LEFT_SHOULDER = CaptureGuide.LEFT_SHOULDER
    private const val RIGHT_SHOULDER = CaptureGuide.RIGHT_SHOULDER
    private const val LEFT_HIP = CaptureGuide.LEFT_HIP
    private const val RIGHT_HIP = CaptureGuide.RIGHT_HIP
    private const val LEFT_ELBOW = 13
    private const val LEFT_KNEE = 25
    private const val RIGHT_ELBOW = 14
    private const val RIGHT_KNEE = 26

    /** The wearer's eye spacing, millimetres, from a close view in which an iris is [irisPx] across and the eyes [ipdPx] apart. */
    fun ipdMm(irisPx: Float, ipdPx: Float) = ipdPx * IRIS_MM / irisPx

    /** The distance round an ellipse [width] across and [depth] deep (Ramanujan's formula). */
    fun around(width: Float, depth: Float): Float {
        val a = width / 2
        val b = depth / 2
        return (PI * (3 * (a + b) - sqrt((3 * a + b) * (a + 3 * b)))).toFloat()
    }

    /**
     * [front] and [frontPoints] are the front photo's silhouette and pose points (x, y from 0 to 1
     * and visibility, as [CaptureGuide] takes them); [side] and [sidePoints] the side photo's, or
     * null. [aspect] is the photos' width over height and [mmPerHeight] how many millimetres the
     * front photo's full height spans at the wearer's distance. Null if the body cannot be found.
     */
    fun measure(
        front: Silhouette, frontPoints: FloatArray, side: Silhouette?, sidePoints: FloatArray?,
        aspect: Float, mmPerHeight: Float, scaleFromFace: Boolean,
    ): Measurements? {
        if (front.top < 0 || front.bottom <= front.top || frontPoints.size < CaptureGuide.POINTS * 3) return null
        val cm = mmPerHeight / 10f
        fun x(p: Int) = frontPoints[p * 3] * aspect
        fun y(p: Int) = frontPoints[p * 3 + 1]
        fun between(a: Int, b: Int) = hypot(x(a) - x(b), y(a) - y(b))

        val height = (front.bottom - front.top + 1f) / front.height
        val joints = abs(x(LEFT_SHOULDER) - x(RIGHT_SHOULDER))
        if (joints <= 0f) return null
        val arm = (between(LEFT_SHOULDER, LEFT_ELBOW) + between(LEFT_ELBOW, CaptureGuide.LEFT_WRIST) +
            between(RIGHT_SHOULDER, RIGHT_ELBOW) + between(RIGHT_ELBOW, CaptureGuide.RIGHT_WRIST)) / 2
        val leg = (between(LEFT_HIP, LEFT_KNEE) + between(LEFT_KNEE, CaptureGuide.LEFT_ANKLE) +
            between(RIGHT_HIP, RIGHT_KNEE) + between(RIGHT_KNEE, CaptureGuide.RIGHT_ANKLE)) / 2
        val shoulderY = (y(LEFT_SHOULDER) + y(RIGHT_SHOULDER)) / 2
        val hipY = (y(LEFT_HIP) + y(RIGHT_HIP)) / 2
        val torso = hipY - shoulderY

        // The side photo is to the same scale once both bodies are made the same height.
        val sideScale = if (side != null && side.top >= 0 && side.bottom > side.top) height / ((side.bottom - side.top + 1f) / side.height) else Float.NaN

        var fromSide = side != null && sidePoints != null && !sideScale.isNaN()
        val girths = FloatArray(3)
        val levels = floatArrayOf(CHEST_LEVEL, WAIST_LEVEL, HIP_LEVEL)
        val usual = floatArrayOf(CHEST_DEPTH, WAIST_DEPTH, HIP_DEPTH)
        val widths = FloatArray(3)
        val depths = FloatArray(3)
        for (i in 0..2) {
            widths[i] = minOf(across(front, frontPoints, levels[i]) * aspect, joints * MAX_TORSO_SPAN)
            if (widths[i] <= 0f) return null
            depths[i] = if (fromSide) across(side!!, sidePoints!!, levels[i]) * aspect * sideScale else Float.NaN
            if (depths[i].isNaN() || depths[i] < widths[i] * MIN_DEPTH || depths[i] > widths[i] * MAX_DEPTH) fromSide = false
        }
        for (i in 0..2) girths[i] = around(widths[i], if (fromSide) depths[i] else widths[i] * usual[i]) * cm

        return Measurements(
            heightCm = height * cm, shoulderCm = joints * SHOULDER_EDGE * cm,
            chestCm = girths[0], waistCm = girths[1], hipCm = girths[2],
            armCm = arm * cm, legCm = leg * cm, torsoCm = torso * cm,
            girthsFromSide = fromSide, scaleFromFace = scaleFromFace,
        )
    }

    /**
     * How wide the body is at [level] between the shoulder line (0) and the hip line (1), as a
     * share of the photo's width: the run of silhouette through the body's centre line there.
     */
    private fun across(s: Silhouette, points: FloatArray, level: Float): Float {
        val shoulderX = (points[LEFT_SHOULDER * 3] + points[RIGHT_SHOULDER * 3]) / 2
        val shoulderY = (points[LEFT_SHOULDER * 3 + 1] + points[RIGHT_SHOULDER * 3 + 1]) / 2
        val hipX = (points[LEFT_HIP * 3] + points[RIGHT_HIP * 3]) / 2
        val hipY = (points[LEFT_HIP * 3 + 1] + points[RIGHT_HIP * 3 + 1]) / 2
        val row = ((shoulderY + level * (hipY - shoulderY)) * s.height).toInt()
        val column = ((shoulderX + level.coerceAtMost(1f) * (hipX - shoulderX)) * s.width).toInt()
        val run = s.run(row, column) ?: return 0f
        return (run.last - run.first + 1f) / s.width
    }
}
