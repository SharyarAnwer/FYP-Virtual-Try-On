package com.shahryar.virtualtryon

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.asin
import kotlin.math.sqrt

/**
 * A face's proportions, measured by Face Mesh (M7). Widths and length are relative to the
 * cheekbone width, the face's widest point, so they do not depend on how far away the face is.
 */
data class FaceProportions(
    /** Top of the forehead to the chin. */
    val lengthToWidth: Float,
    /** Across the forehead at the temples. */
    val foreheadToWidth: Float,
    /** Across the lower jaw, near its corners. */
    val jawToWidth: Float,
    /** At each jaw corner, between the cheek above and the chin, degrees: smaller is more angular. */
    val jawAngleDeg: Float,
)

/** A measured face shape and the proportions it was judged from. */
data class FaceShapeReading(val shape: FaceShape, val proportions: FaceProportions)

/**
 * Measures face shape from Face Mesh points (M7), for the glasses rule in [Recommender].
 *
 * Each frame gives four proportions from ten Face Mesh points, in 3D so a slight turn of the head
 * hardly changes them. Only frames with the face turned less than [MAX_TURN_DEG] are used, and
 * the reading is the median of [framesNeeded] of them, so a blink or a twitch of the tracker does
 * not decide it.
 *
 * The limits that sort proportions into shapes are set around MediaPipe's own canonical face,
 * read from the model file: its average face measures length 1.153, forehead 0.819 and jaw 0.775
 * times its cheekbone width, with jaw corners of 132 degrees, and classifies as oval. Published
 * face-shape guides cannot be used as they are: they measure length to the hairline, about 1.5
 * times the width, and Face Mesh's top point sits lower on the forehead. The bands around the
 * average are starting values, to be checked on real faces; a beard can hide the jawline and
 * widen the measured jaw.
 */
class FaceShapeMeter(val framesNeeded: Int = FRAMES_NEEDED) {

    companion object {
        // Face Mesh points. "Right" and "left" are the person's own.
        const val FOREHEAD_TOP = 10
        const val CHIN = 152
        const val CHEEK_RIGHT = 234
        const val CHEEK_LEFT = 454
        const val FOREHEAD_RIGHT = 54
        const val FOREHEAD_LEFT = 284
        const val JAW_RIGHT = 172
        const val JAW_LEFT = 397
        const val EYE_RIGHT_OUTER = 33
        const val EYE_LEFT_OUTER = 263

        /** Frames averaged into one reading: about a second and a half at 20 FPS. */
        const val FRAMES_NEEDED = 30

        /** A frame is used only with the head turned or tipped less than this, degrees. */
        const val MAX_TURN_DEG = 12f

        // MediaPipe's canonical face (see the class comment): the middle of the oval band.
        const val CANONICAL_LENGTH = 1.153f
        const val CANONICAL_FOREHEAD = 0.819f
        const val CANONICAL_JAW = 0.775f
        const val CANONICAL_JAW_ANGLE = 132.2f

        // Where the shapes begin, around the canonical face.
        /** Longer than this is oblong: about 8% longer than average. */
        const val LONG = 1.25f
        /** Shorter than this is round or square: about 6% shorter than average. */
        const val SHORT = 1.08f
        /** A jaw narrower than this under a forehead at least [HEART_FOREHEAD] wide is a heart. */
        const val HEART_JAW = 0.72f
        const val HEART_FOREHEAD = 0.82f
        /** A short face with a jaw this wide, or corners this sharp, is square rather than round. */
        const val SQUARE_JAW_SHORT = 0.82f
        const val SQUARE_ANGLE_DEG = 127f
        /** Any face with a jaw this wide and corners this sharp is square. */
        const val SQUARE_JAW = 0.85f

        /**
         * One frame's proportions from Face Mesh's [points] - x, y, z per point, x and y as
         * fractions of the [imageWidth] x [imageHeight] image, z on x's scale - or null when the
         * face is turned or tipped too far to measure.
         */
        fun proportions(points: FloatArray, imageWidth: Int, imageHeight: Int): FaceProportions? {
            if (points.size < FaceTracker.POINTS * 3) return null
            fun p(i: Int) = floatArrayOf(points[i * 3] * imageWidth, points[i * 3 + 1] * imageHeight, points[i * 3 + 2] * imageWidth)
            fun sub(a: FloatArray, b: FloatArray) = floatArrayOf(a[0] - b[0], a[1] - b[1], a[2] - b[2])
            fun len(v: FloatArray) = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            fun dist(a: Int, b: Int) = len(sub(p(a), p(b)))
            fun tiltDeg(v: FloatArray) = Math.toDegrees(asin((abs(v[2]) / len(v)).coerceAtMost(1f).toDouble())).toFloat()
            fun angleDeg(corner: Int, a: Int, b: Int): Float {
                val u = sub(p(a), p(corner))
                val w = sub(p(b), p(corner))
                val cos = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (len(u) * len(w))
                return Math.toDegrees(acos(cos.coerceIn(-1f, 1f).toDouble())).toFloat()
            }

            // Turned: the eye corners' depths differ. Tipped: the forehead's and chin's do.
            if (tiltDeg(sub(p(EYE_LEFT_OUTER), p(EYE_RIGHT_OUTER))) > MAX_TURN_DEG) return null
            if (tiltDeg(sub(p(CHIN), p(FOREHEAD_TOP))) > MAX_TURN_DEG) return null
            val width = dist(CHEEK_RIGHT, CHEEK_LEFT)
            if (width <= 0f) return null
            return FaceProportions(
                lengthToWidth = dist(FOREHEAD_TOP, CHIN) / width,
                foreheadToWidth = dist(FOREHEAD_RIGHT, FOREHEAD_LEFT) / width,
                jawToWidth = dist(JAW_RIGHT, JAW_LEFT) / width,
                jawAngleDeg = (angleDeg(JAW_RIGHT, CHEEK_RIGHT, CHIN) + angleDeg(JAW_LEFT, CHEEK_LEFT, CHIN)) / 2f,
            )
        }

        /** The shape these proportions make; see the limits above. */
        fun classify(p: FaceProportions): FaceShape = when {
            p.lengthToWidth >= LONG -> FaceShape.OBLONG
            p.jawToWidth <= HEART_JAW && p.foreheadToWidth >= HEART_FOREHEAD -> FaceShape.HEART
            p.lengthToWidth <= SHORT ->
                if (p.jawToWidth >= SQUARE_JAW_SHORT || p.jawAngleDeg <= SQUARE_ANGLE_DEG) FaceShape.SQUARE else FaceShape.ROUND
            p.jawToWidth >= SQUARE_JAW && p.jawAngleDeg <= SQUARE_ANGLE_DEG -> FaceShape.SQUARE
            else -> FaceShape.OVAL
        }

        private fun median(values: List<Float>): Float {
            val sorted = values.sorted()
            val mid = sorted.size / 2
            return if (sorted.size % 2 == 1) sorted[mid] else (sorted[mid - 1] + sorted[mid]) / 2f
        }
    }

    private val frames = ArrayList<FaceProportions>()

    /** Frames measured so far. */
    val count: Int get() = frames.size

    val done: Boolean get() = frames.size >= framesNeeded

    /** Adds one frame's face; false when it was skipped, the face being turned too far. */
    fun add(points: FloatArray, imageWidth: Int, imageHeight: Int): Boolean {
        if (done) return false
        val p = proportions(points, imageWidth, imageHeight) ?: return false
        frames += p
        return true
    }

    /** The median of each proportion over the frames so far, and its shape; null before any. */
    fun result(): FaceShapeReading? {
        if (frames.isEmpty()) return null
        val p = FaceProportions(
            lengthToWidth = median(frames.map { it.lengthToWidth }),
            foreheadToWidth = median(frames.map { it.foreheadToWidth }),
            jawToWidth = median(frames.map { it.jawToWidth }),
            jawAngleDeg = median(frames.map { it.jawAngleDeg }),
        )
        return FaceShapeReading(classify(p), p)
    }

    fun reset() {
        frames.clear()
    }
}
