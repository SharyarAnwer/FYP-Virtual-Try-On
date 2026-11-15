package com.shahryar.virtualtryon.phototryon

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** How close the wearer is to a photo: the colour the whole screen shows. */
enum class Level { ADJUST, NEARLY, READY }

/** What to tell the wearer. The words for each are in strings_photo_tryon.xml. */
enum class Prompt {
    NO_PERSON,
    MOVE_AWAY, COME_CLOSER,
    MOVE_LEFT, MOVE_RIGHT, STEP_FORWARD, STEP_BACKWARD,
    FACE_PHONE, LOOK_AHEAD, STAND_STRAIGHT,
    ARMS_OUT, ARMS_DOWN, FEET_APART, FEET_CLOSER,
    TOO_DARK,
    HOLD_STILL, COUNTING,
    TURN_SLOWLY, SLOWER, KEEP_TURNING,
    DONE
}

/**
 * One frame's verdict.
 *
 * - [count] is 3, 2 or 1 during the front photo's countdown, otherwise 0.
 * - [capture] is true on exactly one frame: take the front photo now.
 * - [grab] is the number of the view this very frame should be saved as, during the turn; -1
 *   on every other frame.
 * - [taken] has one bit per view, set once that view is had.
 * - [turnDeg] and [fill] are the measured turn and the share of the frame's height the body
 *   fills, for the on-screen readout; NaN when unknown.
 */
class Guidance(
    val prompt: Prompt,
    val level: Level,
    val count: Int,
    val capture: Boolean,
    val grab: Int,
    val taken: Int,
    val turnDeg: Float,
    val fill: Float,
)

/**
 * Decides, frame by frame, what a wearer standing away from the phone should do next so that
 * eight views of them can be taken hands-free (Photo Try-On, stage 1): one every [STEP_DEG]
 * degrees round the body, to be dressed and shown as a turntable of real photos. It only looks at
 * the pose model's points; the camera, the voice and the screen belong to GuidedCaptureActivity.
 *
 * ## Two parts
 *
 * **The front photo** is posed. Its checks, in the order they are raised: somebody is in view;
 * the whole body is in the frame at a sensible size; they are in the middle; they face the phone,
 * head and body; they stand straight; arms a little away from the body and feet a little apart,
 * so the sides of the torso and the inside of the legs can be seen; they hold still. Only the
 * first failing check is reported: one instruction at a time. When all pass and the wearer has
 * been still for [STILL_MS], a three-second countdown runs and the photo is taken at its end.
 * Moving or drifting out of position during the countdown cancels it; the limits are a little
 * looser then ([COUNTING_SLACK]) so a wearer right on one does not cancel it by breathing.
 *
 * **The turn** is not posed: eight held poses are too much to ask. The wearer turns slowly, all
 * the way round, and each time their body passes one of the seven other angles the frame in hand
 * is taken as that view. A frame is taken only while the body is whole in the frame and turning
 * no faster than [MAX_TURN_DEG_S], faster than which it blurs. Either way round works: the views
 * are numbered by angle, not by the order they came in.
 *
 * ## Which way is "left"
 *
 * The pose model must be given the camera's picture as it is, not mirrored, so that its left
 * shoulder is the wearer's own. Each point also comes as a "world" point in metres: x across the
 * picture, z away from the camera. Seen from above, the line from the right shoulder to the left
 * points along +x for a wearer facing the phone, and swings towards +z - the left shoulder moving
 * away - as they turn to their left. Its direction is the turn. The line is smoothed, not the
 * angle, because an angle jumps from 180 to -180 at the back where a line just keeps turning.
 *
 * Not thread-safe: call everything from one thread.
 */
class CaptureGuide {

    companion object {
        // The pose model's point numbers.
        const val NOSE = 0
        const val LEFT_EAR = 7
        const val RIGHT_EAR = 8
        const val LEFT_SHOULDER = 11
        const val RIGHT_SHOULDER = 12
        const val LEFT_WRIST = 15
        const val RIGHT_WRIST = 16
        const val LEFT_HIP = 23
        const val RIGHT_HIP = 24
        const val LEFT_ANKLE = 27
        const val RIGHT_ANKLE = 28
        const val LEFT_HEEL = 29
        const val RIGHT_HEEL = 30
        const val LEFT_TOE = 31
        const val RIGHT_TOE = 32

        const val POINTS = 33

        /** Points are packed three floats each: image points as x, y, visibility; world as x, y, z. */
        const val STRIDE = 3

        /** The views of a capture: the front is view 0, and the rest follow every [STEP_DEG] to the wearer's left. */
        const val VIEWS = 8
        const val STEP_DEG = 360f / VIEWS

        /** [taken] with every view had. */
        const val ALL_TAKEN = (1 shl VIEWS) - 1

        /** +1 if a turn to the wearer's left moves the left shoulder away from the camera in world z. */
        const val TURN_SIGN = 1f

        /** A body point counts as seen at or above this visibility. */
        const val SEEN = 0.5f

        /** Room that must be left above the head and below the feet, as a share of the frame's height. */
        const val EDGE_MARGIN = 0.02f

        /** The body, head to feet, must fill between these shares of the frame's height. */
        const val MIN_FILL = 0.55f
        const val MAX_FILL = 0.88f

        /** How far the hips' middle may be from the frame's middle, as a share of its width. */
        const val CENTRE_TOLERANCE = 0.12f

        /** How far off facing the phone the wearer may be for the front photo, degrees. */
        const val FACING_TOLERANCE_DEG = 15f

        /** How far the head may be turned from the body, degrees. */
        const val HEAD_TOLERANCE_DEG = 35f

        /** How far the torso may lean from upright, degrees. */
        const val LEAN_TOLERANCE_DEG = 8f

        /** Each arm, shoulder to wrist, must be this far from straight down, degrees: hanging is about 10. */
        const val MIN_ARM_DEG = 14f
        const val MAX_ARM_DEG = 45f

        /** The ankles must be between these multiples of the hips' width apart. */
        const val MIN_FEET_RATIO = 0.7f
        const val MAX_FEET_RATIO = 2.4f

        /** Moving further than this share of the body's height counts as moving. */
        const val STILL_MOVE = 0.02f

        /** How long the wearer must be still before the countdown starts, ms. */
        const val STILL_MS = 700L

        /** The countdown: three counts of this long, ms. */
        const val COUNT_MS = 1000L
        const val COUNTS = 3

        /** During the countdown every limit is this much looser. */
        const val COUNTING_SLACK = 1.4f

        /** Share of each new frame in the smoothed shoulder line: for the posed photo, then for the turn. */
        const val SMOOTHING = 0.35f
        const val TURN_SMOOTHING = 0.6f

        /** The shortest shoulder line that is believed, metres: under half an adult's. */
        const val MIN_SHOULDER_SPAN_M = 0.15f

        /** During the turn, a view is taken within this many degrees of its angle, or as the body passes it. */
        const val GRAB_WITHIN_DEG = 4f

        /** A body passing an angle is caught only if the frame before was this near to it, degrees. */
        const val PASSING_WITHIN_DEG = 25f

        /** Turning faster than this, degrees a second, blurs the frame: none is taken. */
        const val MAX_TURN_DEG_S = 60f

        /** During the turn the body may fill this much more or less of the frame than in the front photo. */
        const val TURN_DISTANCE = 0.18f

        /** With no view taken for this long during the turn, the wearer is told to keep going, ms. */
        const val KEEP_TURNING_MS = 5000L

        private val TRACKED = intArrayOf(NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST,
            LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE)

        /** [deg] brought into -180..180. */
        fun wrap(deg: Float): Float {
            var d = deg % 360f
            if (d > 180f) d -= 360f
            if (d < -180f) d += 360f
            return d
        }

        private fun degrees(y: Float, x: Float) = Math.toDegrees(atan2(y, x).toDouble()).toFloat()
    }

    /** One bit per view, set once it is had. */
    var taken = 0
        private set

    /** True once the front photo is had and the wearer is turning. */
    val turning get() = taken != 0 && !done

    /** True once all eight views are had. */
    val done get() = taken == ALL_TAKEN

    private var lineX = 0f
    private var lineZ = 0f
    private var linePrimed = false

    private val anchor = FloatArray(TRACKED.size * 2)
    private var anchoredAtMs = -1L
    private var countStartMs = -1L
    private var awaitingPhoto = false

    private var frontFill = Float.NaN
    private var lastFill = Float.NaN
    private var lastTurn = Float.NaN
    private var lastTurnMs = -1L
    private var lastGrabMs = -1L

    /** Starts again from the front photo. */
    fun restart() {
        taken = 0
        linePrimed = false
        anchoredAtMs = -1L
        countStartMs = -1L
        awaitingPhoto = false
        frontFill = Float.NaN
        lastTurn = Float.NaN
    }

    /** The front photo asked for by [Guidance.capture] has been saved: the turn begins. */
    fun onCaptured(timeMs: Long) {
        if (!awaitingPhoto) return
        awaitingPhoto = false
        countStartMs = -1L
        anchoredAtMs = -1L
        frontFill = lastFill
        taken = 1
        lastTurn = Float.NaN
        lastGrabMs = timeMs
    }

    /** The front photo could not be taken: go back to guiding for it. */
    fun onCaptureFailed() {
        awaitingPhoto = false
        countStartMs = -1L
        anchoredAtMs = -1L
    }

    /**
     * Takes one frame. [image] holds the pose's 33 points in the picture - x and y from 0 to 1,
     * then visibility - and [world] the same points in metres; both null when nobody was found.
     * [aspect] is the picture's width over its height. [tooDark] says the face is too dark to use.
     */
    fun update(image: FloatArray?, world: FloatArray?, aspect: Float, timeMs: Long, tooDark: Boolean = false): Guidance {
        if (done) return verdict(Prompt.DONE, Level.READY, Float.NaN, Float.NaN)
        if (awaitingPhoto) return verdict(Prompt.COUNTING, Level.READY, Float.NaN, Float.NaN)
        if (image == null || world == null || image.size < POINTS * STRIDE || world.size < POINTS * STRIDE ||
            !seen(image, LEFT_SHOULDER) || !seen(image, RIGHT_SHOULDER) || !seen(image, LEFT_HIP) || !seen(image, RIGHT_HIP)
        ) {
            linePrimed = false
            lastTurn = Float.NaN
            return fail(Prompt.NO_PERSON, Float.NaN, Float.NaN)
        }

        val counting = countStartMs >= 0
        val slack = if (counting) COUNTING_SLACK else 1f

        // Where the body is in the frame. x is put in the same units as y: shares of the frame's height.
        val noseY = y(image, NOSE)
        val shoulderY = (y(image, LEFT_SHOULDER) + y(image, RIGHT_SHOULDER)) / 2
        val hipY = (y(image, LEFT_HIP) + y(image, RIGHT_HIP)) / 2
        val shoulderX = (x(image, LEFT_SHOULDER) + x(image, RIGHT_SHOULDER)) / 2 * aspect
        val hipX = (x(image, LEFT_HIP) + x(image, RIGHT_HIP)) / 2 * aspect
        val headTop = noseY - 0.55f * (shoulderY - noseY)
        val feet = maxOf(maxOf(y(image, LEFT_ANKLE), y(image, RIGHT_ANKLE)), maxOf(y(image, LEFT_HEEL), y(image, RIGHT_HEEL)),
            maxOf(y(image, LEFT_TOE), y(image, RIGHT_TOE)))
        val fill = feet - headTop
        lastFill = fill
        val turn = measureTurn(world, if (turning) TURN_SMOOTHING else SMOOTHING)

        // All of the body in the frame, at a sensible size.
        val feetSeen = seen(image, LEFT_ANKLE) || seen(image, RIGHT_ANKLE)
        if (!feetSeen || headTop < EDGE_MARGIN / slack || feet > 1f - EDGE_MARGIN / slack || fill > MAX_FILL * slack) {
            return fail(Prompt.MOVE_AWAY, turn, fill)
        }
        if (fill < MIN_FILL / slack) return fail(Prompt.COME_CLOSER, turn, fill)

        // In the middle. Which way to move depends on which way they face.
        val off = (x(image, LEFT_HIP) + x(image, RIGHT_HIP)) / 2 - 0.5f
        val centre = CENTRE_TOLERANCE * (if (turning) 1.5f else slack)
        if (abs(off) > centre) return fail(sideways(-off, turn), turn, fill)
        if (turn.isNaN()) return fail(Prompt.NO_PERSON, turn, fill)

        if (turning) return turnVerdict(turn, fill, timeMs)

        // Facing the phone, body and head.
        if (abs(wrap(turn)) > FACING_TOLERANCE_DEG * slack) return fail(Prompt.FACE_PHONE, turn, fill)
        if (headOff(world) > HEAD_TOLERANCE_DEG * slack) return fail(Prompt.LOOK_AHEAD, turn, fill)

        // Standing straight.
        val lean = degrees(abs(shoulderX - hipX), hipY - shoulderY)
        if (lean > LEAN_TOLERANCE_DEG * slack) return fail(Prompt.STAND_STRAIGHT, turn, fill)

        // Arms a little out, feet a little apart.
        val leftArm = armDeg(image, LEFT_SHOULDER, LEFT_WRIST, aspect)
        val rightArm = armDeg(image, RIGHT_SHOULDER, RIGHT_WRIST, aspect)
        if (min(leftArm, rightArm) < MIN_ARM_DEG / slack) return fail(Prompt.ARMS_OUT, turn, fill)
        if (max(leftArm, rightArm) > MAX_ARM_DEG * slack) return fail(Prompt.ARMS_DOWN, turn, fill)
        val hips = abs(x(image, LEFT_HIP) - x(image, RIGHT_HIP))
        if (hips > 0f) {
            val feetRatio = abs(x(image, LEFT_ANKLE) - x(image, RIGHT_ANKLE)) / hips
            if (feetRatio < MIN_FEET_RATIO / slack) return fail(Prompt.FEET_APART, turn, fill)
            if (feetRatio > MAX_FEET_RATIO * slack) return fail(Prompt.FEET_CLOSER, turn, fill)
        }

        if (tooDark) return fail(Prompt.TOO_DARK, turn, fill)

        // Holding still.
        if (moved(image, aspect, fill, STILL_MOVE * slack) || anchoredAtMs < 0) {
            setAnchor(image, aspect, timeMs)
            countStartMs = -1L
            return verdict(Prompt.HOLD_STILL, Level.NEARLY, turn, fill)
        }
        if (!counting) {
            if (timeMs - anchoredAtMs < STILL_MS) return verdict(Prompt.HOLD_STILL, Level.NEARLY, turn, fill)
            countStartMs = timeMs
        }
        val elapsed = timeMs - countStartMs
        if (elapsed >= COUNT_MS * COUNTS) {
            awaitingPhoto = true
            return verdict(Prompt.COUNTING, Level.READY, turn, fill, capture = true)
        }
        return verdict(Prompt.COUNTING, Level.READY, turn, fill, count = COUNTS - (elapsed / COUNT_MS).toInt())
    }

    /** One frame of the turn: whether it is a view, and what to say. */
    private fun turnVerdict(turn: Float, fill: Float, timeMs: Long): Guidance {
        if (fill > frontFill * (1f + TURN_DISTANCE)) return fail(Prompt.MOVE_AWAY, turn, fill)
        if (fill < frontFill * (1f - TURN_DISTANCE)) return fail(Prompt.COME_CLOSER, turn, fill)

        val before = lastTurn
        val speed = if (before.isNaN() || timeMs <= lastTurnMs) 0f else abs(wrap(turn - before)) * 1000f / (timeMs - lastTurnMs)
        lastTurn = turn
        lastTurnMs = timeMs
        if (speed > MAX_TURN_DEG_S) return verdict(Prompt.SLOWER, Level.ADJUST, turn, fill)

        // The view whose angle the body is nearest, and how far from it.
        val view = (((turn / STEP_DEG).roundToInt() % VIEWS) + VIEWS) % VIEWS
        val offNow = wrap(turn - view * STEP_DEG)
        var grab = -1
        if (view != 0 && taken and (1 shl view) == 0) {
            val offBefore = if (before.isNaN()) Float.NaN else wrap(before - view * STEP_DEG)
            val passed = !offBefore.isNaN() && offBefore * offNow < 0f && abs(offBefore) < PASSING_WITHIN_DEG && abs(offNow) < PASSING_WITHIN_DEG
            if (abs(offNow) <= GRAB_WITHIN_DEG || passed) {
                grab = view
                taken = taken or (1 shl view)
                lastGrabMs = timeMs
            }
        }
        if (done) return verdict(Prompt.DONE, Level.READY, turn, fill, grab = grab)
        val prompt = if (timeMs - lastGrabMs > KEEP_TURNING_MS) Prompt.KEEP_TURNING else Prompt.TURN_SLOWLY
        return verdict(prompt, Level.NEARLY, turn, fill, grab = grab)
    }

    private fun verdict(prompt: Prompt, level: Level, turn: Float, fill: Float, count: Int = 0, capture: Boolean = false, grab: Int = -1) =
        Guidance(prompt, level, count, capture, grab, taken, turn, fill)

    private fun fail(prompt: Prompt, turn: Float, fill: Float): Guidance {
        countStartMs = -1L
        anchoredAtMs = -1L
        return verdict(prompt, Level.ADJUST, turn, fill)
    }

    /** The wearer's turn, degrees, from the smoothed shoulder line; NaN if the line cannot be believed. */
    private fun measureTurn(world: FloatArray, smoothing: Float): Float {
        val vx = wx(world, LEFT_SHOULDER) - wx(world, RIGHT_SHOULDER)
        val vz = (wz(world, LEFT_SHOULDER) - wz(world, RIGHT_SHOULDER)) * TURN_SIGN
        if (hypot(vx, vz) < MIN_SHOULDER_SPAN_M) return if (linePrimed) degrees(lineZ, lineX) else Float.NaN
        if (linePrimed) {
            lineX += smoothing * (vx - lineX)
            lineZ += smoothing * (vz - lineZ)
        } else {
            lineX = vx
            lineZ = vz
            linePrimed = true
        }
        return degrees(lineZ, lineX)
    }

    /** How far the head is turned from the body, degrees: the nose against the ears, seen from above. */
    private fun headOff(world: FloatArray): Float {
        val hx = wx(world, NOSE) - (wx(world, LEFT_EAR) + wx(world, RIGHT_EAR)) / 2
        val hz = (wz(world, NOSE) - (wz(world, LEFT_EAR) + wz(world, RIGHT_EAR)) / 2) * TURN_SIGN
        val length = hypot(hx, hz)
        val line = hypot(lineX, lineZ)
        if (length < 1e-4f || line < 1e-4f) return 0f
        // The body faces a quarter turn round from its right-to-left line.
        val fx = lineZ / line
        val fz = -lineX / line
        val cos = ((hx * fx + hz * fz) / length).coerceIn(-1f, 1f)
        return Math.toDegrees(acos(cos).toDouble()).toFloat()
    }

    /**
     * The instruction that moves the wearer [wantX] across the picture (positive: to its right),
     * in their own terms. Facing the phone their left is the picture's right; side-on, across the
     * picture is forwards or backwards for them.
     */
    private fun sideways(wantX: Float, turn: Float): Prompt {
        if (turn.isNaN()) return if (wantX > 0) Prompt.MOVE_LEFT else Prompt.MOVE_RIGHT
        val rad = Math.toRadians(turn.toDouble())
        val leftX = Math.cos(rad).toFloat()
        val forwardX = Math.sin(rad).toFloat()
        return if (abs(leftX) >= abs(forwardX)) {
            if (wantX * leftX > 0) Prompt.MOVE_LEFT else Prompt.MOVE_RIGHT
        } else {
            if (wantX * forwardX > 0) Prompt.STEP_FORWARD else Prompt.STEP_BACKWARD
        }
    }

    /** How far an arm is from hanging straight down, degrees, as it looks in the picture. */
    private fun armDeg(image: FloatArray, shoulder: Int, wrist: Int, aspect: Float): Float {
        val dx = abs(x(image, wrist) - x(image, shoulder)) * aspect
        val dy = y(image, wrist) - y(image, shoulder)
        return if (dy <= 0f) 180f else degrees(dx, dy)
    }

    private fun setAnchor(image: FloatArray, aspect: Float, timeMs: Long) {
        for (i in TRACKED.indices) {
            anchor[i * 2] = x(image, TRACKED[i]) * aspect
            anchor[i * 2 + 1] = y(image, TRACKED[i])
        }
        anchoredAtMs = timeMs
    }

    private fun moved(image: FloatArray, aspect: Float, fill: Float, limit: Float): Boolean {
        if (anchoredAtMs < 0) return true
        for (i in TRACKED.indices) {
            val d = hypot(x(image, TRACKED[i]) * aspect - anchor[i * 2], y(image, TRACKED[i]) - anchor[i * 2 + 1])
            if (d > limit * fill) return true
        }
        return false
    }

    private fun x(image: FloatArray, point: Int) = image[point * STRIDE]
    private fun y(image: FloatArray, point: Int) = image[point * STRIDE + 1]
    private fun seen(image: FloatArray, point: Int) = image[point * STRIDE + 2] >= SEEN
    private fun wx(world: FloatArray, point: Int) = world[point * STRIDE]
    private fun wz(world: FloatArray, point: Int) = world[point * STRIDE + 2]
}
