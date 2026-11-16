package com.shahryar.virtualtryon.phototryon

import kotlin.math.hypot

/**
 * One frame's verdict in free-pose mode. [problem] is the rule broken, or null; [samePose] is true when
 * the rules are kept but the pose is the one just photographed. [count] is 3, 2 or 1 during the countdown;
 * [capture] is true on exactly one frame: take the photo now. [taken] is how many photos are had.
 */
class FreeGuidance(
    val verdict: PoseVerdict?,
    val samePose: Boolean,
    val level: Level,
    val count: Int,
    val capture: Boolean,
    val taken: Int,
)

/**
 * Free-pose capture (Photo Try-On): the wearer poses as they like, up to [MAX_PHOTOS] times, and each
 * photo is taken by itself once the pose keeps [PoseRules] and is held still.
 *
 * - A rule broken: red, and what to fix.
 * - All rules kept: amber, "hold still in this pose for three seconds", held for [HOLD_MS] - long enough
 *   for that sentence - then green and a three-second countdown. Moving or breaking a rule cancels it.
 * - After a photo, the next one waits for a new pose: the same pose twice would add nothing.
 *
 * Not thread-safe: call from one thread.
 */
class FreePoseGuide {

    companion object {
        const val MAX_PHOTOS = CaptureGuide.VIEWS

        /** Amber lasts this long before the countdown: the time to say "hold still in this pose for three seconds". */
        const val HOLD_MS = 2600L
        const val COUNT_MS = 1000L
        const val COUNTS = 3

        /** Moving further than this share of the body's height counts as moving. */
        const val STILL_MOVE = 0.025f

        /**
         * A new pose: some point of the body moved at least this share of its height since the last photo,
         * or the body turned at least [NEW_POSE_TURN_DEG]. Turning hardly moves the points in the picture,
         * so it is measured on its own.
         */
        const val NEW_POSE_MOVE = 0.08f
        const val NEW_POSE_TURN_DEG = 25f

        private val TRACKED = intArrayOf(CaptureGuide.NOSE, CaptureGuide.LEFT_SHOULDER, CaptureGuide.RIGHT_SHOULDER,
            CaptureGuide.LEFT_WRIST, CaptureGuide.RIGHT_WRIST, CaptureGuide.LEFT_HIP, CaptureGuide.RIGHT_HIP,
            CaptureGuide.LEFT_ANKLE, CaptureGuide.RIGHT_ANKLE, 13, 14, 25, 26)
    }

    var taken = 0
        private set
    val done get() = taken >= MAX_PHOTOS

    private val anchor = FloatArray(TRACKED.size * 2)
    private var anchoredAtMs = -1L
    private var countStartMs = -1L
    private var awaitingPhoto = false
    private var lastPhoto: FloatArray? = null
    private var pending: FloatArray? = null
    private var lastTurn = Float.NaN
    private var pendingTurn = Float.NaN

    fun restart() {
        taken = 0
        anchoredAtMs = -1L
        countStartMs = -1L
        awaitingPhoto = false
        lastPhoto = null
        lastTurn = Float.NaN
    }

    fun onCaptured() {
        if (!awaitingPhoto) return
        awaitingPhoto = false
        countStartMs = -1L
        anchoredAtMs = -1L
        lastPhoto = pending
        lastTurn = pendingTurn
        taken++
    }

    fun onCaptureFailed() {
        awaitingPhoto = false
        countStartMs = -1L
        anchoredAtMs = -1L
    }

    fun update(image: FloatArray?, world: FloatArray?, people: Int, aspect: Float, objects: List<Spotted>, tooDark: Boolean, timeMs: Long): FreeGuidance {
        if (done || awaitingPhoto) return FreeGuidance(null, false, Level.READY, 0, false, taken)
        val verdict = PoseRules.check(image, world, people, aspect, objects, tooDark)
        if (verdict != null || image == null) return reset(verdict, false)

        val body = relative(image!!, aspect)
        val turn = turnDeg(world)
        val last = lastPhoto
        val turned = !turn.isNaN() && !lastTurn.isNaN() && kotlin.math.abs(CaptureGuide.wrap(turn - lastTurn)) >= NEW_POSE_TURN_DEG
        if (last != null && !turned && maxMove(body, last) < NEW_POSE_MOVE) return reset(null, true)

        val counting = countStartMs >= 0
        if (anchoredAtMs < 0 || maxMove(body, anchor) > STILL_MOVE * (if (counting) 1.6f else 1f)) {
            body.copyInto(anchor)
            anchoredAtMs = timeMs
            countStartMs = -1L
            return FreeGuidance(null, false, Level.NEARLY, 0, false, taken)
        }
        if (!counting) {
            if (timeMs - anchoredAtMs < HOLD_MS) return FreeGuidance(null, false, Level.NEARLY, 0, false, taken)
            countStartMs = timeMs
        }
        val elapsed = timeMs - countStartMs
        if (elapsed >= COUNT_MS * COUNTS) {
            awaitingPhoto = true
            pending = body
            pendingTurn = turn
            return FreeGuidance(null, false, Level.READY, 0, true, taken)
        }
        return FreeGuidance(null, false, Level.READY, COUNTS - (elapsed / COUNT_MS).toInt(), false, taken)
    }

    /** Which way the body faces, degrees, from the shoulders in 3D; NaN without world points. */
    private fun turnDeg(world: FloatArray?): Float {
        if (world == null || world.size < CaptureGuide.POINTS * 3) return Float.NaN
        val vx = world[CaptureGuide.LEFT_SHOULDER * 3] - world[CaptureGuide.RIGHT_SHOULDER * 3]
        val vz = (world[CaptureGuide.LEFT_SHOULDER * 3 + 2] - world[CaptureGuide.RIGHT_SHOULDER * 3 + 2]) * CaptureGuide.TURN_SIGN
        if (hypot(vx, vz) < CaptureGuide.MIN_SHOULDER_SPAN_M) return Float.NaN
        return Math.toDegrees(kotlin.math.atan2(vz.toDouble(), vx.toDouble())).toFloat()
    }

    private fun reset(verdict: PoseVerdict?, samePose: Boolean): FreeGuidance {
        anchoredAtMs = -1L
        countStartMs = -1L
        return FreeGuidance(verdict, samePose, Level.ADJUST, 0, false, taken)
    }

    /** The tracked points about the hips' middle, scaled by the body's height: stepping aside is not a new pose. */
    private fun relative(image: FloatArray, aspect: Float): FloatArray {
        val cx = (image[CaptureGuide.LEFT_HIP * 3] + image[CaptureGuide.RIGHT_HIP * 3]) / 2 * aspect
        val cy = (image[CaptureGuide.LEFT_HIP * 3 + 1] + image[CaptureGuide.RIGHT_HIP * 3 + 1]) / 2
        val top = image[CaptureGuide.NOSE * 3 + 1]
        val bottom = maxOf(image[CaptureGuide.LEFT_ANKLE * 3 + 1], image[CaptureGuide.RIGHT_ANKLE * 3 + 1])
        val height = (bottom - top).coerceAtLeast(0.05f)
        val out = FloatArray(TRACKED.size * 2)
        for ((k, p) in TRACKED.withIndex()) {
            out[k * 2] = (image[p * 3] * aspect - cx) / height
            out[k * 2 + 1] = (image[p * 3 + 1] - cy) / height
        }
        return out
    }

    private fun maxMove(a: FloatArray, b: FloatArray): Float {
        var most = 0f
        for (k in TRACKED.indices) most = maxOf(most, hypot(a[k * 2] - b[k * 2], a[k * 2 + 1] - b[k * 2 + 1]))
        return most
    }
}
