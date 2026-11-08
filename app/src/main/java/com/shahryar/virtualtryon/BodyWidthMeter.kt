package com.shahryar.virtualtryon

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.hypot

/**
 * Measures the wearer's torso width live, from the pose model's person mask.
 *
 * Garments are fitted to the body in front of the camera rather than tuned per garment: a fixed
 * widening right for one photo and one tester was wrong for the next shirt. This is the
 * measurement that fitting will use (pulled forward from M6, which reuses it for fit warnings).
 *
 * ## What is measured
 *
 * The torso is measured at [LEVELS]: lines across the body, placed as fractions of the way from
 * the shoulder line (0) to the hip line (1), and perpendicular to the line joining the two
 * midpoints, so they tilt with a leaning body. Along each line the meter walks outward from the
 * torso's centre line, one mask pixel at a time, until the mask says "not person". Only a few
 * hundred pixels are read per frame: on device unpacking the mask and measuring take 0.1 and
 * 0.3 ms, where reading the whole mask took 24 ms.
 *
 * ## Arms
 *
 * The mask does not know arm from torso, and an arm hanging against the side would be measured
 * as torso. Where an arm crosses a measuring line - shoulder to elbow to wrist, and on through
 * the hand to the fingertips - the walk on that side may go no further than the arm's inner edge,
 * taken as [ARM_HALF_WIDTH] inside its centre line. The hand counts as arm: hanging relaxed, it
 * rests against the hip, and without it a touching hand was measured as torso. Arm positions are used whether or not the model is confident of them: ignoring an arm it
 * is unsure of let the walk run straight through it, and that wrong width then lingered.
 *
 * Each side of each line ends one of three ways:
 * - **Edge found** before the arm: measured.
 * - **Blocked**: the walk reached the arm still on the mask, so the arm touches the side. The
 *   body reaches at least the arm's inner edge, anything further is behind the arm and will be
 *   under the garment's sleeve, so the inner edge is taken as the side - or the other side's
 *   measured width if that is wider, the torso being close to symmetric, but never past the
 *   arm's centre line.
 * - **Lost**: the walk ran off the frame, found a gap too near the centre to be the torso's
 *   edge, or ran to [MAX_REACH] still on the mask. The other side's width is used, if it has one.
 *
 * A side with nothing usable keeps its last width for [HOLD_FRAMES], then has none. Keeping it
 * indefinitely let a width from seconds earlier - measured as the wearer walked into place -
 * stick out past their arms.
 *
 * The wearer's own clothes are part of the mask, so a loose top measures wider than the body
 * underneath. For a try-on that is the right width: the garment should cover them.
 *
 * Widths are smoothed over time, because the mask's edge shifts by a pixel or two every frame
 * and a garment fitted to it would visibly breathe. Runs on the analysis thread only.
 */
class BodyWidthMeter {

    companion object {
        /** Measuring lines, as fractions of the way from the shoulder line to the hip line. */
        val LEVELS = floatArrayOf(0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f)

        /** Mask confidence at or above which a pixel is person. */
        const val PERSON = 0.5f

        /**
         * Half an upper arm's width, as a fraction of shoulder-joint width: about 0.12 on the
         * tester's M3.5 shots, where the arm measured roughly a quarter of the shoulder span.
         */
        const val ARM_HALF_WIDTH = 0.12f

        /**
         * How far the fingertips reach beyond the wrist, as a multiple of the distance from the
         * wrist to the pose model's two hand points (index and little finger). On device, with
         * the hand hanging relaxed, the fingertips were 2.8 times as far as those points; 2
         * stopped short and the fingers were measured as torso.
         */
        const val HAND_REACH = 3f

        /** Furthest a torso edge can be from the centre line, as a fraction of shoulder width. */
        const val MAX_REACH = 1.5f

        /** A half-width narrower than this fraction of shoulder width is a mask gap, not a torso. */
        const val MIN_HALF = 0.15f

        /** Exponential smoothing per frame: lower is steadier but slower to follow. */
        const val SMOOTHING = 0.3f

        /** Frames a side keeps its last width with nothing new: about half a second at 20 FPS. */
        const val HOLD_FRAMES = 10

        /**
         * Snapshot layout per level: centre x, y; left edge x, y; right edge x, y (normalised
         * image coordinates); then quality per side, left then right, packed as left * 3 + right.
         */
        const val STRIDE = 7

        /** Quality of one side of one line. */
        const val NONE = 0
        const val FILLED = 1
        const val MEASURED = 2

        /** [walk] results other than a found edge. */
        private const val LOST = -1f
        private const val BLOCKED = -2f

        private const val LEFT_SHOULDER = 11
        private const val RIGHT_SHOULDER = 12
        private const val LEFT_ELBOW = 13
        private const val RIGHT_ELBOW = 14
        private const val LEFT_WRIST = 15
        private const val RIGHT_WRIST = 16
        private const val LEFT_PINKY = 17
        private const val RIGHT_PINKY = 18
        private const val LEFT_INDEX = 19
        private const val RIGHT_INDEX = 20
        private const val LEFT_HIP = 23
        private const val RIGHT_HIP = 24
        private const val LEFT_KNEE = 25
        private const val RIGHT_KNEE = 26
        private const val LEFT_ANKLE = 27
        private const val RIGHT_ANKLE = 28
        private const val MIN_VISIBILITY = 0.5f

        /** Per arm: shoulder, elbow, wrist, little-finger knuckle, index knuckle. */
        /**
         * Where along each upper arm its thickness is measured, as fractions of the way from the
         * shoulder to the elbow. Away from the shoulder, where the arm joins the torso and the mask
         * cannot tell one from the other.
         */
        val ARM_SAMPLES = floatArrayOf(0.5f, 0.65f, 0.8f)

        /** Furthest an arm's edge can be from its centre line, as a fraction of shoulder width. */
        const val MAX_ARM_HALF = 0.5f

        /**
         * Arm snapshot layout per arm (left, then right): outer half-thickness, inner half-thickness
         * (away from and towards the body, as fractions of the mask's width, NaN if unknown); then
         * per sample the outer and inner edge points, x0, y0, x1, y1 (normalised image coordinates,
         * NaN where not measured).
         */
        val ARM_STRIDE = 2 + ARM_SAMPLES.size * 4

        private val ARMS = arrayOf(
            intArrayOf(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_PINKY, LEFT_INDEX),
            intArrayOf(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_PINKY, RIGHT_INDEX)
        )

        /**
         * Where along each leg its thickness is measured (M4): 0 at the hip, 1 at the knee, 2 at
         * the ankle, each segment divided evenly. Clear of the crotch, where the legs join the
         * body, and of the ankle, where the foot widens the mask.
         */
        val LEG_SAMPLES = floatArrayOf(0.4f, 0.7f, 1.0f, 1.3f, 1.6f, 1.85f)

        /** Furthest a leg's edge can be from its centre line, as a fraction of hip-joint spacing. */
        const val MAX_LEG_HALF = 1.0f

        /**
         * Half an arm's width, as a fraction of hip-joint spacing: [ARM_HALF_WIDTH] over the
         * spacing of the hip joints to the shoulder joints, about 0.55 on the tester. A leg's
         * outer walk stops this far short of an arm crossing it, as the torso's does.
         */
        const val ARM_HALF_PER_HIP = 0.22f

        /**
         * Leg snapshot layout per leg (left, then right), per sample: outer and inner
         * half-thickness (away from and towards the other leg, as fractions of the mask's width,
         * NaN if unknown); then the outer and inner edge points, x0, y0, x1, y1 (normalised image
         * coordinates, NaN where not measured this frame).
         */
        const val LEG_SAMPLE_STRIDE = 6
        val LEG_STRIDE = LEG_SAMPLES.size * LEG_SAMPLE_STRIDE

        /** Per leg: hip, knee, ankle. */
        private val LEGS = arrayOf(
            intArrayOf(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            intArrayOf(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        )
    }

    /** Smoothed half-widths per level, in mask pixels; NaN when there is none. */
    private val leftHalf = FloatArray(LEVELS.size) { Float.NaN }
    private val rightHalf = FloatArray(LEVELS.size) { Float.NaN }

    /** Frames since each side last had a usable width. */
    private val leftStale = IntArray(LEVELS.size)
    private val rightStale = IntArray(LEVELS.size)

    /** Smoothed arm half-thicknesses per arm, outer and inner, in mask pixels; NaN if unknown. */
    private val armOuter = FloatArray(2) { Float.NaN }
    private val armInner = FloatArray(2) { Float.NaN }
    private val armStale = IntArray(2)

    /**
     * The arms' thickness from the last [measure], laid out as [ARM_STRIDE] describes, or null.
     * Read on the analysis thread right after [measure], then handed to the overlay.
     */
    var lastArms: FloatArray? = null
        private set

    /** Smoothed leg half-thicknesses per leg per sample, outer and inner, in mask pixels. */
    private val legOuter = Array(2) { FloatArray(LEG_SAMPLES.size) { Float.NaN } }
    private val legInner = Array(2) { FloatArray(LEG_SAMPLES.size) { Float.NaN } }
    private val legStale = Array(2) { IntArray(LEG_SAMPLES.size) }
    private val legPoint = FloatArray(4)
    private val otherPoint = FloatArray(4)

    /**
     * The legs' thickness from the last [measure], laid out as [LEG_STRIDE] describes, or null.
     * Read on the analysis thread right after [measure], then handed to the overlay.
     */
    var lastLegs: FloatArray? = null
        private set

    /** Number of lines with both sides measured in the last frame, for the on-screen stats. */
    @Volatile var measuredLevels = 0
        private set

    /** Forgets every width; call when the person is lost, so a new one starts fresh. */
    fun reset() {
        leftHalf.fill(Float.NaN)
        rightHalf.fill(Float.NaN)
        leftStale.fill(0)
        rightStale.fill(0)
        armOuter.fill(Float.NaN)
        armInner.fill(Float.NaN)
        armStale.fill(0)
        lastArms = null
        for (side in 0..1) {
            legOuter[side].fill(Float.NaN)
            legInner[side].fill(Float.NaN)
            legStale[side].fill(0)
        }
        lastLegs = null
        measuredLevels = 0
    }

    /**
     * Measures one frame. [mask] holds one confidence per pixel, row by row, [width] x [height],
     * aligned with the image the landmarks came from. Returns a snapshot laid out as [STRIDE]
     * describes, or null when the torso landmarks are not visible. The legs are measured into
     * [lastLegs] first, whether or not the torso is in sight.
     */
    fun measure(mask: FloatBuffer, width: Int, height: Int, landmarks: List<NormalizedLandmark>): FloatArray? {
        if (landmarks.size <= RIGHT_HIP) return null
        lastLegs = measureLegs(mask, width, height, landmarks)
        for (i in intArrayOf(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)) {
            if (landmarks[i].visibility().orElse(1f) < MIN_VISIBILITY) return null
        }
        fun px(i: Int) = landmarks[i].x() * width
        fun py(i: Int) = landmarks[i].y() * height

        val shoulderX = (px(LEFT_SHOULDER) + px(RIGHT_SHOULDER)) / 2f
        val shoulderY = (py(LEFT_SHOULDER) + py(RIGHT_SHOULDER)) / 2f
        val hipX = (px(LEFT_HIP) + px(RIGHT_HIP)) / 2f
        val hipY = (py(LEFT_HIP) + py(RIGHT_HIP)) / 2f
        val jointWidth = hypot(px(RIGHT_SHOULDER) - px(LEFT_SHOULDER), py(RIGHT_SHOULDER) - py(LEFT_SHOULDER))
        val axisX = hipX - shoulderX
        val axisY = hipY - shoulderY
        val axisLength = hypot(axisX, axisY)
        if (jointWidth < 4f || axisLength < 4f) return null

        // Across the torso, perpendicular to its axis, pointing from the left shoulder towards
        // the right one.
        var ux = -axisY / axisLength
        var uy = axisX / axisLength
        if (ux * (px(RIGHT_SHOULDER) - px(LEFT_SHOULDER)) + uy * (py(RIGHT_SHOULDER) - py(LEFT_SHOULDER)) < 0f) {
            ux = -ux
            uy = -uy
        }

        val maxReach = MAX_REACH * jointWidth
        val armHalf = ARM_HALF_WIDTH * jointWidth
        val out = FloatArray(LEVELS.size * STRIDE)
        var measured = 0
        for (k in LEVELS.indices) {
            val cx = shoulderX + LEVELS[k] * axisX
            val cy = shoulderY + LEVELS[k] * axisY

            // Distance from the centre to the nearest arm on each side; infinite if none crosses.
            val armRight = nearestArm(cx, cy, ux, uy, 1f, landmarks, width, height)
            val armLeft = nearestArm(cx, cy, ux, uy, -1f, landmarks, width, height)
            val limitRight = minOf(maxReach, armRight - armHalf)
            val limitLeft = minOf(maxReach, armLeft - armHalf)
            var walkRight = walk(mask, width, height, cx, cy, ux, uy, limitRight, jointWidth)
            var walkLeft = walk(mask, width, height, cx, cy, -ux, -uy, limitLeft, jointWidth)
            // Still on the mask at MAX_REACH, with no arm there: not a torso edge.
            if (walkRight == BLOCKED && limitRight >= maxReach) walkRight = LOST
            if (walkLeft == BLOCKED && limitLeft >= maxReach) walkLeft = LOST

            var right = resolve(walkRight, limitRight, armRight, walkLeft)
            var left = resolve(walkLeft, limitLeft, armLeft, walkRight)
            var rightQuality = if (walkRight >= 0f) MEASURED else FILLED
            var leftQuality = if (walkLeft >= 0f) MEASURED else FILLED
            // A lost side copies the other side's width, measured or worked out.
            if (right.isNaN() && !left.isNaN()) right = left
            else if (left.isNaN() && !right.isNaN()) left = right
            if (walkRight >= 0f && walkLeft >= 0f) measured++

            rightQuality = update(rightHalf, rightStale, k, right, rightQuality)
            leftQuality = update(leftHalf, leftStale, k, left, leftQuality)

            val o = k * STRIDE
            out[o] = cx / width
            out[o + 1] = cy / height
            val l = leftHalf[k]
            val r = rightHalf[k]
            out[o + 2] = (if (l.isNaN()) cx else cx - ux * l) / width
            out[o + 3] = (if (l.isNaN()) cy else cy - uy * l) / height
            out[o + 4] = (if (r.isNaN()) cx else cx + ux * r) / width
            out[o + 5] = (if (r.isNaN()) cy else cy + uy * r) / height
            out[o + 6] = (leftQuality * 3 + rightQuality).toFloat()
        }
        measuredLevels = measured
        lastArms = measureArms(mask, width, height, landmarks, jointWidth)
        return out
    }

    /**
     * Measures each upper arm's thickness, clothes included: at [ARM_SAMPLES] along shoulder to
     * elbow, walks across the arm both ways from its centre line to the mask's edge. Garment
     * sleeves are fitted to this, rather than to an arm width assumed from the shoulder width,
     * which left the tester's arm - in a loose kameez sleeve, 0.4 shoulder widths thick against
     * the 0.24 assumed - showing below the sleeve.
     *
     * Outer and inner halves are kept apart, so the sleeve follows the arm's real edges even when
     * the pose model's arm line is not at the arm's middle. A sample where either walk runs into
     * the torso or off the frame is skipped; with every sample skipped - an arm against the side,
     * say - the last thickness is kept for [HOLD_FRAMES], then dropped.
     */
    private fun measureArms(
        mask: FloatBuffer, width: Int, height: Int, landmarks: List<NormalizedLandmark>, jointWidth: Float
    ): FloatArray {
        val out = FloatArray(2 * ARM_STRIDE) { Float.NaN }
        val limit = MAX_ARM_HALF * jointWidth
        val outers = FloatArray(ARM_SAMPLES.size)
        val inners = FloatArray(ARM_SAMPLES.size)
        for (side in 0..1) {
            val shoulder = if (side == 0) LEFT_SHOULDER else RIGHT_SHOULDER
            val elbow = if (side == 0) LEFT_ELBOW else RIGHT_ELBOW
            val hip = if (side == 0) LEFT_HIP else RIGHT_HIP
            var n = 0
            val o = side * ARM_STRIDE
            if (landmarks[shoulder].visibility().orElse(1f) >= MIN_VISIBILITY &&
                landmarks[elbow].visibility().orElse(1f) >= MIN_VISIBILITY) {
                val jx = landmarks[shoulder].x() * width
                val jy = landmarks[shoulder].y() * height
                val ex = landmarks[elbow].x() * width - jx
                val ey = landmarks[elbow].y() * height - jy
                val length = hypot(ex, ey)
                if (length >= 4f) {
                    // Across the arm, pointing away from the body (from the hip on this side).
                    var cx = -ey / length
                    var cy = ex / length
                    if ((landmarks[hip].x() * width - jx) * cx + (landmarks[hip].y() * height - jy) * cy > 0f) {
                        cx = -cx
                        cy = -cy
                    }
                    for ((i, f) in ARM_SAMPLES.withIndex()) {
                        val sx = jx + ex * f
                        val sy = jy + ey * f
                        val a = armWalk(mask, width, height, sx, sy, cx, cy, limit)
                        val b = armWalk(mask, width, height, sx, sy, -cx, -cy, limit)
                        if (a.isNaN() || b.isNaN()) continue
                        outers[n] = a
                        inners[n] = b
                        n++
                        val p = o + 2 + i * 4
                        out[p] = (sx + cx * a) / width
                        out[p + 1] = (sy + cy * a) / height
                        out[p + 2] = (sx - cx * b) / width
                        out[p + 3] = (sy - cy * b) / height
                    }
                }
            }
            val outer = if (n > 0) median(outers, n) else Float.NaN
            val inner = if (n > 0) median(inners, n) else Float.NaN
            if (n > 0) {
                armStale[side] = 0
                armOuter[side] = if (armOuter[side].isNaN()) outer else armOuter[side] + SMOOTHING * (outer - armOuter[side])
                armInner[side] = if (armInner[side].isNaN()) inner else armInner[side] + SMOOTHING * (inner - armInner[side])
            } else if (++armStale[side] > HOLD_FRAMES) {
                armOuter[side] = Float.NaN
                armInner[side] = Float.NaN
            }
            out[o] = armOuter[side] / width
            out[o + 1] = armInner[side] / width
        }
        return out
    }

    /**
     * Measures each leg's thickness, clothes included, for fitting trousers row by row
     * (OverlayView.FitMode.ROWS): at [LEG_SAMPLES] along hip to knee to ankle, walks across the
     * leg both ways from its centre line to the mask's edge, square to the thigh or the shin.
     *
     * Outer and inner halves are kept apart, as for the arms. The outer walk stops short of an
     * arm crossing it - a hand hanging beside the thigh is not leg - and a sample whose walk
     * reaches the arm is skipped. The inner walk goes no further than halfway to the other leg:
     * legs standing together meet there, and the mask cannot tell where one ends, so a walk still
     * on the mask at halfway is taken as that far. Each sample is smoothed and held on its own,
     * since the fit follows the leg's thickness down its length.
     */
    private fun measureLegs(
        mask: FloatBuffer, width: Int, height: Int, landmarks: List<NormalizedLandmark>
    ): FloatArray? {
        if (landmarks.size <= RIGHT_ANKLE) return null
        if (landmarks[LEFT_HIP].visibility().orElse(1f) < MIN_VISIBILITY ||
            landmarks[RIGHT_HIP].visibility().orElse(1f) < MIN_VISIBILITY) return null
        val hipSpan = hypot((landmarks[RIGHT_HIP].x() - landmarks[LEFT_HIP].x()) * width,
            (landmarks[RIGHT_HIP].y() - landmarks[LEFT_HIP].y()) * height)
        if (hipSpan < 4f) return null
        val maxHalf = MAX_LEG_HALF * hipSpan
        val armHalf = ARM_HALF_PER_HIP * hipSpan
        val out = FloatArray(2 * LEG_STRIDE) { Float.NaN }
        for (side in 0..1) {
            val other = 1 - side
            for ((i, t) in LEG_SAMPLES.withIndex()) {
                val p = side * LEG_STRIDE + i * LEG_SAMPLE_STRIDE
                var outer = Float.NaN
                var inner = Float.NaN
                if (legAt(side, t, landmarks, width, height, legPoint)) {
                    val sx = legPoint[0]
                    val sy = legPoint[1]
                    // Across the leg, pointing away from the other leg - or from its hip, if the
                    // other leg is out of sight.
                    val otherSeen = legAt(other, t, landmarks, width, height, otherPoint)
                    val rx = if (otherSeen) otherPoint[0] else landmarks[LEGS[other][0]].x() * width
                    val ry = if (otherSeen) otherPoint[1] else landmarks[LEGS[other][0]].y() * height
                    var cx = -legPoint[3]
                    var cy = legPoint[2]
                    if ((rx - sx) * cx + (ry - sy) * cy > 0f) {
                        cx = -cx
                        cy = -cy
                    }
                    val toOther = -((rx - sx) * cx + (ry - sy) * cy)
                    val outLimit = minOf(maxHalf, nearestArm(sx, sy, cx, cy, 1f, landmarks, width, height) - armHalf)
                    val inLimit = if (otherSeen) toOther / 2f else maxHalf
                    if (outLimit >= 1f && inLimit >= 1f) {
                        outer = armWalk(mask, width, height, sx, sy, cx, cy, outLimit)
                        inner = armWalk(mask, width, height, sx, sy, -cx, -cy, inLimit,
                            atLimit = if (otherSeen) inLimit else Float.NaN)
                    }
                    if (!outer.isNaN() && !inner.isNaN()) {
                        out[p + 2] = (sx + cx * outer) / width
                        out[p + 3] = (sy + cy * outer) / height
                        out[p + 4] = (sx - cx * inner) / width
                        out[p + 5] = (sy - cy * inner) / height
                    }
                }
                if (!outer.isNaN() && !inner.isNaN()) {
                    legStale[side][i] = 0
                    val o = legOuter[side][i]
                    val n = legInner[side][i]
                    legOuter[side][i] = if (o.isNaN()) outer else o + SMOOTHING * (outer - o)
                    legInner[side][i] = if (n.isNaN()) inner else n + SMOOTHING * (inner - n)
                } else if (++legStale[side][i] > HOLD_FRAMES) {
                    legOuter[side][i] = Float.NaN
                    legInner[side][i] = Float.NaN
                }
                out[p] = legOuter[side][i] / width
                out[p + 1] = legInner[side][i] / width
            }
        }
        return out
    }

    /**
     * Writes into [out] the point [t] along leg [side] - 0 at the hip, 1 at the knee, 2 at the
     * ankle - and the unit direction of the thigh or shin it lies on, in mask pixels. False when
     * either end of that segment is out of sight.
     */
    private fun legAt(
        side: Int, t: Float, landmarks: List<NormalizedLandmark>, width: Int, height: Int, out: FloatArray
    ): Boolean {
        val joints = LEGS[side]
        val a = if (t <= 1f) joints[0] else joints[1]
        val b = if (t <= 1f) joints[1] else joints[2]
        val f = if (t <= 1f) t else t - 1f
        if (landmarks[a].visibility().orElse(1f) < MIN_VISIBILITY ||
            landmarks[b].visibility().orElse(1f) < MIN_VISIBILITY) return false
        val ax = landmarks[a].x() * width
        val ay = landmarks[a].y() * height
        val dx = landmarks[b].x() * width - ax
        val dy = landmarks[b].y() * height - ay
        val length = hypot(dx, dy)
        if (length < 4f) return false
        out[0] = ax + f * dx
        out[1] = ay + f * dy
        out[2] = dx / length
        out[3] = dy / length
        return true
    }

    private fun median(values: FloatArray, n: Int): Float {
        java.util.Arrays.sort(values, 0, n)
        return values[n / 2]
    }

    /**
     * Distance from a point on the arm's centre line to the arm's edge along a direction, or NaN:
     * the point is not on the mask, or the walk reached the frame's edge, or it reached [limit] -
     * the torso, or a hand against the arm - which gives [atLimit] instead. Legs walk the same way.
     */
    private fun armWalk(
        mask: FloatBuffer, width: Int, height: Int, x0: Float, y0: Float, dx: Float, dy: Float, limit: Float,
        atLimit: Float = Float.NaN
    ): Float {
        var s = 0f
        while (s < limit) {
            val x = (x0 + dx * s).toInt()
            val y = (y0 + dy * s).toInt()
            if (x < 0 || y < 0 || x >= width || y >= height) return Float.NaN
            if (mask.get(y * width + x) < PERSON) return if (s >= 1f) s else Float.NaN
            s += 1f
        }
        return atLimit
    }

    /**
     * One side's width this frame from its own walk, or NaN if it has none of its own: the edge
     * if found; if blocked by the arm, the arm's inner edge, widened to the other side's measured
     * width but never past the arm's centre line; if lost, nothing.
     */
    private fun resolve(own: Float, limit: Float, arm: Float, other: Float): Float = when {
        own >= 0f -> own
        own == BLOCKED -> if (other > limit) minOf(other, arm) else limit
        else -> Float.NaN
    }

    /**
     * Folds this frame's width for side [k] into the smoothed one and returns the quality to draw.
     * With no width this frame, the last one is kept for [HOLD_FRAMES], then dropped.
     */
    private fun update(half: FloatArray, stale: IntArray, k: Int, value: Float, quality: Int): Int {
        if (!value.isNaN()) {
            stale[k] = 0
            half[k] = if (half[k].isNaN()) value else half[k] + SMOOTHING * (value - half[k])
            return quality
        }
        stale[k]++
        if (stale[k] > HOLD_FRAMES) half[k] = Float.NaN
        return if (half[k].isNaN()) NONE else FILLED
    }

    /**
     * Distance from the centre, along the line, to the nearest arm crossing it on one [side]
     * (+1 towards the right shoulder, -1 towards the left), or infinity if no arm does. Each arm
     * is three segments: upper arm, forearm, and hand from wrist to fingertips.
     */
    private fun nearestArm(
        cx: Float, cy: Float, ux: Float, uy: Float, side: Float,
        landmarks: List<NormalizedLandmark>, width: Int, height: Int
    ): Float {
        var nearest = Float.POSITIVE_INFINITY
        for (arm in ARMS) {
            val sx = landmarks[arm[0]].x() * width
            val sy = landmarks[arm[0]].y() * height
            val ex = landmarks[arm[1]].x() * width
            val ey = landmarks[arm[1]].y() * height
            val wx = landmarks[arm[2]].x() * width
            val wy = landmarks[arm[2]].y() * height
            val knucklesX = (landmarks[arm[3]].x() + landmarks[arm[4]].x()) / 2f * width
            val knucklesY = (landmarks[arm[3]].y() + landmarks[arm[4]].y()) / 2f * height
            val tipsX = wx + HAND_REACH * (knucklesX - wx)
            val tipsY = wy + HAND_REACH * (knucklesY - wy)
            nearest = minOf(
                nearest,
                crossing(cx, cy, ux, uy, side, sx, sy, ex, ey),
                minOf(crossing(cx, cy, ux, uy, side, ex, ey, wx, wy),
                    crossing(cx, cy, ux, uy, side, wx, wy, tipsX, tipsY))
            )
        }
        return nearest
    }

    /**
     * Distance from C along the line C + s*U to where the segment A-B crosses it, if it crosses
     * on [side]; infinity otherwise.
     */
    private fun crossing(
        cx: Float, cy: Float, ux: Float, uy: Float, side: Float,
        ax: Float, ay: Float, bx: Float, by: Float
    ): Float {
        val dx = bx - ax
        val dy = by - ay
        val denom = ux * dy - uy * dx
        if (abs(denom) < 1e-4f) return Float.POSITIVE_INFINITY
        val rx = ax - cx
        val ry = ay - cy
        val s = (rx * dy - ry * dx) / denom
        val t = (rx * uy - ry * ux) / denom
        return if (t < 0f || t > 1f || s * side <= 0f) Float.POSITIVE_INFINITY else abs(s)
    }

    /**
     * Distance from the centre to the first non-person pixel along a direction. [BLOCKED] if the
     * walk reached [limit] still on the mask; [LOST] if it ran off the frame, or found a gap too
     * close to the centre to be the torso's edge, or [limit] itself is that close.
     */
    private fun walk(
        mask: FloatBuffer, width: Int, height: Int,
        cx: Float, cy: Float, dx: Float, dy: Float, limit: Float, jointWidth: Float
    ): Float {
        val minimum = MIN_HALF * jointWidth
        if (limit <= minimum) return LOST
        var s = 0f
        while (s < limit) {
            val x = (cx + dx * s).toInt()
            val y = (cy + dy * s).toInt()
            if (x < 0 || y < 0 || x >= width || y >= height) return LOST
            if (mask.get(y * width + x) < PERSON) return if (s >= minimum) s else LOST
            s += 1f
        }
        return BLOCKED
    }
}
