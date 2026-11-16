package com.shahryar.virtualtryon.phototryon

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.hypot

/** What is wrong with a pose for dressing, most basic first. */
enum class PoseProblem {
    NO_PERSON, TWO_PEOPLE, MOVE_AWAY, COME_CLOSER, NOT_STANDING, ARMS_ON_BODY, HANDS_HIDDEN, HOLDING, TOO_DARK
}

/** An object the object detector found: its name and box, x and y from 0 to 1 across the picture. */
class Spotted(val label: String, val left: Float, val top: Float, val right: Float, val bottom: Float)

/** A pose's first problem, and for [PoseProblem.HOLDING] the name of what is held. */
class PoseVerdict(val problem: PoseProblem, val detail: String? = null)

/**
 * The rules a free pose must keep for the dressing model to do well (Photo Try-On): standing, arms off
 * the body, hands in view, nothing held, alone, all of the body in the picture, enough light. Each frame
 * of the camera, or each uploaded photo, is checked and its first broken rule reported; null is a pose
 * fit to dress.
 *
 * Only the pose model's points and the object detector's boxes are used. Points are packed as
 * [CaptureGuide] takes them: x and y from 0 to 1, then visibility; world points in metres.
 */
object PoseRules {

    /** Knees must be at least this straight, degrees: standing is about 170, sitting about 90. */
    const val MIN_KNEE_DEG = 145f

    /** The hips must be this much higher than the knees, as a share of the body's height in the picture. */
    const val MIN_HIP_OVER_KNEE = 0.12f

    /** The body may lean at most this far from upright, degrees. */
    const val MAX_LEAN_DEG = 30f

    /** A hand point below this visibility is out of sight: in a pocket, behind the back. */
    const val HAND_SEEN = 0.3f

    /** Hands are checked only this near facing the camera or facing away; side-on the far hand is hidden anyway. */
    const val HANDS_CHECKED_WITHIN_DEG = 50f

    /** Things that should not be held or carried. */
    val HOLDABLE = setOf("cell phone", "handbag", "backpack", "suitcase", "bottle", "cup", "umbrella", "laptop", "book", "remote")

    private const val LEFT_ELBOW = 13
    private const val RIGHT_ELBOW = 14
    private const val LEFT_PINKY = 17
    private const val RIGHT_PINKY = 18
    private const val LEFT_INDEX = 19
    private const val RIGHT_INDEX = 20
    private const val LEFT_THUMB = 21
    private const val RIGHT_THUMB = 22
    private const val LEFT_KNEE = 25
    private const val RIGHT_KNEE = 26

    /**
     * The first rule [image] breaks, or null. [world] is the same pose in metres, for which way the body
     * faces; [people] how many the pose model found; [aspect] the picture's width over its height.
     */
    fun check(image: FloatArray?, world: FloatArray?, people: Int, aspect: Float, objects: List<Spotted>, tooDark: Boolean): PoseVerdict? {
        if (image == null || image.size < CaptureGuide.POINTS * 3 ||
            !seen(image, CaptureGuide.LEFT_SHOULDER) || !seen(image, CaptureGuide.RIGHT_SHOULDER) ||
            !seen(image, CaptureGuide.LEFT_HIP) || !seen(image, CaptureGuide.RIGHT_HIP)
        ) return PoseVerdict(PoseProblem.NO_PERSON)
        if (people >= 2) return PoseVerdict(PoseProblem.TWO_PEOPLE)

        // All of the body in the picture, at a size that dresses well.
        val noseY = y(image, CaptureGuide.NOSE)
        val shoulderY = (y(image, CaptureGuide.LEFT_SHOULDER) + y(image, CaptureGuide.RIGHT_SHOULDER)) / 2
        val hipY = (y(image, CaptureGuide.LEFT_HIP) + y(image, CaptureGuide.RIGHT_HIP)) / 2
        val headTop = noseY - 0.55f * (shoulderY - noseY)
        val feet = maxOf(y(image, CaptureGuide.LEFT_ANKLE), y(image, CaptureGuide.RIGHT_ANKLE),
            maxOf(y(image, CaptureGuide.LEFT_HEEL), y(image, CaptureGuide.RIGHT_HEEL)))
        val fill = feet - headTop
        val feetSeen = seen(image, CaptureGuide.LEFT_ANKLE) || seen(image, CaptureGuide.RIGHT_ANKLE)
        if (!feetSeen || headTop < CaptureGuide.EDGE_MARGIN || feet > 1f - CaptureGuide.EDGE_MARGIN || fill > CaptureGuide.MAX_FILL) {
            return PoseVerdict(PoseProblem.MOVE_AWAY)
        }
        if (fill < CaptureGuide.MIN_FILL) return PoseVerdict(PoseProblem.COME_CLOSER)

        // Standing: knees nearly straight, hips well above them, body upright.
        val kneeY = (y(image, LEFT_KNEE) + y(image, RIGHT_KNEE)) / 2
        val knees = minOf(angle(image, CaptureGuide.LEFT_HIP, LEFT_KNEE, CaptureGuide.LEFT_ANKLE, aspect),
            angle(image, CaptureGuide.RIGHT_HIP, RIGHT_KNEE, CaptureGuide.RIGHT_ANKLE, aspect))
        val shoulderX = (x(image, CaptureGuide.LEFT_SHOULDER) + x(image, CaptureGuide.RIGHT_SHOULDER)) / 2 * aspect
        val hipX = (x(image, CaptureGuide.LEFT_HIP) + x(image, CaptureGuide.RIGHT_HIP)) / 2 * aspect
        val lean = Math.toDegrees(atan2(abs(shoulderX - hipX).toDouble(), (hipY - shoulderY).toDouble())).toFloat()
        if (knees < MIN_KNEE_DEG || kneeY - hipY < MIN_HIP_OVER_KNEE * fill || hipY <= shoulderY || lean > MAX_LEAN_DEG) {
            return PoseVerdict(PoseProblem.NOT_STANDING)
        }

        // Arms off the body: neither wrist resting on the chest or stomach.
        val shoulderLeftX = minOf(x(image, CaptureGuide.LEFT_SHOULDER), x(image, CaptureGuide.RIGHT_SHOULDER))
        val shoulderRightX = maxOf(x(image, CaptureGuide.LEFT_SHOULDER), x(image, CaptureGuide.RIGHT_SHOULDER))
        val inset = 0.15f * (shoulderRightX - shoulderLeftX)
        val torso = hipY - shoulderY
        for (wrist in intArrayOf(CaptureGuide.LEFT_WRIST, CaptureGuide.RIGHT_WRIST)) {
            val wx = x(image, wrist)
            val wy = y(image, wrist)
            if (seen(image, wrist) && wx > shoulderLeftX + inset && wx < shoulderRightX - inset &&
                wy > shoulderY + 0.08f * torso && wy < hipY - 0.1f * torso
            ) return PoseVerdict(PoseProblem.ARMS_ON_BODY)
        }

        // Hands in view - not in pockets or behind the back. Only when both sides of the body are in view.
        val turn = turnDeg(world)
        val facingOrBack = turn.isNaN() || abs(CaptureGuide.wrap(turn)) <= HANDS_CHECKED_WITHIN_DEG ||
            abs(CaptureGuide.wrap(turn)) >= 180f - HANDS_CHECKED_WITHIN_DEG
        if (facingOrBack) {
            val leftHand = maxOf(vis(image, LEFT_PINKY), vis(image, LEFT_INDEX), vis(image, LEFT_THUMB))
            val rightHand = maxOf(vis(image, RIGHT_PINKY), vis(image, RIGHT_INDEX), vis(image, RIGHT_THUMB))
            if (leftHand < HAND_SEEN || rightHand < HAND_SEEN) return PoseVerdict(PoseProblem.HANDS_HIDDEN)
        }

        // Nothing held or carried: a phone, bag or the like on the person, not merely behind them.
        val visible = (0 until CaptureGuide.POINTS).filter { seen(image, it) }
        val bodyLeft = visible.minOf { x(image, it) }
        val bodyRight = visible.maxOf { x(image, it) }
        for (o in objects) {
            if (o.label !in HOLDABLE) continue
            val cx = (o.left + o.right) / 2
            val cy = (o.top + o.bottom) / 2
            val nearWrist = intArrayOf(CaptureGuide.LEFT_WRIST, CaptureGuide.RIGHT_WRIST).any { w ->
                hypot((x(image, w) - cx) * aspect, y(image, w) - cy) < maxOf((o.right - o.left) * aspect, o.bottom - o.top) + 0.05f
            }
            val onBody = cx in bodyLeft..bodyRight && cy in shoulderY..hipY
            if (nearWrist || onBody) return PoseVerdict(PoseProblem.HOLDING, o.label)
        }

        if (tooDark) return PoseVerdict(PoseProblem.TOO_DARK)
        return null
    }

    /** Which way the body faces, degrees, from the shoulders in 3D; NaN without world points. */
    private fun turnDeg(world: FloatArray?): Float {
        if (world == null || world.size < CaptureGuide.POINTS * 3) return Float.NaN
        val vx = world[CaptureGuide.LEFT_SHOULDER * 3] - world[CaptureGuide.RIGHT_SHOULDER * 3]
        val vz = (world[CaptureGuide.LEFT_SHOULDER * 3 + 2] - world[CaptureGuide.RIGHT_SHOULDER * 3 + 2]) * CaptureGuide.TURN_SIGN
        if (hypot(vx, vz) < CaptureGuide.MIN_SHOULDER_SPAN_M) return Float.NaN
        return Math.toDegrees(atan2(vz.toDouble(), vx.toDouble())).toFloat()
    }

    /** The angle at [b] between [a] and [c] as seen in the picture, degrees; 180 if a point is out of sight. */
    private fun angle(image: FloatArray, a: Int, b: Int, c: Int, aspect: Float): Float {
        if (!seen(image, a) || !seen(image, b) || !seen(image, c)) return 180f
        val ux = (x(image, a) - x(image, b)) * aspect
        val uy = y(image, a) - y(image, b)
        val vx = (x(image, c) - x(image, b)) * aspect
        val vy = y(image, c) - y(image, b)
        val cos = (ux * vx + uy * vy) / (hypot(ux, uy) * hypot(vx, vy)).coerceAtLeast(1e-6f)
        return Math.toDegrees(acos(cos.coerceIn(-1f, 1f).toDouble())).toFloat()
    }

    private fun x(image: FloatArray, p: Int) = image[p * 3]
    private fun y(image: FloatArray, p: Int) = image[p * 3 + 1]
    private fun vis(image: FloatArray, p: Int) = image[p * 3 + 2]
    private fun seen(image: FloatArray, p: Int) = image[p * 3 + 2] >= CaptureGuide.SEEN
}
