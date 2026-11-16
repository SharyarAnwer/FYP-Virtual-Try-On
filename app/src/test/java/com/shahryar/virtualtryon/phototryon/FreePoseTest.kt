package com.shahryar.virtualtryon.phototryon

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/** Free-pose mode: the rules a pose must keep, and the hold-still-then-count flow, on stand-in bodies. */
class FreePoseTest {

    private val aspect = 0.75f
    private var clock = 0L

    private class Body(
        var yawDeg: Float = 0f,
        var armDeg: Float = 22f,            // each arm away from straight down
        var handVis: Float = 1f,            // visibility of the hand points
        var kneeBend: Float = 0f,           // 0 standing, 1 sitting
        var wristOnChest: Boolean = false,
        var fill: Float = 0.75f,
    )

    private class P(val left: Float, val up: Float, val forward: Float, val vis: Float = 1f)

    private fun points(b: Body): Pair<FloatArray, FloatArray> {
        val image = FloatArray(CaptureGuide.POINTS * 3)
        val world = FloatArray(CaptureGuide.POINTS * 3)
        val arm = Math.toRadians(b.armDeg.toDouble())
        val wOut = (0.31 * sin(arm)).toFloat()
        val wDown = (0.31 * cos(arm)).toFloat()
        // Sitting: hips drop to knee height and the thighs point forward.
        val hipUp = 0.52f - 0.24f * b.kneeBend
        val kneeForward = 0.24f * b.kneeBend
        val lw = if (b.wristOnChest) P(0.02f, 0.70f, 0.1f) else P(0.11f + wOut, 0.82f - wDown, 0f)
        val rw = P(-0.11f - wOut, 0.82f - wDown, 0f)
        val parts = mapOf(
            CaptureGuide.NOSE to P(0f, 0.93f, 0.09f),
            CaptureGuide.LEFT_EAR to P(0.07f, 0.93f, 0f), CaptureGuide.RIGHT_EAR to P(-0.07f, 0.93f, 0f),
            CaptureGuide.LEFT_SHOULDER to P(0.11f, 0.82f, 0f), CaptureGuide.RIGHT_SHOULDER to P(-0.11f, 0.82f, 0f),
            13 to P(0.13f, 0.66f, 0f), 14 to P(-0.13f, 0.66f, 0f),
            CaptureGuide.LEFT_WRIST to lw, CaptureGuide.RIGHT_WRIST to rw,
            17 to P(lw.left, lw.up - 0.03f, lw.forward, b.handVis), 19 to P(lw.left, lw.up - 0.04f, lw.forward, b.handVis),
            21 to P(lw.left, lw.up - 0.02f, lw.forward, b.handVis),
            18 to P(rw.left, rw.up - 0.03f, 0f, b.handVis), 20 to P(rw.left, rw.up - 0.04f, 0f, b.handVis),
            22 to P(rw.left, rw.up - 0.02f, 0f, b.handVis),
            CaptureGuide.LEFT_HIP to P(0.07f, hipUp, 0f), CaptureGuide.RIGHT_HIP to P(-0.07f, hipUp, 0f),
            25 to P(0.07f, 0.28f, kneeForward), 26 to P(-0.07f, 0.28f, kneeForward),
            CaptureGuide.LEFT_ANKLE to P(0.08f, 0.04f, 0f), CaptureGuide.RIGHT_ANKLE to P(-0.08f, 0.04f, 0f),
            CaptureGuide.LEFT_HEEL to P(0.08f, 0.01f, -0.03f), CaptureGuide.RIGHT_HEEL to P(-0.08f, 0.01f, -0.03f),
        )
        val yaw = Math.toRadians(b.yawDeg.toDouble())
        val top = (1f - b.fill) / 2
        for ((i, p) in parts) {
            val wx = (p.left * cos(yaw) + p.forward * sin(yaw)).toFloat()
            val wz = (p.left * sin(yaw) - p.forward * cos(yaw)).toFloat()
            world[i * 3] = wx; world[i * 3 + 1] = -p.up; world[i * 3 + 2] = wz
            image[i * 3] = 0.5f + wx * b.fill / aspect
            // Sitting is seen from the front: the thighs foreshorten, so the knee sits near the hips in the picture.
            image[i * 3 + 1] = top + (1f - p.up) * b.fill
            image[i * 3 + 2] = p.vis
        }
        return image to world
    }

    private fun check(b: Body, people: Int = 1, objects: List<Spotted> = emptyList(), dark: Boolean = false): PoseVerdict? {
        val (image, world) = points(b)
        return PoseRules.check(image, world, people, aspect, objects, dark)
    }

    // ------------------------------------------------------------------------------ the rules

    @Test
    fun aRelaxedStandingPoseKeepsEveryRule() {
        assertNull(check(Body()))
        assertNull(check(Body(yawDeg = 90f)))
        assertNull(check(Body(armDeg = 40f)))
    }

    @Test
    fun nobodyAndTwoPeople() {
        assertEquals(PoseProblem.NO_PERSON, PoseRules.check(null, null, 0, aspect, emptyList(), false)!!.problem)
        assertEquals(PoseProblem.TWO_PEOPLE, check(Body(), people = 2)!!.problem)
    }

    @Test
    fun sittingIsNotStanding() {
        assertEquals(PoseProblem.NOT_STANDING, check(Body(kneeBend = 1f))!!.problem)
    }

    @Test
    fun aHandOnTheChestIsCaught() {
        assertEquals(PoseProblem.ARMS_ON_BODY, check(Body(wristOnChest = true))!!.problem)
    }

    @Test
    fun hiddenHandsAreCaughtFacingTheCameraButNotSideOn() {
        assertEquals(PoseProblem.HANDS_HIDDEN, check(Body(handVis = 0.1f))!!.problem)
        assertEquals(PoseProblem.HANDS_HIDDEN, check(Body(handVis = 0.1f, yawDeg = 180f))!!.problem)
        // Side-on the far hand is behind the body anyway: not held against the wearer.
        assertNull(check(Body(handVis = 0.1f, yawDeg = 90f)))
    }

    @Test
    fun aPhoneInTheHandIsCaughtButNotOneOnATableBehind() {
        val (image, _) = points(Body())
        val wx = image[CaptureGuide.RIGHT_WRIST * 3]
        val wy = image[CaptureGuide.RIGHT_WRIST * 3 + 1]
        val inHand = Spotted("cell phone", wx - 0.02f, wy - 0.03f, wx + 0.02f, wy + 0.03f)
        val verdict = check(Body(), objects = listOf(inHand))!!
        assertEquals(PoseProblem.HOLDING, verdict.problem)
        assertEquals("cell phone", verdict.detail)
        assertNull(check(Body(), objects = listOf(Spotted("cell phone", 0.02f, 0.1f, 0.06f, 0.15f))))
        // Something that is not carried, beside the wearer, is no problem.
        assertNull(check(Body(), objects = listOf(Spotted("chair", wx - 0.1f, wy - 0.1f, wx + 0.1f, wy + 0.1f))))
    }

    @Test
    fun tooDarkComesLast() {
        assertEquals(PoseProblem.TOO_DARK, check(Body(), dark = true)!!.problem)
    }

    @Test
    fun outOfTheFrameComesBeforeThePose() {
        assertEquals(PoseProblem.MOVE_AWAY, check(Body(fill = 0.97f, kneeBend = 1f))!!.problem)
        assertEquals(PoseProblem.COME_CLOSER, check(Body(fill = 0.4f))!!.problem)
    }

    // ------------------------------------------------------------------------------ the flow

    private fun FreePoseGuide.see(b: Body, stepMs: Long = 50): FreeGuidance {
        clock += stepMs
        val (image, world) = points(b)
        return update(image, world, 1, aspect, emptyList(), false, clock)
    }

    @Test
    fun aKeptPoseIsHeldAmberThenCountedDownAndTaken() {
        val guide = FreePoseGuide()
        var g = guide.see(Body())
        assertEquals(Level.NEARLY, g.level)
        val start = clock
        while (g.count == 0 && !g.capture) g = guide.see(Body())
        // Amber long enough for "hold still in this pose for three seconds".
        assertTrue(clock - start >= FreePoseGuide.HOLD_MS)
        assertEquals(3, g.count)
        while (!g.capture) g = guide.see(Body())
        guide.onCaptured()
        assertEquals(1, guide.taken)
    }

    @Test
    fun theSamePoseTwiceIsNotTakenTwice() {
        val guide = FreePoseGuide()
        var g = guide.see(Body())
        while (!g.capture) g = guide.see(Body())
        guide.onCaptured()
        repeat(100) { g = guide.see(Body()) }
        assertTrue(g.samePose)
        assertFalse(g.capture)
        // A new pose - arms raised - starts the next photo.
        g = guide.see(Body(armDeg = 40f, yawDeg = 45f))
        assertFalse(g.samePose)
        assertEquals(Level.NEARLY, g.level)
    }

    @Test
    fun breakingARuleDuringTheCountdownStopsIt() {
        val guide = FreePoseGuide()
        var g = guide.see(Body())
        while (g.count == 0) g = guide.see(Body())
        g = guide.see(Body(handVis = 0.1f))
        assertEquals(PoseProblem.HANDS_HIDDEN, g.verdict!!.problem)
        assertEquals(0, g.count)
        assertEquals(Level.ADJUST, g.level)
    }

    @Test
    fun eightPhotosAndItIsDone() {
        val guide = FreePoseGuide()
        for (k in 0 until FreePoseGuide.MAX_PHOTOS) {
            val pose = Body(yawDeg = k * 45f, armDeg = if (k % 2 == 0) 20f else 38f)
            var g = guide.see(pose)
            var guard = 0
            while (!g.capture) { g = guide.see(pose); assertTrue("photo $k never taken", ++guard < 400) }
            guide.onCaptured()
        }
        assertTrue(guide.done)
    }
}
