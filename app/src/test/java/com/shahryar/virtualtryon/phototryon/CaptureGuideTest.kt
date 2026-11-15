package com.shahryar.virtualtryon.phototryon

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/**
 * Photo Try-On, stage 1: the guide's instructions for a wearer built from sums rather than filmed.
 * The stand-in body is one unit tall; it is turned, placed in the frame and handed to the guide as
 * the pose model would hand it over, image points and world points both.
 */
class CaptureGuideTest {

    private val aspect = 0.75f          // a 3:4 portrait frame
    private var clock = 0L

    /** One stand-in wearer; change a field to break one thing. */
    private class Body(
        var fill: Float = 0.75f,         // share of the frame's height, head to feet
        var centreX: Float = 0.5f,       // where the hips' middle is across the frame
        var yawDeg: Float = 0f,          // turned to their left
        var headYawDeg: Float = 0f,      // the head turned further than the body
        var armDeg: Float = 22f,         // each arm away from straight down
        var feetHalf: Float = 0.08f,     // each ankle from the centre line, in body heights
        var leanDeg: Float = 0f,
        var shiftY: Float = 0f,          // moved down the frame
    )

    /** A point in the body's own terms: [left] of the centre line, [up] from the floor, [forward] of it. */
    private class P(val left: Float, val up: Float, val forward: Float)

    private fun points(b: Body): Pair<FloatArray, FloatArray> {
        val image = FloatArray(CaptureGuide.POINTS * 3)
        val world = FloatArray(CaptureGuide.POINTS * 3)
        val arm = Math.toRadians(b.armDeg.toDouble())
        val wristOut = (0.31 * sin(arm)).toFloat()
        val wristDown = (0.31 * cos(arm)).toFloat()
        val body = mapOf(
            CaptureGuide.NOSE to P(0f, 0.93f, 0.09f),
            CaptureGuide.LEFT_EAR to P(0.07f, 0.93f, 0f),
            CaptureGuide.RIGHT_EAR to P(-0.07f, 0.93f, 0f),
            CaptureGuide.LEFT_SHOULDER to P(0.11f, 0.82f, 0f),
            CaptureGuide.RIGHT_SHOULDER to P(-0.11f, 0.82f, 0f),
            CaptureGuide.LEFT_WRIST to P(0.11f + wristOut, 0.82f - wristDown, 0f),
            CaptureGuide.RIGHT_WRIST to P(-0.11f - wristOut, 0.82f - wristDown, 0f),
            CaptureGuide.LEFT_HIP to P(0.07f, 0.52f, 0f),
            CaptureGuide.RIGHT_HIP to P(-0.07f, 0.52f, 0f),
            CaptureGuide.LEFT_ANKLE to P(b.feetHalf, 0.04f, 0f),
            CaptureGuide.RIGHT_ANKLE to P(-b.feetHalf, 0.04f, 0f),
            CaptureGuide.LEFT_HEEL to P(b.feetHalf, 0.01f, -0.03f),
            CaptureGuide.RIGHT_HEEL to P(-b.feetHalf, 0.01f, -0.03f),
            CaptureGuide.LEFT_TOE to P(b.feetHalf, 0f, 0.10f),
            CaptureGuide.RIGHT_TOE to P(-b.feetHalf, 0f, 0.10f),
        )
        val yaw = Math.toRadians(b.yawDeg.toDouble())
        val headYaw = Math.toRadians((b.yawDeg + b.headYawDeg).toDouble())
        val lean = Math.tan(Math.toRadians(b.leanDeg.toDouble())).toFloat()
        val top = (1f - b.fill) / 2 + b.shiftY
        for ((index, p) in body) {
            // The head turns about its own upright axis; everything else with the body.
            val a = if (index == CaptureGuide.NOSE || index == CaptureGuide.LEFT_EAR || index == CaptureGuide.RIGHT_EAR) headYaw else yaw
            // Turning to their left swings the left shoulder away from the camera: left = (cos, sin), forward = (sin, -cos).
            val wx = (p.left * cos(a) + p.forward * sin(a)).toFloat() + lean * (p.up - 0.52f).coerceAtLeast(0f)
            val wz = (p.left * sin(a) - p.forward * cos(a)).toFloat()
            world[index * 3] = wx
            world[index * 3 + 1] = -p.up
            world[index * 3 + 2] = wz
            image[index * 3] = b.centreX + wx * b.fill / aspect
            image[index * 3 + 1] = top + (1f - p.up) * b.fill
            image[index * 3 + 2] = 1f
        }
        return image to world
    }

    private fun CaptureGuide.see(b: Body, stepMs: Long = 50): Guidance {
        clock += stepMs
        val (image, world) = points(b)
        return update(image, world, aspect, clock)
    }

    /** Shows [b] to the guide until it asks for the photo, or for [limitMs]; returns the last verdict. */
    private fun CaptureGuide.holdUntilPhoto(b: Body, limitMs: Long = 6000): Guidance {
        var g = see(b)
        var waited = 0L
        while (!g.capture && waited < limitMs) {
            g = see(b)
            waited += 50
        }
        return g
    }

    /** A guide whose front photo has been taken: the wearer is about to turn. */
    private fun turningGuide(): CaptureGuide {
        val guide = CaptureGuide()
        assertTrue(guide.holdUntilPhoto(Body()).capture)
        guide.onCaptured(clock)
        assertTrue(guide.turning)
        return guide
    }

    /** Turns the wearer [stepDeg] a frame for [frames] frames; returns the views taken, in order, and the prompts heard. */
    private fun CaptureGuide.turn(stepDeg: Float, frames: Int, from: Float = 0f, body: (Float) -> Body = { Body(yawDeg = it) }): Pair<List<Int>, Set<Prompt>> {
        val grabbed = ArrayList<Int>()
        val prompts = HashSet<Prompt>()
        for (k in 1..frames) {
            val g = see(body(from + stepDeg * k))
            if (g.grab >= 0) grabbed += g.grab
            prompts += g.prompt
        }
        return grabbed to prompts
    }

    // ------------------------------------------------------------------------------ the front photo

    @Test
    fun nobodyInViewSaysSo() {
        val g = CaptureGuide().update(null, null, aspect, 0)
        assertEquals(Prompt.NO_PERSON, g.prompt)
        assertEquals(Level.ADJUST, g.level)
    }

    @Test
    fun aGoodPoseHeldStillIsCountedDownAndTaken() {
        val guide = CaptureGuide()
        val b = Body()
        val counts = ArrayList<Int>()
        var g = guide.see(b)
        assertEquals(Prompt.HOLD_STILL, g.prompt)
        assertEquals(Level.NEARLY, g.level)
        while (!g.capture) {
            g = guide.see(b)
            if (g.count > 0 && (counts.isEmpty() || counts.last() != g.count)) counts += g.count
            assertTrue("took too long", clock < 10_000)
        }
        assertEquals(listOf(3, 2, 1), counts)
        // The photo is asked for once only, and nothing more happens until it is saved.
        assertFalse(guide.see(b).capture)
        assertFalse(guide.turning)
        guide.onCaptured(clock)
        assertTrue(guide.turning)
        assertEquals(1, guide.taken)
    }

    @Test
    fun tooCloseOrCutOffMeansMoveAway() {
        assertEquals(Prompt.MOVE_AWAY, CaptureGuide().see(Body(fill = 0.95f)).prompt)
        // Feet below the bottom edge, though the body is a fair size.
        assertEquals(Prompt.MOVE_AWAY, CaptureGuide().see(Body(fill = 0.8f, shiftY = 0.12f)).prompt)
    }

    @Test
    fun tooFarMeansComeCloser() {
        assertEquals(Prompt.COME_CLOSER, CaptureGuide().see(Body(fill = 0.4f)).prompt)
    }

    @Test
    fun offCentreIsCorrectedInTheWearersOwnTerms() {
        // Facing the phone, the wearer's left is the picture's right: standing at the picture's left, move left.
        assertEquals(Prompt.MOVE_LEFT, CaptureGuide().see(Body(centreX = 0.3f)).prompt)
        assertEquals(Prompt.MOVE_RIGHT, CaptureGuide().see(Body(centreX = 0.7f)).prompt)
    }

    @Test
    fun theFrontPhotoNeedsBodyAndHeadFacingThePhone() {
        assertEquals(Prompt.FACE_PHONE, CaptureGuide().see(Body(yawDeg = 40f)).prompt)
        assertEquals(Prompt.LOOK_AHEAD, CaptureGuide().see(Body(headYawDeg = 60f)).prompt)
    }

    @Test
    fun leaningArmsAndFeet() {
        assertEquals(Prompt.STAND_STRAIGHT, CaptureGuide().see(Body(leanDeg = 14f)).prompt)
        assertEquals(Prompt.ARMS_OUT, CaptureGuide().see(Body(armDeg = 5f)).prompt)
        assertEquals(Prompt.ARMS_DOWN, CaptureGuide().see(Body(armDeg = 70f)).prompt)
        assertEquals(Prompt.FEET_APART, CaptureGuide().see(Body(feetHalf = 0.02f)).prompt)
        assertEquals(Prompt.FEET_CLOSER, CaptureGuide().see(Body(feetHalf = 0.2f)).prompt)
    }

    @Test
    fun movingDuringTheCountdownCancelsIt() {
        val guide = CaptureGuide()
        val b = Body()
        var g = guide.see(b)
        while (g.count == 0) g = guide.see(b)
        assertEquals(3, g.count)
        // A step to one side.
        g = guide.see(Body(centreX = 0.56f))
        assertEquals(0, g.count)
        assertFalse(g.capture)
        assertEquals(Prompt.HOLD_STILL, g.prompt)
        // Standing still again starts the count from three.
        while (g.count == 0) g = guide.see(Body(centreX = 0.56f))
        assertEquals(3, g.count)
    }

    @Test
    fun aFailedPhotoIsAskedForAgain() {
        val guide = CaptureGuide()
        assertTrue(guide.holdUntilPhoto(Body()).capture)
        guide.onCaptureFailed()
        assertFalse(guide.turning)
        assertTrue(guide.holdUntilPhoto(Body()).capture)
    }

    // ------------------------------------------------------------------------------ the turn

    @Test
    fun aSlowTurnToTheLeftTakesEveryViewOnceInOrder() {
        val guide = turningGuide()
        // Two degrees a frame, twenty frames a second: a full turn in nine seconds.
        val (grabbed, prompts) = guide.turn(2f, 185)
        assertEquals(listOf(1, 2, 3, 4, 5, 6, 7), grabbed)
        assertTrue(guide.done)
        assertEquals(CaptureGuide.ALL_TAKEN, guide.taken)
        assertTrue(Prompt.TURN_SLOWLY in prompts)
        assertFalse(Prompt.SLOWER in prompts)
        assertEquals(Prompt.DONE, guide.see(Body()).prompt)
    }

    @Test
    fun turningToTheRightTakesTheSameViewsTheOtherWayRound() {
        val guide = turningGuide()
        val (grabbed, _) = guide.turn(-2f, 185)
        assertEquals(listOf(7, 6, 5, 4, 3, 2, 1), grabbed)
        assertTrue(guide.done)
    }

    @Test
    fun aViewIsTakenCloseToItsAngle() {
        val guide = turningGuide()
        var g = guide.see(Body(yawDeg = 0f))
        var yaw = 0f
        while (g.grab != 2) {
            yaw += 2f
            g = guide.see(Body(yawDeg = yaw))
            assertTrue("never took the side view", yaw < 120f)
        }
        // The body itself, not just the smoothed reading, is within a few degrees of side-on.
        assertEquals(90f, yaw, 8f)
    }

    @Test
    fun turningTooFastTakesNothingAndSaysSo() {
        val guide = turningGuide()
        // Six degrees a frame is 120 degrees a second.
        val (grabbed, prompts) = guide.turn(6f, 60)
        assertTrue(grabbed.isEmpty())
        assertTrue(Prompt.SLOWER in prompts)
        assertFalse(guide.done)
    }

    @Test
    fun aViewMissedByGoingTooFastIsCaughtOnTheNextTimeRound() {
        val guide = turningGuide()
        guide.turn(6f, 60)                                   // a whole turn, too fast
        val (grabbed, _) = guide.turn(2f, 185)               // and another, slowly
        assertEquals(listOf(1, 2, 3, 4, 5, 6, 7), grabbed)
        assertTrue(guide.done)
    }

    @Test
    fun outOfTheFrameDuringTheTurnTakesNothing() {
        val guide = turningGuide()
        // Walking towards the phone while turning: the body no longer fits.
        val (grabbed, prompts) = guide.turn(2f, 60) { Body(yawDeg = it, fill = 0.95f) }
        assertTrue(grabbed.isEmpty())
        assertTrue(Prompt.MOVE_AWAY in prompts)
    }

    @Test
    fun stoppingPartWayRoundIsToldToKeepTurning() {
        val guide = turningGuide()
        guide.turn(2f, 30)                                   // to 60 degrees: the first view is taken
        var g = guide.see(Body(yawDeg = 60f))
        repeat(120) { g = guide.see(Body(yawDeg = 60f)) }    // six seconds standing there
        assertEquals(Prompt.KEEP_TURNING, g.prompt)
        assertEquals(Level.NEARLY, g.level)
    }

    @Test
    fun wrapKeepsAnglesBetweenMinusAndPlus180() {
        assertEquals(-90f, CaptureGuide.wrap(270f), 1e-3f)
        assertEquals(90f, CaptureGuide.wrap(-270f), 1e-3f)
        assertEquals(170f, CaptureGuide.wrap(-190f), 1e-3f)
    }
}
