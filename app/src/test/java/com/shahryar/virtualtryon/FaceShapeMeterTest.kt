package com.shahryar.virtualtryon

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/**
 * Face-shape measuring (M7). The face used is MediaPipe's canonical face - its ten points that
 * the meter reads, in centimetres, image-like (+y down), as read from face_landmarker.task.
 */
class FaceShapeMeterTest {

    private val canonical = mapOf(
        10 to floatArrayOf(0f, -8.262f, 4.482f),
        152 to floatArrayOf(0f, 9.403f, 4.264f),
        234 to floatArrayOf(-7.664f, -0.673f, -2.436f),
        454 to floatArrayOf(7.664f, -0.673f, -2.436f),
        54 to floatArrayOf(-6.279f, -6.615f, 1.426f),
        284 to floatArrayOf(6.279f, -6.615f, 1.426f),
        172 to floatArrayOf(-5.941f, 6.224f, -0.631f),
        397 to floatArrayOf(5.941f, 6.224f, -0.631f),
        33 to floatArrayOf(-4.446f, -2.664f, 3.173f),
        263 to floatArrayOf(4.446f, -2.664f, 3.173f),
    )

    /**
     * The canonical face as Face Mesh would report it on a 1000 x 1000 image, 1 cm to 10 px,
     * turned [turnDeg] about the vertical axis and stretched [stretch] times in height.
     */
    private fun face(turnDeg: Float = 0f, stretch: Float = 1f): FloatArray {
        val out = FloatArray(FaceTracker.POINTS * 3)
        val a = Math.toRadians(turnDeg.toDouble())
        for ((i, p) in canonical) {
            val x = (p[0] * cos(a) + p[2] * sin(a)).toFloat()
            val z = (-p[0] * sin(a) + p[2] * cos(a)).toFloat()
            out[i * 3] = (500f + x * 10f) / 1000f
            out[i * 3 + 1] = (500f + p[1] * stretch * 10f) / 1000f
            out[i * 3 + 2] = z * 10f / 1000f
        }
        return out
    }

    @Test
    fun theCanonicalFaceMeasuresAsItsOwnProportions() {
        val p = FaceShapeMeter.proportions(face(), 1000, 1000)
        assertNotNull(p)
        assertEquals(FaceShapeMeter.CANONICAL_LENGTH, p!!.lengthToWidth, 0.002f)
        assertEquals(FaceShapeMeter.CANONICAL_FOREHEAD, p.foreheadToWidth, 0.002f)
        assertEquals(FaceShapeMeter.CANONICAL_JAW, p.jawToWidth, 0.002f)
        assertEquals(FaceShapeMeter.CANONICAL_JAW_ANGLE, p.jawAngleDeg, 0.2f)
    }

    @Test
    fun theAverageFaceIsOval() {
        assertEquals(FaceShape.OVAL, FaceShapeMeter.classify(FaceShapeMeter.proportions(face(), 1000, 1000)!!))
    }

    @Test
    fun aSlightTurnHardlyChangesTheProportions() {
        val straight = FaceShapeMeter.proportions(face(), 1000, 1000)!!
        val turned = FaceShapeMeter.proportions(face(turnDeg = 8f), 1000, 1000)!!
        // Measured in 3D, so a turn under the limit changes nothing.
        assertEquals(straight.lengthToWidth, turned.lengthToWidth, 0.001f)
        assertEquals(straight.jawToWidth, turned.jawToWidth, 0.001f)
    }

    @Test
    fun aTurnedFaceIsNotMeasured() {
        assertNull(FaceShapeMeter.proportions(face(turnDeg = 20f), 1000, 1000))
        assertNull(FaceShapeMeter.proportions(FloatArray(10), 1000, 1000))
    }

    @Test
    fun aLongFaceIsOblong() {
        val p = FaceShapeMeter.proportions(face(stretch = 1.15f), 1000, 1000)!!
        assertTrue(p.lengthToWidth >= FaceShapeMeter.LONG)
        assertEquals(FaceShape.OBLONG, FaceShapeMeter.classify(p))
    }

    @Test
    fun eachShapeHasItsProportions() {
        fun shape(length: Float, forehead: Float, jaw: Float, angle: Float) =
            FaceShapeMeter.classify(FaceProportions(length, forehead, jaw, angle))
        assertEquals(FaceShape.OVAL, shape(1.15f, 0.82f, 0.78f, 132f))
        assertEquals(FaceShape.OBLONG, shape(1.30f, 0.82f, 0.78f, 132f))
        assertEquals(FaceShape.HEART, shape(1.15f, 0.86f, 0.70f, 134f))
        assertEquals(FaceShape.ROUND, shape(1.04f, 0.82f, 0.78f, 138f))
        assertEquals(FaceShape.SQUARE, shape(1.04f, 0.84f, 0.86f, 124f))
        // Short with sharp corners, though the jaw is average: square.
        assertEquals(FaceShape.SQUARE, shape(1.04f, 0.82f, 0.78f, 125f))
        // Average length with a wide, sharp jaw: square.
        assertEquals(FaceShape.SQUARE, shape(1.15f, 0.84f, 0.88f, 124f))
        // A wide jaw with soft corners stays oval.
        assertEquals(FaceShape.OVAL, shape(1.15f, 0.84f, 0.88f, 134f))
    }

    @Test
    fun aReadingIsTheMedianOfItsFrames() {
        val meter = FaceShapeMeter(framesNeeded = 5)
        assertNull(meter.result())
        repeat(4) { assertTrue(meter.add(face(), 1000, 1000)) }
        // A tracker glitch stretching one frame does not move the median.
        assertTrue(meter.add(face(stretch = 1.4f), 1000, 1000))
        assertTrue(meter.done)
        assertFalse(meter.add(face(), 1000, 1000))
        val reading = meter.result()!!
        assertEquals(FaceShape.OVAL, reading.shape)
        assertEquals(FaceShapeMeter.CANONICAL_LENGTH, reading.proportions.lengthToWidth, 0.002f)
    }

    @Test
    fun turnedFramesAreSkippedAndResetStartsAgain() {
        val meter = FaceShapeMeter(framesNeeded = 3)
        assertFalse(meter.add(face(turnDeg = 25f), 1000, 1000))
        assertEquals(0, meter.count)
        meter.add(face(), 1000, 1000)
        assertEquals(1, meter.count)
        meter.reset()
        assertEquals(0, meter.count)
    }
}
