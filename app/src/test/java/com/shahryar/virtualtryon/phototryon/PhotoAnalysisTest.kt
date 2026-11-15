package com.shahryar.virtualtryon.phototryon

import com.shahryar.virtualtryon.BodyType
import com.shahryar.virtualtryon.Garment
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Photo Try-On, stages 2 and 3: measuring from silhouettes, reading skin tone, ranking outfits. */
class PhotoAnalysisTest {

    // ------------------------------------------------------------------------------ measuring

    private val w = 300
    private val h = 400
    private val aspect = w.toFloat() / h

    /** A body made of boxes: head, a torso [torsoPx] wide, two legs; arms as separate boxes if [arms]. */
    private fun body(torsoPx: Int, arms: Boolean = false, top: Int = 40, bottom: Int = 359): Silhouette {
        val on = BooleanArray(w * h)
        fun box(x0: Int, x1: Int, y0: Int, y1: Int) { for (y in y0..y1) for (x in x0..x1) on[y * w + x] = true }
        box(140, 159, top, 79)                                       // head
        box(150 - torsoPx / 2, 149 + torsoPx / 2, 80, 219)           // torso
        box(125, 145, 220, bottom); box(154, 174, 220, bottom)       // legs
        if (arms) { box(60, 75, 85, 200); box(224, 239, 85, 200) }   // arms, clear of the torso
        return Silhouette(w, h, on)
    }

    /** Pose points for that body: shoulders at row 90, hips at row 210, joints [jointsPx] apart. */
    private fun points(jointsPx: Int = 60): FloatArray {
        val p = FloatArray(CaptureGuide.POINTS * 3)
        fun set(i: Int, x: Float, y: Float) { p[i * 3] = x / w; p[i * 3 + 1] = y / h; p[i * 3 + 2] = 1f }
        set(CaptureGuide.LEFT_SHOULDER, 150f + jointsPx / 2, 90f); set(CaptureGuide.RIGHT_SHOULDER, 150f - jointsPx / 2, 90f)
        set(13, 150f + jointsPx / 2, 140f); set(14, 150f - jointsPx / 2, 140f)                       // elbows
        set(CaptureGuide.LEFT_WRIST, 150f + jointsPx / 2, 190f); set(CaptureGuide.RIGHT_WRIST, 150f - jointsPx / 2, 190f)
        set(CaptureGuide.LEFT_HIP, 164f, 210f); set(CaptureGuide.RIGHT_HIP, 136f, 210f)
        set(25, 164f, 285f); set(26, 136f, 285f)                                                     // knees
        set(CaptureGuide.LEFT_ANKLE, 164f, 355f); set(CaptureGuide.RIGHT_ANKLE, 136f, 355f)
        return p
    }

    /** The whole frame's height in millimetres such that the 320-row body is 1.76 m tall. */
    private val mmPerHeight = 1760f * h / 320f

    @Test
    fun heightAndLengthsComeFromTheFrontPhoto() {
        val m = BodyMeasure.measure(body(80), points(), null, null, aspect, mmPerHeight, true)!!
        assertEquals(176f, m.heightCm, 0.2f)
        // 60 px between the joints, 5.5 mm a pixel, and the shoulders' edge beyond the joints.
        assertEquals(60 * 0.55f * BodyMeasure.SHOULDER_EDGE, m.shoulderCm, 0.2f)
        assertEquals(100 * 0.55f, m.armCm, 0.2f)
        assertEquals(145 * 0.55f, m.legCm, 0.5f)
        assertEquals(120 * 0.55f, m.torsoCm, 0.2f)
    }

    @Test
    fun girthsUseTheSidePhotosDepth() {
        // 80 px wide and 60 px deep: an ellipse 44 cm by 33 cm.
        val m = BodyMeasure.measure(body(80), points(), body(60), points(), aspect, mmPerHeight, true)!!
        assertTrue(m.girthsFromSide)
        val expected = BodyMeasure.around(44f, 33f)
        assertEquals(expected, m.chestCm, 0.6f)
        assertEquals(expected, m.waistCm, 0.6f)
    }

    @Test
    fun withoutASidePhotoDepthIsAUsualShareOfWidth() {
        val m = BodyMeasure.measure(body(80), points(), null, null, aspect, mmPerHeight, false)!!
        assertFalse(m.girthsFromSide)
        assertFalse(m.scaleFromFace)
        assertEquals(BodyMeasure.around(44f, 44f * BodyMeasure.CHEST_DEPTH), m.chestCm, 0.6f)
        assertEquals(BodyMeasure.around(44f, 44f * BodyMeasure.WAIST_DEPTH), m.waistCm, 0.6f)
    }

    @Test
    fun armsHeldAwayAreNotMeasuredAsTorso() {
        val without = BodyMeasure.measure(body(80), points(), null, null, aspect, mmPerHeight, true)!!
        val with = BodyMeasure.measure(body(80, arms = true), points(), null, null, aspect, mmPerHeight, true)!!
        assertEquals(without.chestCm, with.chestCm, 0.01f)
    }

    @Test
    fun aSideDepthThatCannotBeRightIsNotUsed() {
        // A "side" silhouette wider than the front twice over: an arm swung out, or the wrong photo.
        val m = BodyMeasure.measure(body(60), points(), body(160), points(), aspect, mmPerHeight, true)!!
        assertFalse(m.girthsFromSide)
    }

    @Test
    fun anEmptyPhotoGivesNothing() {
        assertNull(BodyMeasure.measure(Silhouette(w, h, BooleanArray(w * h)), points(), null, null, aspect, mmPerHeight, true))
    }

    @Test
    fun theRunThroughTheCentreLineIsTheTorso() {
        val s = body(80, arms = true)
        assertEquals(110..189, s.run(150, 150))
        assertEquals(60..75, s.run(150, 50))
        assertNull(s.run(10, 150))
    }

    @Test
    fun eyeSpacingFromTheIris() {
        // An iris 20 px across and eyes 108 px apart: 108 / 20 irises of 11.7 mm.
        assertEquals(63.18f, BodyMeasure.ipdMm(20f, 108f), 0.01f)
    }

    @Test
    fun aroundACircleIsItsCircumference() {
        assertEquals((Math.PI * 30).toFloat(), BodyMeasure.around(30f, 30f), 0.01f)
    }

    // ------------------------------------------------------------------------------ skin tone

    private fun skin(r: Int, g: Int, b: Int) = SkinTone.read(IntArray(100) { (0xFF shl 24) or (r shl 16) or (g shl 8) or b })!!

    @Test
    fun labOfWhiteBlackAndGrey() {
        assertEquals(100f, SkinTone.toLab(255, 255, 255)[0], 0.1f)
        assertEquals(0f, SkinTone.toLab(0, 0, 0)[0], 0.1f)
        val grey = SkinTone.toLab(119, 119, 119)
        assertEquals(50f, grey[0], 0.6f)
        assertEquals(0f, grey[1], 0.1f)
        assertEquals(0f, grey[2], 0.1f)
    }

    @Test
    fun lighterSkinHasAHigherAngleThanDeeperSkin() {
        val light = skin(241, 211, 190)
        val medium = skin(198, 152, 118)
        val deep = skin(92, 62, 45)
        assertTrue(light.itaDeg > medium.itaDeg && medium.itaDeg > deep.itaDeg)
        assertTrue(light.depth.ordinal < medium.depth.ordinal && medium.depth.ordinal < deep.depth.ordinal)
        assertEquals(SkinDepth.DEEP, deep.depth)
    }

    @Test
    fun theBandsOfTheTypologyAngle() {
        assertEquals(SkinDepth.VERY_LIGHT, SkinTone.depthFor(60f))
        assertEquals(SkinDepth.LIGHT, SkinTone.depthFor(45f))
        assertEquals(SkinDepth.INTERMEDIATE, SkinTone.depthFor(30f))
        assertEquals(SkinDepth.TAN, SkinTone.depthFor(15f))
        assertEquals(SkinDepth.BROWN, SkinTone.depthFor(0f))
        assertEquals(SkinDepth.DEEP, SkinTone.depthFor(-40f))
    }

    @Test
    fun undertoneFollowsTheHue() {
        assertEquals(Undertone.WARM, SkinTone.undertoneFor(70f))
        assertEquals(Undertone.NEUTRAL, SkinTone.undertoneFor(54f))
        assertEquals(Undertone.COOL, SkinTone.undertoneFor(40f))
        // A yellower skin colour reads warmer than a pinker one of the same lightness.
        assertTrue(skin(200, 160, 110).hueDeg > skin(200, 150, 140).hueDeg)
    }

    @Test
    fun shadowAndShineAreDroppedBeforeAveraging() {
        val base = (0xFF shl 24) or (198 shl 16) or (152 shl 8) or 118
        val mixed = IntArray(100) { i -> if (i < 10) 0xFF101010.toInt() else if (i >= 90) 0xFFFFFFFF.toInt() else base }
        assertEquals(skin(198, 152, 118).itaDeg, SkinTone.read(mixed)!!.itaDeg, 0.01f)
        assertNull(SkinTone.read(IntArray(5)))
    }

    // ------------------------------------------------------------------------------ ranking

    private fun garment(id: String, category: String, colour: String, family: String, occasion: String = "smart-casual") = Garment(
        id = id, name = id, category = category, imageWidth = 1, imageHeight = 1, anchors = emptyMap(), parts = emptyList(),
        colourName = colour, colourHex = "#000000", colourFamily = family, occasion = occasion, gender = "unisex",
        fitStyle = "regular", suitableBodyTypes = listOf("slim", "average", "broad"), sizes = emptyList(), isPlaceholder = false,
    )

    private val tops = listOf(garment("navy_polo", "upper_body", "navy", "neutral"), garment("yellow_polo", "upper_body", "yellow", "warm"),
        garment("white_shirt", "upper_body", "white", "neutral"))
    private val bottoms = listOf(garment("khaki_chinos", "lower_body", "khaki", "neutral"), garment("black_trousers", "lower_body", "black", "neutral"))

    private fun measured(chest: Float, height: Float = 176f) =
        Measurements(height, 45f, chest, chest - 8, chest, 60f, 80f, 60f, girthsFromSide = true, scaleFromFace = true)

    @Test
    fun buildComesFromChestAgainstHeight() {
        assertEquals(BodyType.SLIM, PhotoRecommender.buildFor(measured(86f)))
        assertEquals(BodyType.AVERAGE, PhotoRecommender.buildFor(measured(98f)))
        assertEquals(BodyType.BROAD, PhotoRecommender.buildFor(measured(112f)))
    }

    @Test
    fun sizeComesFromTheChest() {
        assertEquals("S", PhotoRecommender.sizeFor(88f))
        assertEquals("M", PhotoRecommender.sizeFor(96f))
        assertEquals("L", PhotoRecommender.sizeFor(104f))
        assertEquals("XL", PhotoRecommender.sizeFor(112f))
        assertEquals("XXL", PhotoRecommender.sizeFor(120f))
    }

    @Test
    fun aDeeperWarmSkinToneLiftsALightWarmShirt() {
        val deepWarm = SkinReading(SkinDepth.BROWN, Undertone.WARM, 0f, 65f, 0)
        val reasons = PhotoRecommender.skinReasons(deepWarm, "yellow")
        assertEquals(PhotoRecommender.UNDERTONE_SUITS + PhotoRecommender.CONTRAST_SUITS, reasons.sumOf { it.points })
        // The same shirt for a light, cool skin tone gains nothing.
        assertTrue(PhotoRecommender.skinReasons(SkinReading(SkinDepth.LIGHT, Undertone.COOL, 45f, 40f, 0), "yellow").isEmpty())
        // A colour close to the skin loses points.
        assertEquals(PhotoRecommender.NEAR_SKIN, PhotoRecommender.skinReasons(SkinReading(SkinDepth.TAN, Undertone.NEUTRAL, 15f, 54f, 0), "khaki").sumOf { it.points })
    }

    @Test
    fun skinToneChangesWhichOutfitComesFirst() {
        val plain = PhotoRecommender.rank(PhotoProfile(), tops, bottoms)
        val deepWarm = PhotoRecommender.rank(PhotoProfile(skin = SkinReading(SkinDepth.BROWN, Undertone.WARM, 0f, 65f, 0)), tops, bottoms)
        assertEquals(6, plain.size)
        assertEquals("yellow_polo", deepWarm.first().top.id)
        assertTrue(deepWarm.first().reasons.any { "skin" in it.text })
        assertTrue(deepWarm.first().score > plain.first { it.top.id == "yellow_polo" && it.bottom.id == deepWarm.first().bottom.id }.score)
    }

    @Test
    fun measurementsGiveTheProfileItsBuildAndTheOutfitItsSize() {
        val p = PhotoProfile(measurements = measured(112f))
        assertEquals(BodyType.BROAD, PhotoRecommender.profile(p).bodyType)
        assertNull(PhotoRecommender.profile(p).heightCm)
        val first = PhotoRecommender.rank(p, tops, bottoms, 2)
        assertEquals(2, first.size)
        assertEquals("XL", first[0].size)
        assertNotNull(first[0].reasons.firstOrNull { "broad" in it.text })
    }
}
