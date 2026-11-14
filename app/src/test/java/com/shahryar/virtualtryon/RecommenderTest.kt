package com.shahryar.virtualtryon

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * M7's definition of done, as tests: a profile gives sensible ranked suggestions, and changing
 * any single input visibly changes the ranking.
 *
 * The garments and glasses are the real catalogs' items with the details judged for them by hand
 * (assets/garments/real/catalog.json, assets/glasses/catalog.json). They are written out here
 * because a local unit test has no Android JSON reader to load the catalogs with.
 */
class RecommenderTest {

    private val sizes = listOf(
        SizeBand("S", 0.58f, 0.70f, 150, 168),
        SizeBand("M", 0.68f, 0.80f, 165, 180),
        SizeBand("L", 0.78f, 0.92f, 177, 196),
    )
    private val all = listOf("slim", "average", "broad")

    private fun garment(
        id: String, name: String, category: String, colour: String, family: String, gender: String,
        fit: String, builds: List<String>, occasion: String, leg: String? = null,
    ) = Garment(
        id = id, name = name, category = category, imageWidth = 1, imageHeight = 1, anchors = emptyMap(),
        parts = emptyList(), colourName = colour, colourHex = "#000000", colourFamily = family,
        occasion = occasion, legShape = leg, gender = gender, fitStyle = fit, suitableBodyTypes = builds,
        sizes = sizes, isPlaceholder = false,
    )

    private val tops = listOf(
        garment("polo_pink", "Pink Polo", "upper_body", "pink", "warm", "unisex", "regular", all, "smart-casual"),
        garment("navy_polo", "Navy Polo", "upper_body", "navy", "neutral", "unisex", "regular", all, "smart-casual"),
        garment("green_polo", "Green Polo", "upper_body", "green", "cool", "unisex", "regular", all, "smart-casual"),
        garment("yellow_polo", "Yellow Polo", "upper_body", "yellow", "warm", "unisex", "regular", all, "smart-casual"),
        garment("maroon_polo", "Maroon Polo", "upper_body", "maroon", "warm", "unisex", "regular", all, "smart-casual"),
        garment("grey_tee", "Grey Tee", "upper_body", "grey", "neutral", "unisex", "regular", all, "casual"),
        garment("white_shirt", "White Waffle Shirt", "upper_body", "white", "neutral", "unisex", "relaxed",
            listOf("average", "broad"), "casual"),
    )
    private val bottoms = listOf(
        garment("khaki_chinos", "Khaki Chinos", "lower_body", "khaki", "neutral", "male", "regular", all,
            "smart-casual", "straight"),
        garment("green_work_pants", "Green Work Pants", "lower_body", "green", "cool", "male", "relaxed",
            listOf("average", "broad"), "casual", "wide"),
        garment("black_slim_trousers", "Black Slim Trousers", "lower_body", "black", "neutral", "male", "slim",
            listOf("slim", "average"), "formal", "tapered"),
        garment("black_belted_trousers", "Black Belted Trousers", "lower_body", "black", "neutral", "male",
            "regular", all, "formal", "straight"),
    )
    private val glasses = listOf(
        GlassesPair("sunglasses_khronos", "Aviators", "", 131f, "aviator", "silver", "sun"),
        GlassesPair("black_wood_frames", "Black wood", "", 101f, "rectangular", "black", "clear"),
        GlassesPair("black_violet_frames", "Black & violet", "", 132f, "rounded-square", "black", "clear"),
        GlassesPair("brown_round_frames", "Brown round", "", 108f, "round", "brown", "light-tint"),
        GlassesPair("tortoise_cat_eye", "Tortoise cat-eye", "", 106f, "cat-eye", "tortoiseshell", "sun"),
    )

    private val engine = Recommender()

    private fun ids(list: List<Suggestion<Garment>>) = list.map { it.item.id }
    private fun glassIds(list: List<Suggestion<GlassesPair>>) = list.map { it.item.id }

    // ------------------------------------------------------- each input changes the ranking

    @Test
    fun bodyTypeChangesTheRanking() {
        val slim = engine.rankGarments(Profile(bodyType = BodyType.SLIM), bottoms)
        val broad = engine.rankGarments(Profile(bodyType = BodyType.BROAD), bottoms)
        assertNotEquals(ids(slim), ids(broad))
        // Slim trousers first for a slim build, last for a broad one; wide work pants the other way round.
        assertEquals("black_slim_trousers", slim.first().item.id)
        assertEquals("green_work_pants", slim.last().item.id)
        assertEquals("green_work_pants", broad.first().item.id)
        assertEquals("black_slim_trousers", broad.last().item.id)
    }

    @Test
    fun heightChangesTheRanking() {
        val shorter = engine.rankGarments(Profile(heightCm = 158), bottoms)
        val tall = engine.rankGarments(Profile(heightCm = 188), bottoms)
        assertNotEquals(ids(shorter), ids(tall))
        assertEquals("black_slim_trousers", shorter.first().item.id) // tapered
        assertEquals("green_work_pants", shorter.last().item.id) // wide
        assertEquals("green_work_pants", tall.first().item.id)
    }

    @Test
    fun colourChangesTheRanking() {
        val pink = engine.rankGarments(Profile(colours = setOf("pink")), tops)
        val green = engine.rankGarments(Profile(colours = setOf("green")), tops)
        assertNotEquals(ids(pink), ids(green))
        assertEquals("polo_pink", pink.first().item.id)
        assertEquals("green_polo", green.first().item.id)
        // The rest of the warm family follows a warm favourite.
        assertEquals(setOf("maroon_polo", "yellow_polo"), ids(pink).subList(1, 3).toSet())
    }

    @Test
    fun genderChangesTheRanking() {
        val garments = tops + bottoms
        val male = engine.rankGarments(Profile(gender = Gender.MALE), garments)
        val female = engine.rankGarments(Profile(gender = Gender.FEMALE), garments)
        assertNotEquals(ids(male), ids(female))
        // As a preference, men's trousers stay for a woman, below every shirt.
        assertEquals(tops.size, female.indexOfFirst { it.item.category == "lower_body" })
        assertTrue(female.filter { it.item.category == "lower_body" }.all { s -> s.reasons.any { it.text == "Cut for men" } })
    }

    @Test
    fun faceShapeChangesTheRanking() {
        val round = engine.rankGlasses(Profile(faceShape = FaceShape.ROUND), glasses)
        val square = engine.rankGlasses(Profile(faceShape = FaceShape.SQUARE), glasses)
        assertNotEquals(glassIds(round), glassIds(square))
        assertEquals("brown_round_frames", round.last().item.id) // round frames repeat a round face
        assertEquals("black_wood_frames", square.last().item.id) // rectangular repeats a square jaw
        assertTrue(square.first().item.frameShape in setOf("round", "aviator", "cat-eye"))
    }

    @Test
    fun frameColourChangesTheRanking() {
        val silver = engine.rankGlasses(Profile(colours = setOf("silver")), glasses)
        val brown = engine.rankGlasses(Profile(colours = setOf("brown")), glasses)
        assertEquals("sunglasses_khronos", silver.first().item.id)
        assertEquals("brown_round_frames", brown.first().item.id)
    }

    // --------------------------------------------------------------------- the gender switch

    @Test
    fun filterModeLeavesOutTheOtherGendersItems() {
        val filtered = Recommender(GenderMode.FILTER).rankGarments(Profile(gender = Gender.FEMALE), tops + bottoms)
        assertEquals(tops.map { it.id }.toSet(), ids(filtered).toSet())
        // Unisex items, and every item for a user who gives no gender, are never filtered.
        assertEquals((tops + bottoms).size, Recommender(GenderMode.FILTER).rankGarments(Profile(), tops + bottoms).size)
    }

    @Test
    fun preferenceIsTheDefault() {
        assertEquals(GenderMode.PREFERENCE, Recommender.DEFAULT_GENDER_MODE)
    }

    // ------------------------------------------------------------------------------- outfits

    @Test
    fun outfitsPairByOccasionAndColour() {
        val outfits = engine.rankOutfits(Profile(), tops, bottoms, limit = Int.MAX_VALUE)
        assertEquals(tops.size * bottoms.size, outfits.size)
        fun outfit(top: String, bottom: String) = outfits.single { it.top.item.id == top && it.bottom.item.id == bottom }
        // A polo with chinos: same occasion, neutral trousers.
        val polo = outfit("navy_polo", "khaki_chinos")
        // A casual tee with formal trousers does not mix.
        val tee = outfit("grey_tee", "black_slim_trousers")
        assertTrue(polo.score > tee.score)
        assertTrue(tee.pairing.any { it.points == Recommender.CLASHING_OCCASION })
        // Green on green clashes.
        assertTrue(outfit("green_polo", "green_work_pants").pairing.any { it.points == Recommender.SAME_COLOUR_CLASH })
    }

    @Test
    fun outfitsFollowTheProfile() {
        val slimPink = engine.rankOutfits(Profile(bodyType = BodyType.SLIM, colours = setOf("pink")), tops, bottoms)
        val broadGreen = engine.rankOutfits(Profile(bodyType = BodyType.BROAD, colours = setOf("green")), tops, bottoms)
        assertEquals("polo_pink", slimPink.first().top.item.id)
        assertNotEquals(slimPink.map { it.top.item.id to it.bottom.item.id }, broadGreen.map { it.top.item.id to it.bottom.item.id })
    }

    // ------------------------------------------------------------------------ the basics

    @Test
    fun anEmptyProfileScoresNothing() {
        assertTrue(engine.rankGarments(Profile(), tops + bottoms).all { it.score == 0 && it.reasons.isEmpty() })
        assertTrue(engine.rankGlasses(Profile(), glasses).all { it.score == 0 })
    }

    @Test
    fun sizesFollowHeight() {
        val polo = tops.first()
        assertEquals("S", engine.sizeFor(polo, 158)?.label)
        assertEquals("M", engine.sizeFor(polo, 174)?.label)
        assertEquals("L", engine.sizeFor(polo, 188)?.label)
        assertEquals(null, engine.sizeFor(polo, 205))
        val ranked = engine.rankGarments(Profile(heightCm = 174), tops)
        assertTrue(ranked.all { it.size == "M" && it.reasons.any { r -> r.text == "Size M for 174 cm" } })
    }

    @Test
    fun everySuggestionSaysWhy() {
        val profile = Profile(Gender.MALE, 172, BodyType.AVERAGE, setOf("navy"), FaceShape.OVAL)
        val ranked = engine.rankGarments(profile, tops + bottoms) + engine.rankGlasses(profile, glasses).map {
            Suggestion(tops.first(), it.score, it.reasons)
        }
        assertTrue(ranked.all { s -> s.reasons.isNotEmpty() && s.score == s.reasons.sumOf { it.points } })
        assertTrue(ranked.all { s -> s.reasons.all { it.text.isNotBlank() } })
    }
}
