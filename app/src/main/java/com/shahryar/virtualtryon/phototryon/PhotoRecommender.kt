package com.shahryar.virtualtryon.phototryon

import com.shahryar.virtualtryon.BodyType
import com.shahryar.virtualtryon.FaceShape
import com.shahryar.virtualtryon.Garment
import com.shahryar.virtualtryon.Gender
import com.shahryar.virtualtryon.Profile
import com.shahryar.virtualtryon.Reason
import com.shahryar.virtualtryon.Recommender

/** What the photos say about the wearer; any part may be missing. */
class PhotoProfile(
    val measurements: Measurements? = null,
    val skin: SkinReading? = null,
    val faceShape: FaceShape? = null,
    val gender: Gender = Gender.UNSPECIFIED,
    val favourites: Set<String> = emptySet(),
)

/** A shirt and trousers to try on, the score they were ranked by, and every reason behind it. */
class RankedOutfit(val top: Garment, val bottom: Garment, val score: Int, val reasons: List<Reason>, val size: String?)

/**
 * Ranks outfits for a wearer measured from their photos (Photo Try-On, stage 3).
 *
 * The live try-on's [Recommender] does the ranking it always did - build, colour preferences,
 * which shirt goes with which trousers - and is used as it is, not changed. This adds what only
 * the photos can know:
 *
 * - **Build** comes from the measured chest against the measured height, not from a form.
 * - **Skin tone** scores the shirt's colour, the one worn next to the face: a colour of the
 *   wearer's undertone gains [UNDERTONE_SUITS], one that stands out from their skin gains
 *   [CONTRAST_SUITS], and one close to it loses [NEAR_SKIN]. Common styling guidance, like the
 *   rules it joins; the tables are starting values.
 * - **Size** is read from the measured chest on a usual men's chart.
 */
object PhotoRecommender {

    const val UNDERTONE_SUITS = 10
    const val CONTRAST_SUITS = 10
    const val NEAR_SKIN = -10

    /** Chest round over height: below the first is slim, above the second broad. */
    const val SLIM_BELOW = 0.52f
    const val BROAD_ABOVE = 0.60f

    private val WARM_COLOURS = setOf("khaki", "brown", "maroon", "yellow", "green", "tortoiseshell")
    private val COOL_COLOURS = setOf("navy", "grey", "black", "pink", "silver", "white")
    private val LIGHT_COLOURS = setOf("white", "yellow", "pink", "khaki", "grey")
    private val DARK_COLOURS = setOf("navy", "maroon", "green", "black", "brown")
    private val NEAR = mapOf(
        SkinDepth.VERY_LIGHT to setOf("white"), SkinDepth.LIGHT to setOf("white", "khaki"),
        SkinDepth.INTERMEDIATE to setOf("khaki"), SkinDepth.TAN to setOf("khaki", "brown"),
        SkinDepth.BROWN to setOf("brown", "maroon"), SkinDepth.DEEP to setOf("black", "brown"),
    )

    /** The build a measured body is: its chest round against its height. */
    fun buildFor(m: Measurements): BodyType {
        val ratio = m.chestCm / m.heightCm
        return when {
            ratio < SLIM_BELOW -> BodyType.SLIM
            ratio > BROAD_ABOVE -> BodyType.BROAD
            else -> BodyType.AVERAGE
        }
    }

    /** A shirt size for a chest of [chestCm] round, on a usual men's chart. */
    fun sizeFor(chestCm: Float) = when {
        chestCm < 92f -> "S"
        chestCm < 100f -> "M"
        chestCm < 108f -> "L"
        chestCm < 116f -> "XL"
        else -> "XXL"
    }

    /** What [skin] says about a shirt of [colour]; empty when it says nothing. */
    fun skinReasons(skin: SkinReading?, colour: String?): List<Reason> {
        if (skin == null || colour == null) return emptyList()
        val name = colour.replaceFirstChar { it.uppercase() }
        val out = ArrayList<Reason>()
        when (skin.undertone) {
            Undertone.WARM -> if (colour in WARM_COLOURS) out += Reason(UNDERTONE_SUITS, "$name suits your warm skin undertone")
            Undertone.COOL -> if (colour in COOL_COLOURS) out += Reason(UNDERTONE_SUITS, "$name suits your cool skin undertone")
            Undertone.NEUTRAL -> Unit
        }
        val deep = skin.depth == SkinDepth.BROWN || skin.depth == SkinDepth.DEEP
        val light = skin.depth == SkinDepth.VERY_LIGHT || skin.depth == SkinDepth.LIGHT
        when {
            colour in NEAR[skin.depth].orEmpty() -> out += Reason(NEAR_SKIN, "$name is close to your skin tone, so it stands out less")
            deep && colour in LIGHT_COLOURS -> out += Reason(CONTRAST_SUITS, "$name stands out well against a deeper skin tone")
            light && colour in DARK_COLOURS -> out += Reason(CONTRAST_SUITS, "$name stands out well against a lighter skin tone")
        }
        return out
    }

    /** The live try-on's profile for this wearer: what its [Recommender] ranks by. */
    fun profile(p: PhotoProfile) = Profile(
        gender = p.gender,
        // The catalog's sizes are not tied to heights, so height is left out and size comes from the chest.
        heightCm = null,
        bodyType = p.measurements?.let { buildFor(it) },
        colours = p.favourites,
        faceShape = p.faceShape,
    )

    /** Every shirt with every pair of trousers, best first, [limit] at most. */
    fun rank(p: PhotoProfile, tops: List<Garment>, bottoms: List<Garment>, limit: Int = 6): List<RankedOutfit> {
        val size = p.measurements?.let { sizeFor(it.chestCm) }
        return Recommender().rankOutfits(profile(p), tops, bottoms, tops.size * bottoms.size).map { outfit ->
            val skin = skinReasons(p.skin, outfit.top.item.colourName)
            RankedOutfit(outfit.top.item, outfit.bottom.item, outfit.score + skin.sumOf { it.points },
                skin + outfit.top.reasons + outfit.bottom.reasons + outfit.pairing, size)
        }.sortedWith(compareByDescending<RankedOutfit> { it.score }.thenBy { it.top.name }.thenBy { it.bottom.name }).take(limit)
    }
}
