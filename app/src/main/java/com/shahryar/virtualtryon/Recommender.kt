package com.shahryar.virtualtryon

import kotlin.math.abs

/** How the user describes themselves (M7). */
enum class Gender(val id: String) { MALE("male"), FEMALE("female"), UNSPECIFIED("unspecified") }

/** Build, as the profile screen offers it; [id] matches the catalog's suitableBodyTypes. */
enum class BodyType(val id: String) { SLIM("slim"), AVERAGE("average"), BROAD("broad") }

/** Face shape, measured by Face Mesh or picked by the user; [id] names it in the rules below. */
enum class FaceShape(val id: String) { OVAL("oval"), ROUND("round"), SQUARE("square"), HEART("heart"), OBLONG("oblong") }

/**
 * Everything the engine knows about the user. Any field can be left out: a rule whose input is
 * missing simply does not score.
 */
data class Profile(
    val gender: Gender = Gender.UNSPECIFIED,
    val heightCm: Int? = null,
    val bodyType: BodyType? = null,
    /** Favourite colours, by the catalogs' colour names: "navy", "khaki", "tortoiseshell"... */
    val colours: Set<String> = emptySet(),
    val faceShape: FaceShape? = null,
)

/** Whether gender lowers an item's rank ([PREFERENCE]) or removes it ([FILTER]). */
enum class GenderMode { PREFERENCE, FILTER }

/** One rule's verdict on an item: its points and a sentence saying why. */
data class Reason(val points: Int, val text: String)

/** An item, its total score and the reasons that make it up; [size] for garments with one. */
data class Suggestion<T>(val item: T, val score: Int, val reasons: List<Reason>, val size: String? = null)

/** A shirt and trousers worn together: their own scores plus how well they go together. */
data class Outfit(
    val top: Suggestion<Garment>,
    val bottom: Suggestion<Garment>,
    val score: Int,
    /** How the two go together; each item's own reasons stay with it. */
    val pairing: List<Reason>,
)

/**
 * The recommendation engine (M7): ranks garments, outfits and glasses for a [Profile] with plain
 * rules over the catalogs' hand-judged details. Each rule adds or takes away points and says why
 * in a sentence, so every suggestion can show its reasons, and the rules can be listed in full in
 * the report. They encode common styling and optician guidance, not measured preferences.
 *
 * - Gender: an item cut for the other gender loses points; with [GenderMode.FILTER], it is not
 *   suggested at all. All four trousers are men's, so it is a preference until women's trousers
 *   join the catalog; [DEFAULT_GENDER_MODE] is the one switch to change then.
 * - Body type: an item cut for the user's build gains points, one not cut for it loses them, and
 *   the fit made for that build - slim for slim, regular for average, relaxed for broad - gains a
 *   little more.
 * - Height: picks the size, and for trousers the leg shape: a tapered leg suits a shorter frame,
 *   a wide leg a tall one.
 * - Colour: a favourite colour gains the most; another colour of the same family - neutral, warm
 *   or cool - gains a little.
 * - Outfits: a shirt and trousers for the same occasion go together; a casual tee with formal
 *   trousers does not; a neutral piece goes with anything; two pieces of the same bright colour
 *   clash.
 * - Glasses: frame shapes that balance the user's face shape gain points, ones that repeat it
 *   lose them; frame colour scores as garment colour does.
 */
class Recommender(private val genderMode: GenderMode = DEFAULT_GENDER_MODE) {

    companion object {
        /**
         * Gender lowers a rank for now. Switch to [GenderMode.FILTER] once the catalog has
         * trousers for women, so a woman is no longer shown men's trousers at all.
         */
        val DEFAULT_GENDER_MODE = GenderMode.PREFERENCE

        // Points per rule.
        const val GENDER_MISMATCH = -30
        const val SUITS_BUILD = 20
        const val NOT_FOR_BUILD = -20
        const val FIT_FOR_BUILD = 10
        const val NO_SIZE = -10
        const val LEG_GOOD = 10
        const val LEG_FAIR = 5
        const val LEG_POOR = -10
        const val FAVOURITE_COLOUR = 30
        const val FAVOURITE_FAMILY = 10
        const val SAME_OCCASION = 15
        const val NEAR_OCCASION = 5
        const val CLASHING_OCCASION = -15
        const val NEUTRAL_GOES_WITH_ALL = 10

        /**
         * As much as a favourite colour gains: when both pieces are that colour, the outfit gains
         * it once, not twice, and two bright pieces of one colour do not top the list.
         */
        const val SAME_COLOUR_CLASH = -FAVOURITE_COLOUR
        const val FACE_SUITS = 30
        const val FACE_AVOID = -25
        const val FRAME_COLOUR = 20
        const val FRAME_FAMILY = 8

        /** Heights below are "shorter", above "tall", for the trouser leg rule, cm. */
        const val SHORTER_BELOW_CM = 165
        const val TALL_ABOVE_CM = 180

        /**
         * Colour families. Garments carry their own; this covers the names they do not - glasses
         * frames - and the user's favourites, which are only names.
         */
        val COLOUR_FAMILIES = mapOf(
            "black" to "neutral", "white" to "neutral", "grey" to "neutral", "navy" to "neutral",
            "khaki" to "neutral", "silver" to "neutral",
            "pink" to "warm", "yellow" to "warm", "maroon" to "warm", "brown" to "warm", "tortoiseshell" to "warm",
            "green" to "cool",
        )

        /** Occasions from least to most formal. */
        val OCCASIONS = listOf("casual", "smart-casual", "formal")

        /** The fit made for each build. */
        private val FIT_FOR = mapOf(BodyType.SLIM to "slim", BodyType.AVERAGE to "regular", BodyType.BROAD to "relaxed")

        /**
         * Frame shapes that suit and that are best avoided with each face shape, and why: angles
         * balance a soft face, curves a strong one. Common optician guidance.
         */
        private class FaceRule(val suits: Set<String>, val avoid: Set<String>, val why: String, val whyNot: String)

        private val FACE_RULES = mapOf(
            FaceShape.ROUND to FaceRule(setOf("rectangular", "rounded-square", "cat-eye"), setOf("round"),
                "angles balance a round face", "round frames repeat a round face"),
            FaceShape.SQUARE to FaceRule(setOf("round", "aviator", "cat-eye"), setOf("rectangular"),
                "curves soften a square jaw", "square corners repeat a square jaw"),
            FaceShape.OVAL to FaceRule(setOf("aviator", "rectangular", "rounded-square", "round", "cat-eye"), emptySet(),
                "an oval face suits most frames", ""),
            FaceShape.HEART to FaceRule(setOf("aviator", "round", "rounded-square"), setOf("cat-eye"),
                "these balance a wider forehead", "upswept corners widen the forehead further"),
            FaceShape.OBLONG to FaceRule(setOf("round", "aviator", "rounded-square"), setOf("rectangular"),
                "deeper frames shorten a long face", "narrow frames lengthen a long face further"),
        )

        private fun capitalise(s: String) = s.replaceFirstChar { it.uppercase() }
    }

    /** Garments ranked best first; mixed categories rank together. */
    fun rankGarments(profile: Profile, garments: List<Garment>): List<Suggestion<Garment>> =
        garments.mapNotNull { score(profile, it) }.sortedWith(compareByDescending<Suggestion<Garment>> { it.score }.thenBy { it.item.name })

    /** Shirts and trousers paired every way, ranked best first, [limit] at most. */
    fun rankOutfits(profile: Profile, tops: List<Garment>, bottoms: List<Garment>, limit: Int = 5): List<Outfit> {
        val scoredTops = tops.mapNotNull { score(profile, it) }
        val scoredBottoms = bottoms.mapNotNull { score(profile, it) }
        return scoredTops.flatMap { top -> scoredBottoms.map { bottom -> pair(top, bottom) } }
            .sortedWith(compareByDescending<Outfit> { it.score }.thenBy { it.top.item.name }.thenBy { it.bottom.item.name })
            .take(limit)
    }

    /** Glasses ranked best first. */
    fun rankGlasses(profile: Profile, pairs: List<GlassesPair>): List<Suggestion<GlassesPair>> =
        pairs.mapNotNull { score(profile, it) }.sortedWith(compareByDescending<Suggestion<GlassesPair>> { it.score }.thenBy { it.item.name })

    /** The size for [heightCm]: the band holding it, nearest its middle where two overlap. */
    fun sizeFor(garment: Garment, heightCm: Int): SizeBand? =
        garment.sizes.filter { heightCm in it.heightMinCm..it.heightMaxCm }
            .minByOrNull { abs(heightCm * 2 - it.heightMinCm - it.heightMaxCm) }

    // ------------------------------------------------------------------------------- garments

    private fun score(profile: Profile, garment: Garment): Suggestion<Garment>? {
        val reasons = ArrayList<Reason>()
        if (!genderRule(profile, garment.gender, reasons)) return null
        profile.bodyType?.let { build ->
            if (build.id in garment.suitableBodyTypes) {
                reasons += Reason(SUITS_BUILD, "Suits a ${build.id} build")
            } else {
                reasons += Reason(NOT_FOR_BUILD, "${capitalise(garment.fitStyle)} fit is not cut for a ${build.id} build")
            }
            if (FIT_FOR[build] == garment.fitStyle) {
                reasons += Reason(FIT_FOR_BUILD, "A ${garment.fitStyle} fit is made for a ${build.id} build")
            }
        }
        var size: String? = null
        profile.heightCm?.let { height ->
            val band = sizeFor(garment, height)
            if (band == null) {
                reasons += Reason(NO_SIZE, "No size for $height cm")
            } else {
                size = band.label
                reasons += Reason(0, "Size ${band.label} for $height cm")
            }
            garment.legShape?.let { legRule(height, it)?.let(reasons::add) }
        }
        colourRule(profile, garment.colourName, garment.colourFamily ?: COLOUR_FAMILIES[garment.colourName],
            FAVOURITE_COLOUR, FAVOURITE_FAMILY, reasons)
        return Suggestion(garment, reasons.sumOf { it.points }, reasons, size)
    }

    /** Leg shape against height: tapered lengthens a shorter frame, wide suits a tall one. */
    private fun legRule(height: Int, leg: String): Reason? = when {
        height < SHORTER_BELOW_CM -> when (leg) {
            "tapered" -> Reason(LEG_GOOD, "A tapered leg lengthens a shorter frame")
            "straight" -> Reason(LEG_FAIR, "A straight leg keeps a shorter frame in proportion")
            "wide" -> Reason(LEG_POOR, "A wide leg can shorten a shorter frame")
            else -> null
        }
        height > TALL_ABOVE_CM -> when (leg) {
            "wide" -> Reason(LEG_GOOD, "A tall frame carries a wide leg well")
            "straight" -> Reason(LEG_FAIR, "A straight leg suits a tall frame")
            else -> null
        }
        else -> when (leg) {
            "tapered", "straight" -> Reason(LEG_FAIR, "A ${leg} leg suits your height")
            else -> null
        }
    }

    /** Scores gender into [reasons]; false when [genderMode] filters the item out. */
    private fun genderRule(profile: Profile, itemGender: String, reasons: MutableList<Reason>): Boolean {
        val mismatch = profile.gender != Gender.UNSPECIFIED && itemGender != "unisex" && itemGender != profile.gender.id
        if (!mismatch) return true
        if (genderMode == GenderMode.FILTER) return false
        reasons += Reason(GENDER_MISMATCH, "Cut for ${if (itemGender == "male") "men" else "women"}")
        return true
    }

    private fun colourRule(profile: Profile, colour: String?, family: String?, exact: Int, near: Int, reasons: MutableList<Reason>) {
        if (colour == null || profile.colours.isEmpty()) return
        if (colour in profile.colours) {
            reasons += Reason(exact, "${capitalise(colour)} is one of your colours")
            return
        }
        val like = profile.colours.firstOrNull { family != null && COLOUR_FAMILIES[it] == family } ?: return
        reasons += Reason(near, "${capitalise(family!!)}, like your $like")
    }

    // -------------------------------------------------------------------------------- outfits

    private fun pair(top: Suggestion<Garment>, bottom: Suggestion<Garment>): Outfit {
        val pairing = ArrayList<Reason>()
        val a = OCCASIONS.indexOf(top.item.occasion)
        val b = OCCASIONS.indexOf(bottom.item.occasion)
        if (a >= 0 && b >= 0) {
            pairing += when (abs(a - b)) {
                0 -> Reason(SAME_OCCASION, "Both ${OCCASIONS[a]}")
                1 -> Reason(NEAR_OCCASION, "${capitalise(OCCASIONS[a])} and ${OCCASIONS[b]} sit well together")
                else -> Reason(CLASHING_OCCASION, "${capitalise(OCCASIONS[a])} and ${OCCASIONS[b]} do not mix")
            }
        }
        val topFamily = top.item.colourFamily ?: COLOUR_FAMILIES[top.item.colourName]
        val bottomFamily = bottom.item.colourFamily ?: COLOUR_FAMILIES[bottom.item.colourName]
        when {
            top.item.colourName == bottom.item.colourName && topFamily != "neutral" ->
                pairing += Reason(SAME_COLOUR_CLASH, "${capitalise(top.item.colourName)} on ${top.item.colourName} clashes")
            bottomFamily == "neutral" ->
                pairing += Reason(NEUTRAL_GOES_WITH_ALL, "${capitalise(bottom.item.colourName)} trousers go with any shirt")
            topFamily == "neutral" ->
                pairing += Reason(NEUTRAL_GOES_WITH_ALL, "A ${top.item.colourName} shirt goes with any trousers")
        }
        return Outfit(top, bottom, top.score + bottom.score + pairing.sumOf { it.points }, pairing)
    }

    // -------------------------------------------------------------------------------- glasses

    private fun score(profile: Profile, pair: GlassesPair): Suggestion<GlassesPair>? {
        val reasons = ArrayList<Reason>()
        if (!genderRule(profile, pair.gender, reasons)) return null
        val shape = pair.frameShape
        val face = profile.faceShape
        if (shape != null && face != null) {
            val rule = FACE_RULES.getValue(face)
            when (shape) {
                in rule.suits -> reasons += Reason(FACE_SUITS, "${capitalise(shape)} frames suit a ${face.id} face: ${rule.why}")
                in rule.avoid -> reasons += Reason(FACE_AVOID, "${capitalise(shape)} frames: ${rule.whyNot}")
            }
        }
        colourRule(profile, pair.frameColour, pair.frameColour?.let { COLOUR_FAMILIES[it] }, FRAME_COLOUR, FRAME_FAMILY, reasons)
        return Suggestion(pair, reasons.sumOf { it.points }, reasons)
    }
}
