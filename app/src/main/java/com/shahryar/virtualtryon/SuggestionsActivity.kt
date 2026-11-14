package com.shahryar.virtualtryon

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.View
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.color.MaterialColors
import com.shahryar.virtualtryon.databinding.ActivitySuggestionsBinding
import com.shahryar.virtualtryon.databinding.ItemSuggestionBinding
import java.util.concurrent.Executors

/**
 * Suggestions ranked by [Recommender] for the saved profile (M7): the best outfits, then every
 * shirt, pair of trousers and pair of glasses, each with its size and the reasons behind its
 * score. "Try on" returns to the camera with the item - or the whole outfit - put on, which joins
 * the recommendation use case to the try-on one.
 *
 * The lists are ranked again whenever this screen comes back into view, so editing the profile
 * shows its effect straight away.
 */
class SuggestionsActivity : AppCompatActivity() {

    companion object {
        // What to put on, by catalog id, in the result for the camera screen.
        const val EXTRA_TOP = "com.shahryar.virtualtryon.TOP"
        const val EXTRA_BOTTOM = "com.shahryar.virtualtryon.BOTTOM"
        const val EXTRA_GLASSES = "com.shahryar.virtualtryon.GLASSES"

        /** Outfits listed: the best few of every shirt-and-trousers pairing. */
        const val OUTFITS_SHOWN = 3

        /** Garment photos are decoded down to about this many px high for the cards. */
        private const val PICTURE_PX = 216
        private const val TAG = "VTO"
    }

    private lateinit var binding: ActivitySuggestionsBinding
    private lateinit var store: ProfileStore
    private val engine = Recommender()
    private val worker = Executors.newSingleThreadExecutor()

    private var tops: List<Garment> = emptyList()
    private var bottoms: List<Garment> = emptyList()
    private var glasses: List<GlassesPair> = emptyList()
    private val pictures = HashMap<String, Bitmap>()
    private var loaded = false

    /** The profile can ask for a face measurement, which only the camera screen can make. */
    private val profileScreen = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == ProfileActivity.RESULT_MEASURE_FACE) {
            setResult(ProfileActivity.RESULT_MEASURE_FACE)
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivitySuggestionsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(binding.suggestionsRoot) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        store = ProfileStore(this)
        binding.toolbar.setNavigationOnClickListener { finish() }
        binding.editProfileButton.setOnClickListener {
            profileScreen.launch(Intent(this, ProfileActivity::class.java))
        }

        worker.execute {
            // Placeholders are test fixtures for the overlay, not things to suggest.
            val garments = GarmentCatalog.load(this).filterNot { it.isPlaceholder }
            val pairs = GlassesShape.catalog(this)
            val photos = garments.mapNotNull { g -> g.frontPath?.let { path -> picture(path)?.let { g.id to it } } }
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                tops = garments.filter { it.category == "upper_body" }
                bottoms = garments.filter { it.category == "lower_body" }
                glasses = pairs
                pictures.putAll(photos)
                loaded = true
                binding.loading.visibility = View.GONE
                render()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (loaded) render()
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.shutdown()
    }

    /** Ranks everything for the saved profile and lists it. */
    private fun render() {
        val profile = store.load()
        showProfile(profile)
        val sections = binding.sections
        sections.removeAllViews()

        val outfits = engine.rankOutfits(profile, tops, bottoms, OUTFITS_SHOWN)
        if (outfits.isNotEmpty()) {
            sections.addView(sectionTitle(R.string.suggestions_outfits))
            for (o in outfits) sections.addView(outfitCard(o))
        }
        if (tops.isNotEmpty()) {
            sections.addView(sectionTitle(R.string.suggestions_shirts))
            for (s in engine.rankGarments(profile, tops)) sections.addView(garmentCard(s))
        }
        if (bottoms.isNotEmpty()) {
            sections.addView(sectionTitle(R.string.suggestions_trousers))
            for (s in engine.rankGarments(profile, bottoms)) sections.addView(garmentCard(s))
        }
        if (glasses.isNotEmpty()) {
            sections.addView(sectionTitle(R.string.suggestions_glasses))
            if (profile.faceShape == null) sections.addView(sectionNote(R.string.suggestions_glasses_no_face))
            for (s in engine.rankGlasses(profile, glasses)) sections.addView(glassesCard(s))
        }
    }

    private fun showProfile(profile: Profile) {
        val parts = ArrayList<String>()
        when (profile.gender) {
            Gender.MALE -> parts += getString(R.string.profile_gender_male)
            Gender.FEMALE -> parts += getString(R.string.profile_gender_female)
            Gender.UNSPECIFIED -> Unit
        }
        profile.heightCm?.let { parts += getString(R.string.summary_height, it) }
        profile.bodyType?.let { parts += getString(R.string.summary_build, it.id) }
        if (profile.colours.isNotEmpty()) parts += getString(R.string.summary_likes, profile.colours.sorted().joinToString(", "))
        profile.faceShape?.let {
            val name = getString(faceShapeName(it)).lowercase()
            parts += getString(if (store.isFaceShapeMeasured()) R.string.summary_face_measured else R.string.summary_face_picked, name)
        }
        if (parts.isEmpty()) {
            binding.profileSummary.setText(R.string.suggestions_empty_profile)
            binding.editProfileButton.setText(R.string.suggestions_fill_profile)
        } else {
            binding.profileSummary.text = parts.joinToString(" · ").replaceFirstChar { it.uppercase() }
            binding.editProfileButton.setText(R.string.suggestions_edit_profile)
        }
    }

    // ---------------------------------------------------------------------------------- cards

    private fun garmentCard(s: Suggestion<Garment>): View {
        val card = ItemSuggestionBinding.inflate(layoutInflater, binding.sections, false)
        val g = s.item
        card.title.text = g.name
        card.subtitle.text = listOfNotNull(
            s.size?.let { getString(R.string.suggestions_size, it) },
            getString(R.string.suggestions_fit, g.fitStyle),
            g.legShape?.let { getString(R.string.suggestions_leg, it) },
            g.occasion,
        ).joinToString(" · ")
        showScore(card, s.score)
        showReasons(card.reasons, s.reasons)
        showPicture(card, g.id, null)
        card.tryOnButton.setOnClickListener {
            tryOn(Intent().putExtra(if (g.category == "lower_body") EXTRA_BOTTOM else EXTRA_TOP, g.id))
        }
        return card.root
    }

    private fun outfitCard(o: Outfit): View {
        val card = ItemSuggestionBinding.inflate(layoutInflater, binding.sections, false)
        card.title.text = "${o.top.item.name} + ${o.bottom.item.name}"
        card.subtitle.text = listOfNotNull(
            getString(R.string.suggestions_outfit_scores, signed(o.top.score), signed(o.bottom.score)),
            o.top.size?.let { getString(R.string.suggestions_size, it) },
        ).joinToString(" · ")
        showScore(card, o.score)
        showReasons(card.reasons, o.pairing)
        showPicture(card, o.top.item.id, o.bottom.item.id)
        card.tryOnButton.setOnClickListener {
            tryOn(Intent().putExtra(EXTRA_TOP, o.top.item.id).putExtra(EXTRA_BOTTOM, o.bottom.item.id))
        }
        return card.root
    }

    private fun glassesCard(s: Suggestion<GlassesPair>): View {
        val card = ItemSuggestionBinding.inflate(layoutInflater, binding.sections, false)
        val p = s.item
        card.title.text = p.name
        val lenses = when (p.lenses) {
            "sun" -> getString(R.string.suggestions_lenses_sun)
            "light-tint" -> getString(R.string.suggestions_lenses_tint)
            "clear" -> getString(R.string.suggestions_lenses_clear)
            else -> p.lenses
        }
        card.subtitle.text = if (p.frameShape != null && p.frameColour != null && lenses != null) {
            getString(R.string.suggestions_frame, p.frameShape.replaceFirstChar { it.uppercase() }, p.frameColour, lenses)
        } else {
            ""
        }
        showScore(card, s.score)
        showReasons(card.reasons, s.reasons)
        card.pictures.visibility = View.GONE
        card.tryOnButton.setOnClickListener { tryOn(Intent().putExtra(EXTRA_GLASSES, p.id)) }
        return card.root
    }

    private fun tryOn(result: Intent) {
        setResult(RESULT_OK, result)
        finish()
    }

    private fun showScore(card: ItemSuggestionBinding, score: Int) {
        card.score.text = signed(score)
        card.score.setTextColor(pointsColour(card.score, score))
    }

    /** One line per scoring reason, its points coloured by sign; reasons worth nothing are left out. */
    private fun showReasons(view: TextView, reasons: List<Reason>) {
        val text = SpannableStringBuilder()
        for (r in reasons.filter { it.points != 0 }) {
            if (text.isNotEmpty()) text.append('\n')
            val start = text.length
            text.append(signed(r.points))
            text.setSpan(ForegroundColorSpan(pointsColour(view, r.points)), start, text.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            text.append("  ").append(r.text)
        }
        view.text = text
        view.visibility = if (text.isEmpty()) View.GONE else View.VISIBLE
    }

    private fun showPicture(card: ItemSuggestionBinding, first: String, second: String?) {
        card.picture.setImageBitmap(pictures[first])
        if (second != null) {
            card.pictureSecond.setImageBitmap(pictures[second])
            card.pictureSecond.visibility = View.VISIBLE
        }
    }

    private fun sectionTitle(text: Int): View =
        (layoutInflater.inflate(R.layout.item_section_title, binding.sections, false) as TextView).apply { setText(text) }

    private fun sectionNote(text: Int): View =
        (layoutInflater.inflate(R.layout.item_section_note, binding.sections, false) as TextView).apply { setText(text) }

    private fun signed(points: Int) = when {
        points > 0 -> "+$points"
        points < 0 -> "−${-points}"
        else -> "0"
    }

    private fun pointsColour(view: View, points: Int) = MaterialColors.getColor(view, when {
        points > 0 -> androidx.appcompat.R.attr.colorPrimary
        points < 0 -> androidx.appcompat.R.attr.colorError
        else -> com.google.android.material.R.attr.colorOnSurfaceVariant
    })

    /** A garment photo from the assets, decoded down to about [PICTURE_PX] high. */
    private fun picture(path: String): Bitmap? = try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        assets.open(path).use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (bounds.outHeight / (sample * 2) >= PICTURE_PX) sample *= 2
        assets.open(path).use { BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample }) }
    } catch (e: Exception) {
        Log.w(TAG, "No picture for $path", e)
        null
    }

    private fun faceShapeName(shape: FaceShape) = when (shape) {
        FaceShape.OVAL -> R.string.face_shape_oval
        FaceShape.ROUND -> R.string.face_shape_round
        FaceShape.SQUARE -> R.string.face_shape_square
        FaceShape.HEART -> R.string.face_shape_heart
        FaceShape.OBLONG -> R.string.face_shape_oblong
    }
}
