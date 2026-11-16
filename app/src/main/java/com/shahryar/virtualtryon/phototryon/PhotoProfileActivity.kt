package com.shahryar.virtualtryon.phototryon

import android.app.AlertDialog
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.view.View
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import com.shahryar.virtualtryon.Garment
import com.shahryar.virtualtryon.GarmentCatalog
import com.shahryar.virtualtryon.ProfileStore
import com.shahryar.virtualtryon.R
import com.shahryar.virtualtryon.databinding.ActivityPhotoProfileBinding
import com.shahryar.virtualtryon.databinding.ItemPhotoOutfitBinding
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * Photo Try-On, stages 2 and 3: shows what the four photos say about the wearer - measurements,
 * skin tone, face shape - and the outfits ranked for them, each with its reasons and a button to
 * try it on.
 *
 * While the wearer reads this, the engine is already dressing the first outfit, so that tapping
 * it shows a picture straight away rather than starting a wait.
 *
 * The photos leave the phone only for the engine, and only after the wearer has said yes; the
 * answer is kept with the capture, so it is asked once.
 */
class PhotoProfileActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "VTO-Photo"
        const val EXTRA_PHOTOS_DIR = GuidedCaptureActivity.EXTRA_PHOTOS_DIR

        /** The file in a capture's folder that records the wearer agreeing to send its photos to the engine. */
        const val CONSENT_FILE = "consent"

        private const val OUTFITS_SHOWN = 6
        private const val REASONS_SHOWN = 4
        private const val PICTURE_PX = 256
    }

    private lateinit var binding: ActivityPhotoProfileBinding
    private lateinit var photosDir: File
    private lateinit var settings: EngineSettings
    private val worker = Executors.newSingleThreadExecutor()
    private val network = Executors.newSingleThreadExecutor()
    private var outfits: List<RankedOutfit> = emptyList()
    private var engineUp = false
    private var preDressed = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPhotoProfileBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val path = intent.getStringExtra(EXTRA_PHOTOS_DIR)
        if (path == null) {
            finish()
            return
        }
        photosDir = File(path)
        settings = EngineSettings(this)
        binding.engineButton.setOnClickListener { askAddress() }

        worker.execute {
            val analysis = PhotoAnalyzer(applicationContext).analyse(photosDir)
            val saved = ProfileStore(this).load()
            val profile = PhotoProfile(analysis.measurements, analysis.skin, analysis.faceShape, saved.gender, saved.colours)
            val garments = GarmentCatalog.load(this).filterNot { it.isPlaceholder }
            val ranked = PhotoRecommender.rank(profile, garments.filter { it.category == "upper_body" },
                garments.filter { it.category == "lower_body" }, OUTFITS_SHOWN)
            val pictures = ranked.flatMap { listOf(it.top, it.bottom) }.distinctBy { it.id }
                .associate { g -> g.id to g.frontPath?.let { picture(it) } }
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                outfits = ranked
                showProfile(analysis)
                showOutfits(ranked, pictures)
                preDress()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        checkEngine()
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.shutdown()
        network.shutdown()
    }

    // ------------------------------------------------------------------------------ the profile

    private fun showProfile(a: PhotoAnalysis) {
        binding.working.visibility = View.GONE
        val m = a.measurements
        if (m == null) {
            binding.workingText.setText(R.string.photo_profile_failed)
            return
        }
        binding.workingText.visibility = View.GONE
        binding.profileCard.visibility = View.VISIBLE
        binding.measurementsText.text = getString(R.string.photo_profile_measurements, m.heightCm, m.shoulderCm, m.chestCm, m.waistCm, m.hipCm, m.armCm, m.legCm)
        binding.buildText.text = getString(R.string.photo_profile_build, PhotoRecommender.buildFor(m).id, PhotoRecommender.sizeFor(m.chestCm))
        val skin = a.skin
        if (skin != null) {
            binding.skinSwatch.setBackgroundColor(skin.rgb)
            binding.skinText.text = getString(R.string.photo_profile_skin, words(skin.depth.name), words(skin.undertone.name))
        } else {
            binding.skinSwatch.visibility = View.GONE
            binding.skinText.visibility = View.GONE
        }
        val face = a.faceShape
        if (face != null) binding.faceText.text = getString(R.string.photo_profile_face, face.id) else binding.faceText.visibility = View.GONE
        val notes = ArrayList<String>()
        if (!a.scaleKnown) notes += getString(R.string.photo_profile_note_no_scale)
        else if (!m.scaleFromFace) notes += getString(R.string.photo_profile_note_average_eyes)
        if (!m.girthsFromSide) notes += getString(R.string.photo_profile_note_no_side)
        notes += getString(R.string.photo_profile_note_clothes)
        binding.notesText.text = notes.joinToString("\n")
    }

    private fun words(name: String) = name.lowercase(Locale.US).replace('_', ' ')

    // ------------------------------------------------------------------------------ the outfits

    private fun showOutfits(ranked: List<RankedOutfit>, pictures: Map<String, Bitmap?>) {
        binding.outfitsTitle.visibility = View.VISIBLE
        binding.outfits.removeAllViews()
        for (outfit in ranked) {
            val card = ItemPhotoOutfitBinding.inflate(layoutInflater, binding.outfits, false)
            card.topPicture.setImageBitmap(pictures[outfit.top.id])
            card.bottomPicture.setImageBitmap(pictures[outfit.bottom.id])
            card.outfitName.text = "${outfit.top.name} + ${outfit.bottom.name}"
            if (outfit.size != null) card.outfitSize.text = getString(R.string.photo_profile_size, outfit.size) else card.outfitSize.visibility = View.GONE
            card.outfitReasons.text = outfit.reasons.sortedByDescending { abs(it.points) }.take(REASONS_SHOWN).joinToString("\n") {
                (if (it.points > 0) "+ " else if (it.points < 0) "− " else "· ") + it.text
            }
            card.tryButton.setOnClickListener { withConsent { open(outfit.top, outfit.bottom) } }
            binding.outfits.addView(card.root)
        }
    }

    private fun open(top: Garment, bottom: Garment) {
        startActivity(Intent(this, TryOnResultActivity::class.java)
            .putExtra(EXTRA_PHOTOS_DIR, photosDir.absolutePath)
            .putExtra(TryOnResultActivity.EXTRA_TOP, top.id)
            .putExtra(TryOnResultActivity.EXTRA_BOTTOM, bottom.id))
    }

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

    // ------------------------------------------------------------------------------ the engine

    /** Runs [then] once the wearer has agreed to their photos going to the engine; asks if they have not yet. */
    private fun withConsent(then: () -> Unit) {
        val flag = File(photosDir, CONSENT_FILE)
        if (flag.exists()) {
            then()
            return
        }
        AlertDialog.Builder(this).setMessage(R.string.result_consent)
            .setPositiveButton(R.string.result_send) { _, _ ->
                flag.writeText("yes")
                then()
            }
            .setNegativeButton(R.string.engine_cancel, null).show()
    }

    private fun askAddress() {
        val box = EditText(this).apply {
            setText(settings.address)
            setHint(R.string.engine_hint)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        }
        AlertDialog.Builder(this).setTitle(R.string.engine_title).setView(box)
            .setPositiveButton(R.string.engine_save) { _, _ ->
                settings.address = box.text.toString()
                checkEngine()
            }
            .setNegativeButton(R.string.engine_cancel, null).show()
    }

    private fun checkEngine() {
        val address = settings.address
        if (address.isBlank()) {
            binding.engineStatus.setText(R.string.engine_not_set)
            return
        }
        binding.engineStatus.setText(R.string.engine_checking)
        network.execute {
            val problem = try {
                TryOnClient(address).health()
                null
            } catch (e: Exception) {
                e.message ?: "unknown"
            }
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                engineUp = problem == null
                binding.engineStatus.text = if (problem == null) getString(R.string.engine_connected) else getString(R.string.engine_unreachable, problem)
                preDress()
            }
        }
    }

    /**
     * Starts the engine on the first outfit before the wearer asks for it. Only once they have agreed
     * to their photos being sent - asked here, as soon as the engine is reachable and the outfits are up.
     */
    private fun preDress() {
        val first = outfits.firstOrNull() ?: return
        if (!engineUp || preDressed) return
        if (Results.complete(photosDir, Results.dir(photosDir, first.top, first.bottom))) return
        preDressed = true
        withConsent {
            val address = settings.address
            network.execute {
                try {
                    TryOnRunner(applicationContext, TryOnClient(address), photosDir).request(first.top, first.bottom)
                } catch (e: Exception) {
                    Log.w(TAG, "Could not start the first outfit early", e)
                }
            }
        }
    }
}
