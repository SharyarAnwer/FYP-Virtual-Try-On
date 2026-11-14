package com.shahryar.virtualtryon

import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.View
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.widget.doAfterTextChanged
import com.google.android.material.chip.Chip
import com.google.android.material.color.MaterialColors
import com.shahryar.virtualtryon.databinding.ActivityProfileBinding

/**
 * The profile the recommendation engine ranks for (M7): gender, height, body type, favourite
 * colours and face shape. Every change is saved as it is made, in [ProfileStore].
 *
 * The face shape is Face Mesh's measurement, or one the user picks instead. "Measure with camera"
 * closes this screen with [RESULT_MEASURE_FACE]; the camera screen measures, saves the reading
 * and opens this screen again on it.
 */
class ProfileActivity : AppCompatActivity() {

    companion object {
        /** Result asking the camera screen to measure the face shape, then come back here. */
        const val RESULT_MEASURE_FACE = RESULT_FIRST_USER + 1

        const val MIN_HEIGHT_CM = 120
        const val MAX_HEIGHT_CM = 230

        /** Swatches for glasses frame colours no garment shares. */
        private val FRAME_COLOUR_HEX = mapOf(
            "silver" to "#C4C7CE", "brown" to "#5C3A21", "tortoiseshell" to "#7A4A24", "black" to "#151515",
        )
        private val FAMILY_ORDER = listOf("neutral", "warm", "cool")
    }

    private data class ColourOption(val name: String, val hex: String, val family: String?)

    private lateinit var binding: ActivityProfileBinding
    private lateinit var store: ProfileStore
    private var profile = Profile()
    private var faceMeasured = false

    /** True while the form is being filled from the store, so filling it saves nothing. */
    private var filling = false

    private val faceChipIds by lazy {
        mapOf(
            FaceShape.OVAL to binding.faceOval.id, FaceShape.ROUND to binding.faceRound.id,
            FaceShape.SQUARE to binding.faceSquare.id, FaceShape.HEART to binding.faceHeart.id,
            FaceShape.OBLONG to binding.faceOblong.id,
        )
    }
    private val genderButtonIds by lazy {
        mapOf(Gender.MALE to binding.genderMale.id, Gender.FEMALE to binding.genderFemale.id,
            Gender.UNSPECIFIED to binding.genderUnspecified.id)
    }
    private val bodyButtonIds by lazy {
        mapOf(BodyType.SLIM to binding.bodySlim.id, BodyType.AVERAGE to binding.bodyAverage.id,
            BodyType.BROAD to binding.bodyBroad.id)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityProfileBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(binding.profileRoot) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            val keyboard = insets.getInsets(WindowInsetsCompat.Type.ime())
            view.setPadding(bars.left, bars.top, bars.right, maxOf(bars.bottom, keyboard.bottom))
            insets
        }
        store = ProfileStore(this)

        binding.toolbar.setNavigationOnClickListener { finish() }
        binding.doneButton.setOnClickListener { finish() }
        binding.measureFaceButton.setOnClickListener {
            setResult(RESULT_MEASURE_FACE)
            finish()
        }

        binding.genderGroup.addOnButtonCheckedListener { _, id, checked ->
            if (!checked) return@addOnButtonCheckedListener
            val gender = genderButtonIds.entries.firstOrNull { it.value == id }?.key ?: return@addOnButtonCheckedListener
            update(profile.copy(gender = gender))
        }
        binding.bodyGroup.addOnButtonCheckedListener { group, _, _ ->
            val body = bodyButtonIds.entries.firstOrNull { it.value == group.checkedButtonId }?.key
            update(profile.copy(bodyType = body))
        }
        binding.heightInput.doAfterTextChanged { text ->
            if (filling) return@doAfterTextChanged
            val raw = text?.toString()?.trim().orEmpty()
            val height = raw.toIntOrNull()
            when {
                raw.isEmpty() -> {
                    binding.heightLayout.error = null
                    update(profile.copy(heightCm = null))
                }
                height == null || height !in MIN_HEIGHT_CM..MAX_HEIGHT_CM ->
                    binding.heightLayout.error = getString(R.string.profile_height_error, MIN_HEIGHT_CM, MAX_HEIGHT_CM)
                else -> {
                    binding.heightLayout.error = null
                    update(profile.copy(heightCm = height))
                }
            }
        }
        binding.faceChips.setOnCheckedStateChangeListener { _, checkedIds ->
            if (filling) return@setOnCheckedStateChangeListener
            val shape = faceChipIds.entries.firstOrNull { it.value in checkedIds }?.key
            faceMeasured = shape != null && store.faceReading()?.shape == shape
            update(profile.copy(faceShape = shape))
            showFaceStatus()
        }

        loadColours()
    }

    override fun onResume() {
        super.onResume()
        // The camera screen may have measured the face meanwhile.
        fill()
    }

    /** Sets the form from the saved profile. */
    private fun fill() {
        filling = true
        profile = store.load()
        faceMeasured = store.isFaceShapeMeasured()
        binding.genderGroup.check(genderButtonIds.getValue(profile.gender))
        binding.bodyGroup.clearChecked()
        profile.bodyType?.let { binding.bodyGroup.check(bodyButtonIds.getValue(it)) }
        binding.heightInput.setText(profile.heightCm?.toString().orEmpty())
        binding.heightLayout.error = null
        binding.faceChips.clearCheck()
        profile.faceShape?.let { binding.faceChips.check(faceChipIds.getValue(it)) }
        for (i in 0 until binding.colourChips.childCount) {
            val chip = binding.colourChips.getChildAt(i) as Chip
            chip.isChecked = chip.tag in profile.colours
        }
        showFaceReading()
        showFaceStatus()
        filling = false
    }

    private fun update(changed: Profile) {
        if (filling) return
        profile = changed
        store.save(profile, faceMeasured)
    }

    private fun showFaceReading() {
        val reading = store.faceReading()
        if (reading == null) {
            binding.faceMeasuredText.setText(R.string.profile_face_not_measured)
            binding.faceDetailText.visibility = View.GONE
            binding.measureFaceButton.setText(R.string.profile_face_measure)
        } else {
            val p = reading.proportions
            binding.faceMeasuredText.text = getString(R.string.profile_face_measured, getString(faceShapeName(reading.shape)))
            binding.faceDetailText.text = getString(R.string.profile_face_detail,
                p.lengthToWidth, p.foreheadToWidth, p.jawToWidth, p.jawAngleDeg)
            binding.faceDetailText.visibility = View.VISIBLE
            binding.measureFaceButton.setText(R.string.profile_face_remeasure)
        }
    }

    private fun showFaceStatus() {
        binding.faceStatusText.setText(when {
            profile.faceShape == null -> R.string.profile_face_none
            faceMeasured -> R.string.profile_face_using_measured
            else -> R.string.profile_face_using_picked
        })
    }

    /** One chip per colour in the garment and glasses catalogs, read off the main thread. */
    private fun loadColours() {
        Thread {
            val garments = GarmentCatalog.load(this).filterNot { it.isPlaceholder }
            val pairs = GlassesShape.catalog(this)
            val options = colourOptions(garments, pairs)
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                binding.colourChips.removeAllViews()
                for (option in options) binding.colourChips.addView(colourChip(option))
                fill()
            }
        }.start()
    }

    private fun colourOptions(garments: List<Garment>, pairs: List<GlassesPair>): List<ColourOption> {
        val byName = LinkedHashMap<String, ColourOption>()
        for (g in garments) {
            byName.getOrPut(g.colourName) {
                ColourOption(g.colourName, g.colourHex, g.colourFamily ?: Recommender.COLOUR_FAMILIES[g.colourName])
            }
        }
        for (p in pairs) {
            val name = p.frameColour ?: continue
            byName.getOrPut(name) {
                ColourOption(name, FRAME_COLOUR_HEX[name] ?: "#808080", Recommender.COLOUR_FAMILIES[name])
            }
        }
        return byName.values.sortedWith(compareBy<ColourOption> {
            FAMILY_ORDER.indexOf(it.family).let { i -> if (i < 0) FAMILY_ORDER.size else i }
        }.thenBy { it.name })
    }

    private fun colourChip(option: ColourOption): Chip {
        val chip = layoutInflater.inflate(R.layout.chip_colour, binding.colourChips, false) as Chip
        chip.text = option.name.replaceFirstChar { it.uppercase() }
        chip.tag = option.name
        val density = resources.displayMetrics.density
        chip.chipIcon = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(Color.parseColor(option.hex))
            setStroke((1 * density).toInt(), MaterialColors.getColor(chip, com.google.android.material.R.attr.colorOutline))
        }
        chip.isChipIconVisible = true
        chip.setOnCheckedChangeListener { button, checked ->
            val name = button.tag as String
            update(profile.copy(colours = if (checked) profile.colours + name else profile.colours - name))
        }
        return chip
    }

    private fun faceShapeName(shape: FaceShape) = when (shape) {
        FaceShape.OVAL -> R.string.face_shape_oval
        FaceShape.ROUND -> R.string.face_shape_round
        FaceShape.SQUARE -> R.string.face_shape_square
        FaceShape.HEART -> R.string.face_shape_heart
        FaceShape.OBLONG -> R.string.face_shape_oblong
    }
}
