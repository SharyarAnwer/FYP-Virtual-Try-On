package com.shahryar.virtualtryon.phototryon

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.shahryar.virtualtryon.R
import com.shahryar.virtualtryon.databinding.ActivityUploadSortBinding
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * Photos chosen from the gallery, sorted into the eight views by which way the person faces in each (see
 * [UploadSorter]) and shown in their places to check. Tapping two places swaps them, so a photo the
 * sorting got wrong is put right in two taps; one it could not read is marked "?".
 *
 * Continue renames the photos to the views they are in and goes on to the profile, as a capture does.
 */
class UploadSortActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_DIR = GuidedCaptureActivity.EXTRA_PHOTOS_DIR

        /** The name an uploaded photo has until it is sorted: its place in the order picked. */
        fun pickFile(dir: File, index: Int) = File(dir, "pick_$index.jpg")
    }

    private lateinit var binding: ActivityUploadSortBinding
    private lateinit var dir: File
    private val worker = Executors.newSingleThreadExecutor()

    /** For each view, the photo in it (an index into [picks]), or -1. */
    private val placed = IntArray(CaptureGuide.VIEWS) { -1 }
    private var picks: List<File> = emptyList()
    private var thumbs: List<Bitmap?> = emptyList()
    private var unsure: Set<Int> = emptySet()
    private val cells = ArrayList<FrameLayout>()
    private var selected = -1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityUploadSortBinding.inflate(layoutInflater)
        setContentView(binding.root)
        dir = File(intent.getStringExtra(EXTRA_DIR) ?: run { finish(); return })
        picks = generateSequence(0) { it + 1 }.map { pickFile(dir, it) }.takeWhile { it.exists() }.toList()
        buildCells()
        binding.againButton.setOnClickListener {
            dir.deleteRecursively()
            finish()
        }
        binding.continueButton.setOnClickListener { finishSorting() }

        worker.execute {
            val directions = UploadSorter(applicationContext).directions(picks)
            val small = picks.map { Photos.loadUpright(it, 500) }
            val views = UploadSorter.assign(directions.map { it.angleDeg })
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                thumbs = small
                unsure = directions.indices.filter { directions[it].angleDeg == null }.toSet()
                for ((photo, view) in views.withIndex()) if (view >= 0) placed[view] = photo
                binding.sortNote.setText(R.string.upload_sort_note)
                refresh()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.shutdown()
    }

    private fun buildCells() {
        val labels = resources.getStringArray(R.array.upload_view_names)
        val margin = (3 * resources.displayMetrics.density).roundToInt()
        for (view in 0 until CaptureGuide.VIEWS) {
            val cell = FrameLayout(this)
            cell.addView(ImageView(this).apply { scaleType = ImageView.ScaleType.CENTER_CROP },
                FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
            cell.addView(TextView(this).apply {
                text = labels[view]
                setTextColor(Color.WHITE)
                textSize = 12f
                setBackgroundColor(0x99000000.toInt())
                gravity = Gravity.CENTER
            }, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM))
            cell.addView(TextView(this).apply {
                text = "?"
                setTextColor(Color.BLACK)
                textSize = 18f
                gravity = Gravity.CENTER
                background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(CaptureOverlayView.AMBER) }
            }, FrameLayout.LayoutParams((28 * resources.displayMetrics.density).roundToInt(),
                (28 * resources.displayMetrics.density).roundToInt(), Gravity.TOP or Gravity.END))
            cell.setOnClickListener { tapped(view) }
            val row = if (view < CaptureGuide.VIEWS / 2) binding.rowTop else binding.rowBottom
            row.addView(cell, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f).apply { setMargins(margin, margin, margin, margin) })
            cells += cell
        }
        refresh()
    }

    /** First tap picks a place; the second swaps the two (a photo moves into an empty place). */
    private fun tapped(view: Int) {
        if (selected < 0) {
            if (placed[view] >= 0) selected = view
        } else {
            val other = selected
            selected = -1
            if (other != view) placed[other] = placed[view].also { placed[view] = placed[other] }
        }
        refresh()
    }

    private fun refresh() {
        for ((view, cell) in cells.withIndex()) {
            val photo = placed[view]
            (cell.getChildAt(0) as ImageView).setImageBitmap(if (photo >= 0) thumbs.getOrNull(photo) else null)
            cell.getChildAt(2).visibility = if (photo >= 0 && photo in unsure) android.view.View.VISIBLE else android.view.View.GONE
            cell.background = GradientDrawable().apply {
                setColor(0xFF1B2126.toInt())
                if (view == selected) setStroke((4 * resources.displayMetrics.density).roundToInt(), CaptureOverlayView.GREEN)
            }
        }
        val haveFront = placed[0] >= 0
        binding.continueButton.isEnabled = haveFront && thumbs.isNotEmpty()
        if (thumbs.isNotEmpty() && !haveFront) binding.sortNote.setText(R.string.upload_sort_need_front)
    }

    /** Each photo renamed to the view it is in; then on to the profile. */
    private fun finishSorting() {
        for (view in Photos.VIEWS) Photos.file(dir, view).delete()
        val renamed = ArrayList<Pair<File, File>>()
        for ((view, photo) in placed.withIndex()) {
            if (photo < 0) continue
            val staged = File(dir, "sorted_$view.jpg")
            if (picks[photo].renameTo(staged)) renamed += staged to Photos.file(dir, Photos.VIEWS[view])
        }
        for ((staged, target) in renamed) staged.renameTo(target)
        picks.forEach { it.delete() }
        if (!Photos.usable(dir)) {
            Toast.makeText(this, R.string.capture_upload_failed, Toast.LENGTH_LONG).show()
            return
        }
        startActivity(Intent(this, PhotoProfileActivity::class.java).putExtra(PhotoProfileActivity.EXTRA_PHOTOS_DIR, dir.absolutePath))
        finish()
    }
}
