package com.shahryar.virtualtryon.videotryon

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.text.TextUtils
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.shahryar.virtualtryon.BuildConfig
import com.shahryar.virtualtryon.Garment
import com.shahryar.virtualtryon.GarmentCatalog
import com.shahryar.virtualtryon.R
import com.shahryar.virtualtryon.databinding.ActivityVideoDressBinding
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * Video Try-On, step 2: choose a top, a bottom or both; the video (uncropped) and the garments go to the video
 * engine; its progress is shown; then the dressed video plays on a loop, with the wearer's own beside it a tap
 * away. Each outfit's result is kept in the video's folder, so it is fetched once.
 */
class VideoDressActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "VTO-Video"
        private const val POLL_MS = 3000L
    }

    private lateinit var binding: ActivityVideoDressBinding
    private lateinit var dir: File
    private val worker = Executors.newSingleThreadExecutor()
    @Volatile private var closed = false

    private var tops: List<Garment> = emptyList()
    private var bottoms: List<Garment> = emptyList()
    private var top: Garment? = null
    private var bottom: Garment? = null
    private var showingOriginal = false
    private var resultDir: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVideoDressBinding.inflate(layoutInflater)
        setContentView(binding.root)
        dir = File(intent.getStringExtra(VideoCaptureActivity.EXTRA_DIR) ?: run { finish(); return })
        binding.dressButton.setOnClickListener { dress() }
        binding.retryButton.setOnClickListener { dress() }
        binding.otherClothesButton.setOnClickListener { showChoose() }
        binding.newVideoButton.setOnClickListener { finish() }
        binding.toggleButton.setOnClickListener {
            showingOriginal = !showingOriginal
            play()
        }
        worker.execute {
            // Only the real garments, the ones with product photos: the engine dresses from the photo.
            val all = GarmentCatalog.load(this).filter { !it.isPlaceholder && it.frontPath != null }
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                tops = all.filter { it.category == "upper_body" }
                bottoms = all.filter { it.category == "lower_body" }
                buildRow(binding.topsRow, tops, isTop = true)
                buildRow(binding.bottomsRow, bottoms, isTop = false)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        closed = true
        worker.shutdownNow()
    }

    // --- Choosing -----------------------------------------------------------------------------------------

    private fun buildRow(row: LinearLayout, garments: List<Garment>, isTop: Boolean) {
        row.removeAllViews()
        val d = resources.displayMetrics.density
        val width = (118 * d).roundToInt()
        for (garment in listOf<Garment?>(null) + garments) {
            val cell = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                val pad = (8 * d).roundToInt()
                setPadding(pad, pad, pad, pad)
            }
            // The garment's photo on a rounded white tile, or a quiet "None" tile.
            val picture: View = if (garment != null) {
                ImageView(this).apply {
                    scaleType = ImageView.ScaleType.FIT_CENTER
                    background = GradientDrawable().apply { setColor(Color.WHITE); cornerRadius = 12 * d }
                    clipToOutline = true
                    val p = (6 * d).roundToInt()
                    setPadding(p, p, p, p)
                    garment.frontPath?.let { setImageBitmap(thumbnail(it)) }
                }
            } else {
                TextView(this).apply {
                    text = getString(R.string.video_none)
                    gravity = Gravity.CENTER
                    textSize = 15f
                    setTextColor(ContextCompat.getColor(context, R.color.home_text_hint))
                    background = GradientDrawable().apply {
                        setColor(0x0FFFFFFF)
                        cornerRadius = 12 * d
                        setStroke((1 * d).roundToInt(), ContextCompat.getColor(context, R.color.home_card_stroke),
                            6 * d, 4 * d)
                    }
                }
            }
            cell.addView(picture, LinearLayout.LayoutParams(width, 0, 1f))
            cell.addView(TextView(this).apply {
                text = garment?.name ?: getString(R.string.video_none)
                setTextColor(Color.WHITE)
                textSize = 13f
                gravity = Gravity.CENTER
                maxLines = 2
                ellipsize = TextUtils.TruncateAt.END
            }, LinearLayout.LayoutParams(width, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = (6 * d).roundToInt() })
            cell.tag = garment
            cell.setOnClickListener {
                if (isTop) top = garment else bottom = garment
                markChosen()
            }
            row.addView(cell, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.MATCH_PARENT)
                .apply { marginEnd = (10 * d).roundToInt() })
        }
        markChosen()
    }

    private fun thumbnail(asset: String): Bitmap? = try {
        assets.open(asset).use { input ->
            BitmapFactory.decodeStream(input, null, BitmapFactory.Options().apply { inSampleSize = 4 })
        }
    } catch (e: Exception) {
        null
    }

    /** The chosen tile in each row: an accent outline and its name in bold; the others a faint outline. */
    private fun markChosen() {
        val d = resources.displayMetrics.density
        val card = ContextCompat.getColor(this, R.color.home_card)
        val faint = ContextCompat.getColor(this, R.color.home_card_stroke)
        val accent = ContextCompat.getColor(this, R.color.home_accent_video)
        for ((row, chosen) in listOf(binding.topsRow to top, binding.bottomsRow to bottom)) {
            for (i in 0 until row.childCount) {
                val cell = row.getChildAt(i) as LinearLayout
                val isChosen = cell.tag == chosen
                cell.background = GradientDrawable().apply {
                    setColor(card)
                    cornerRadius = 18 * d
                    if (isChosen) setStroke((3 * d).roundToInt(), accent) else setStroke((1 * d).roundToInt(), faint)
                }
                (cell.getChildAt(1) as TextView).setTypeface(null, if (isChosen) Typeface.BOLD else Typeface.NORMAL)
            }
        }
        binding.dressButton.isEnabled = top != null || bottom != null
        binding.dressButton.alpha = if (binding.dressButton.isEnabled) 1f else 0.45f
    }

    private fun showChoose() {
        binding.videoView.stopPlayback()
        binding.choosePanel.visibility = View.VISIBLE
        binding.waitPanel.visibility = View.GONE
        binding.resultPanel.visibility = View.GONE
    }

    // --- Dressing -----------------------------------------------------------------------------------------

    private fun outfitDir(top: Garment?, bottom: Garment?) =
        File(dir, "results/${top?.id ?: "none"}__${bottom?.id ?: "none"}").apply { mkdirs() }

    private fun dress() {
        val top = top
        val bottom = bottom
        if (top == null && bottom == null) return
        val out = outfitDir(top, bottom)
        if (File(out, "dressed.mp4").exists() && File(out, "original.mp4").exists()) {   // made before: no engine
            showResult(out)
            return
        }
        binding.choosePanel.visibility = View.GONE
        binding.resultPanel.visibility = View.GONE
        binding.waitPanel.visibility = View.VISIBLE
        binding.retryButton.visibility = View.GONE
        binding.stageText.setText(R.string.video_sending)
        binding.elapsedText.text = ""
        worker.execute {
            try {
                val client = VideoTryOnClient(BuildConfig.VIDEO_TRYON_ENGINE)
                if (!BuildConfig.VIDEO_TRYON_ENGINE.startsWith("http")) throw VideoEngineException(getString(R.string.video_no_engine))
                val session = dir.name
                client.health()
                client.putVideo(session, VideoCaptureActivity.videoFile(dir))
                for (g in listOfNotNull(top, bottom)) client.putGarment(g.id, assets.open(g.frontPath!!).use { it.readBytes() })
                val job = client.startJob(session, top?.id, top?.name, bottom?.id, bottom?.name)
                var status = client.status(job)
                while (!status.finished && !closed) {
                    val s = status
                    runOnUiThread { showStage(s) }
                    Thread.sleep(POLL_MS)
                    status = client.status(job)
                }
                if (closed) return@execute
                if (status.state == "error") throw VideoEngineException(status.error ?: "the engine stopped")
                client.download(job, "dressed.mp4", File(out, "dressed.mp4"))
                client.download(job, "original.mp4", File(out, "original.mp4"))
                runOnUiThread { if (!isDestroyed) showResult(out) }
            } catch (e: InterruptedException) {
                return@execute
            } catch (e: Exception) {
                Log.e(TAG, "Video try-on failed", e)
                val message = (e.message ?: e.toString()).lines().last { it.isNotBlank() }.take(300)
                runOnUiThread {
                    if (isDestroyed) return@runOnUiThread
                    binding.stageText.text = getString(R.string.video_failed, message)
                    binding.elapsedText.text = ""
                    binding.retryButton.visibility = View.VISIBLE
                }
            }
        }
    }

    private fun showStage(s: VideoJobStatus) {
        if (isDestroyed) return
        binding.stageText.text = getString(R.string.video_stage, s.stage.replaceFirstChar { it.uppercase() })
        val seconds = s.elapsed.roundToInt()
        binding.elapsedText.text = getString(R.string.video_elapsed, seconds / 60, seconds % 60)
    }

    // --- Watching -----------------------------------------------------------------------------------------

    private fun showResult(out: File) {
        resultDir = out
        showingOriginal = false
        binding.choosePanel.visibility = View.GONE
        binding.waitPanel.visibility = View.GONE
        binding.resultPanel.visibility = View.VISIBLE
        play()
    }

    private fun play() {
        val out = resultDir ?: return
        val file = File(out, if (showingOriginal) "original.mp4" else "dressed.mp4")
        binding.toggleButton.setText(if (showingOriginal) R.string.video_show_dressed else R.string.video_show_original)
        binding.videoView.setOnPreparedListener { player ->
            player.isLooping = true
            player.start()
        }
        binding.videoView.setVideoPath(file.absolutePath)
    }
}
