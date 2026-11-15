package com.shahryar.virtualtryon.phototryon

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import com.shahryar.virtualtryon.Garment
import com.shahryar.virtualtryon.GarmentCatalog
import com.shahryar.virtualtryon.R
import com.shahryar.virtualtryon.databinding.ActivityTryonResultBinding
import java.io.File
import kotlin.math.roundToInt

/**
 * Photo Try-On, stage 5: one outfit on the wearer, as a turntable of real photos. The engine
 * dresses the capture's eight views - one every 45 degrees round the wearer - and they are shown
 * one at a time: drag a finger across the picture and it steps to the next view, each fading into
 * the next. It is how shops show a product from all sides, and the reason it is used here is that
 * every view is a photograph of the wearer: nothing about them is made up except the clothes.
 *
 * The views arrive one by one, front first, and the turntable works from the first of them: a view
 * that has not arrived yet is stepped over. Every view is saved with the capture as it arrives, so
 * an outfit seen once is shown again from the phone, with no engine and no wait.
 */
class TryOnResultActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "VTO-Photo"
        const val EXTRA_TOP = "top"
        const val EXTRA_BOTTOM = "bottom"
        private const val POLL_MS = 1000L

        /** How far a finger moves, in dp, for the turntable to step one view. */
        private const val STEP_DP = 44f
        private const val FADE_MS = 140L
    }

    private lateinit var binding: ActivityTryonResultBinding
    private lateinit var photosDir: File
    private lateinit var resultDir: File
    private var top: Garment? = null
    private var bottom: Garment? = null
    private val frames = arrayOfNulls<Bitmap>(Photos.VIEWS.size)
    private val dots = ArrayList<View>()
    private var showing = -1
    private var chosen = false
    @Volatile private var stopped = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTryonResultBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val path = intent.getStringExtra(PhotoProfileActivity.EXTRA_PHOTOS_DIR)
        if (path == null) {
            finish()
            return
        }
        photosDir = File(path)
        val garments = GarmentCatalog.load(this)
        top = garments.firstOrNull { it.id == intent.getStringExtra(EXTRA_TOP) }
        bottom = garments.firstOrNull { it.id == intent.getStringExtra(EXTRA_BOTTOM) }
        resultDir = Results.dir(photosDir, top, bottom)
        binding.resultTitle.text = listOfNotNull(top?.name, bottom?.name).joinToString(" + ")
        setUpDots()
        setUpDrag()

        // Whatever is already on the phone is shown at once.
        for ((i, view) in Photos.VIEWS.withIndex()) if (Results.file(resultDir, view).exists()) arrived(i)
        if (Results.complete(photosDir, resultDir)) {
            binding.resultWorking.visibility = View.GONE
            binding.resultStatus.setText(R.string.result_saved)
        } else {
            askThenFetch()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopped = true
    }

    // ------------------------------------------------------------------------------ fetching

    private fun askThenFetch() {
        val flag = File(photosDir, PhotoProfileActivity.CONSENT_FILE)
        if (flag.exists()) {
            fetch()
            return
        }
        AlertDialog.Builder(this).setMessage(R.string.result_consent)
            .setPositiveButton(R.string.result_send) { _, _ ->
                flag.writeText("yes")
                fetch()
            }
            .setNegativeButton(R.string.engine_cancel) { _, _ -> finish() }.setCancelable(false).show()
    }

    /** Asks the engine for the outfit and brings each view over as it is finished. */
    private fun fetch() {
        binding.resultStatus.setText(R.string.result_starting)
        val address = EngineSettings(this).address
        Thread {
            try {
                val client = TryOnClient(address)
                val job = TryOnRunner(applicationContext, client, photosDir).request(top, bottom)
                while (!stopped) {
                    val status = client.status(job)
                    for (part in status.parts) {
                        val index = Photos.VIEWS.indexOf(part)
                        val file = if (index >= 0) Results.file(resultDir, part) else continue
                        if (!file.exists()) {
                            client.download(job, TryOnClient.FILES.getValue(part), file)
                            ui { arrived(index) }
                        }
                    }
                    ui {
                        when {
                            status.error != null -> failed(status.error)
                            status.finished -> {
                                binding.resultWorking.visibility = View.GONE
                                binding.resultStatus.setText(R.string.result_done)
                            }
                            status.stage.isNotBlank() ->
                                binding.resultStatus.text = getString(R.string.result_stage, status.stage.replaceFirstChar { it.uppercase() })
                        }
                    }
                    if (status.finished) break
                    Thread.sleep(POLL_MS)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Fetching the outfit failed", e)
                ui { failed(e.message ?: "unknown") }
            }
        }.start()
    }

    private fun ui(block: () -> Unit) = runOnUiThread { if (!isDestroyed && !stopped) block() }

    private fun failed(why: String) {
        binding.resultWorking.visibility = View.GONE
        binding.resultStatus.text = getString(R.string.result_error, why.lines().lastOrNull { it.isNotBlank() } ?: why)
    }

    /** A view is on the phone: read it in, and show it if nothing better is showing. */
    private fun arrived(index: Int) {
        frames[index] = BitmapFactory.decodeFile(Results.file(resultDir, Photos.VIEWS[index]).path) ?: return
        // The front is what the wearer wants first, unless they have already turned it themselves.
        if (showing < 0 || (index == 0 && !chosen)) show(index) else paintDots()
        if (frames.count { it != null } >= 2) binding.turnHint.visibility = if (chosen) View.GONE else View.VISIBLE
    }

    // ------------------------------------------------------------------------------ the turntable

    /** Fades from the view showing to view [index]. */
    private fun show(index: Int) {
        val picture = frames[index] ?: return
        if (showing >= 0) binding.pictureBelow.setImageBitmap(frames[showing])
        binding.pictureAbove.animate().cancel()
        binding.pictureAbove.alpha = if (showing >= 0) 0f else 1f
        binding.pictureAbove.setImageBitmap(picture)
        binding.pictureAbove.animate().alpha(1f).setDuration(FADE_MS).start()
        showing = index
        paintDots()
    }

    /** Steps [by] views round, over any that have not arrived. */
    private fun turn(by: Int) {
        if (showing < 0 || by == 0) return
        val n = frames.size
        val step = if (by > 0) 1 else -1
        var at = showing
        repeat(kotlin.math.abs(by)) {
            for (k in 1..n) {
                val next = ((at + step * k) % n + n) % n
                if (frames[next] != null) {
                    at = next
                    break
                }
            }
        }
        if (at != showing) show(at)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setUpDrag() {
        val stepPx = STEP_DP * resources.displayMetrics.density
        var startX = 0f
        var used = 0f
        binding.turntable.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    startX = event.x
                    used = 0f
                }
                MotionEvent.ACTION_MOVE -> {
                    val steps = ((event.x - startX - used) / stepPx).toInt()
                    if (steps != 0) {
                        used += steps * stepPx
                        chosen = true
                        binding.turnHint.visibility = View.GONE
                        // Dragging to the left brings round the next view to the wearer's left.
                        turn(-steps)
                    }
                }
            }
            true
        }
    }

    private fun setUpDots() {
        val size = (10 * resources.displayMetrics.density).roundToInt()
        val gap = (5 * resources.displayMetrics.density).roundToInt()
        repeat(frames.size) {
            val dot = View(this).apply { background = GradientDrawable().apply { shape = GradientDrawable.OVAL } }
            binding.dots.addView(dot, LinearLayout.LayoutParams(size, size).apply { setMargins(gap, 0, gap, 0) })
            dots += dot
        }
        paintDots()
    }

    /** White for the view showing, grey for one that has arrived, dark for one still to come. */
    private fun paintDots() {
        for ((i, dot) in dots.withIndex()) {
            (dot.background as GradientDrawable).setColor(when {
                i == showing -> 0xFFFFFFFF.toInt()
                frames[i] != null -> 0xFF8A9499.toInt()
                else -> 0xFF2A3136.toInt()
            })
        }
    }
}
