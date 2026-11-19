package com.shahryar.virtualtryon.videotryon

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.MediaMetadataRetriever
import android.media.ToneGenerator
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.ContextCompat
import com.shahryar.virtualtryon.R
import com.shahryar.virtualtryon.databinding.ActivityVideoCaptureBinding
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Video Try-On, step 1: the wearer records a 10-second video of themselves with the front camera (a spoken
 * countdown gives them time to step back), or chooses a video of at most 10 seconds from the phone.
 * The video is kept in its own folder and handed to [VideoDressActivity].
 */
class VideoCaptureActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "VTO-Video"
        const val EXTRA_DIR = "video_dir"
        const val MAX_MS = 10_000L
        private const val MAX_ALLOWED_MS = 10_500L          // a chosen video may run a little over 10 s
        private const val MIN_MS = 3_000L
        private const val COUNTDOWN = 8                     // seconds to step back; the spoken hint fits before "3"

        /** All of the Video Try-On's videos and results. */
        fun root(activity: AppCompatActivity) = File(activity.filesDir, "video_tryon")
        fun videoFile(dir: File) = File(dir, "video.mp4")
    }

    private lateinit var binding: ActivityVideoCaptureBinding
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private val tones = runCatching { ToneGenerator(AudioManager.STREAM_MUSIC, 90) }.getOrNull()
    private val handler = Handler(Looper.getMainLooper())
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var counting = false

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) showRecorder() else Toast.makeText(this, R.string.video_permission_needed, Toast.LENGTH_LONG).show()
    }

    private val videoPicker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) useChosen(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVideoCaptureBinding.inflate(layoutInflater)
        setContentView(binding.root)
        tts = TextToSpeech(this, this)
        binding.backButton.setOnClickListener { finish() }
        binding.recordButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                showRecorder()
            } else {
                cameraPermission.launch(Manifest.permission.CAMERA)
            }
        }
        binding.uploadButton.setOnClickListener {
            videoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
        }
        binding.cancelButton.setOnClickListener { stopAll(); showChoice() }
        binding.startButton.setOnClickListener { startCountdown() }
    }

    override fun onInit(status: Int) {
        val voice = tts ?: return
        if (status == TextToSpeech.SUCCESS) {
            if (voice.setLanguage(Locale.US) < TextToSpeech.LANG_AVAILABLE) voice.setLanguage(Locale.getDefault())
            ttsReady = true
        }
    }

    override fun onPause() {
        super.onPause()
        if (counting || recording != null) {             // leaving the screen stops a recording half made
            stopAll()
            showChoice()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)
        recording?.close()
        tts?.shutdown()
        tones?.release()
    }

    private fun say(text: String) {
        if (ttsReady) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, text)
    }

    private fun showChoice() {
        binding.choicePanel.visibility = View.VISIBLE
        binding.recordPanel.visibility = View.GONE
        binding.previewView.visibility = View.GONE
        binding.countText.visibility = View.GONE
        if (videoCapture != null) runCatching { ProcessCameraProvider.getInstance(this).get().unbindAll() }
        videoCapture = null
    }

    /** The front camera on screen, ready to record. */
    private fun showRecorder() {
        binding.choicePanel.visibility = View.GONE
        binding.previewView.visibility = View.VISIBLE
        binding.recordPanel.visibility = View.VISIBLE
        binding.startButton.isEnabled = true
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val preview = Preview.Builder().build().also { it.surfaceProvider = binding.previewView.surfaceProvider }
                val recorder = Recorder.Builder()
                    .setQualitySelector(QualitySelector.from(Quality.HD, FallbackStrategy.lowerQualityOrHigherThan(Quality.SD)))
                    .build()
                val capture = VideoCapture.withOutput(recorder)
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, capture)
                videoCapture = capture
            } catch (e: Exception) {
                Log.e(TAG, "Camera could not start", e)
                Toast.makeText(this, R.string.video_record_failed, Toast.LENGTH_LONG).show()
                showChoice()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** Start pressed: the wearer has COUNTDOWN seconds to step back, counted out loud for the last three. */
    private fun startCountdown() {
        if (videoCapture == null || counting) return
        counting = true
        binding.recordPanel.visibility = View.GONE
        binding.countText.visibility = View.VISIBLE
        say(getString(R.string.video_say_step_back))
        for (k in 0 until COUNTDOWN) {
            val left = COUNTDOWN - k
            handler.postDelayed({
                binding.countText.text = left.toString()
                if (left <= 3) {
                    say(left.toString())
                    tones?.startTone(ToneGenerator.TONE_PROP_BEEP, 120)
                }
            }, k * 1000L)
        }
        handler.postDelayed({ counting = false; record() }, COUNTDOWN * 1000L)
    }

    private fun record() {
        val capture = videoCapture ?: return
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(root(this), "rec_$stamp").apply { mkdirs() }
        val file = videoFile(dir)
        tones?.startTone(ToneGenerator.TONE_PROP_ACK, 250)
        say(getString(R.string.video_say_recording))
        binding.countText.text = getString(R.string.video_recording, (MAX_MS / 1000).toInt())
        val options = FileOutputOptions.Builder(file).setDurationLimitMillis(MAX_MS).build()
        recording = capture.output.prepareRecording(this, options)
            .start(ContextCompat.getMainExecutor(this)) { event ->
                when (event) {
                    is VideoRecordEvent.Status -> {
                        val done = (event.recordingStats.recordedDurationNanos / 1_000_000_000L).toInt()
                        binding.countText.text = getString(R.string.video_recording, ((MAX_MS / 1000).toInt() - done).coerceAtLeast(0))
                    }
                    is VideoRecordEvent.Finalize -> {
                        recording = null
                        val ok = !event.hasError() || event.error == VideoRecordEvent.Finalize.ERROR_DURATION_LIMIT_REACHED
                        if (ok && file.length() > 0) {
                            tones?.startTone(ToneGenerator.TONE_PROP_ACK, 400)
                            say(getString(R.string.video_say_done))
                            openDressing(dir)
                        } else {
                            Log.e(TAG, "Recording failed: error ${event.error}", event.cause)
                            dir.deleteRecursively()
                            Toast.makeText(this, R.string.video_record_failed, Toast.LENGTH_LONG).show()
                            showRecorder()
                        }
                    }
                }
            }
    }

    private fun stopAll() {
        handler.removeCallbacksAndMessages(null)
        counting = false
        recording?.stop()
        recording = null
        tts?.stop()
    }

    /** A video from the phone: copied into its own folder if it runs 3 to 10 seconds. */
    private fun useChosen(uri: Uri) {
        val ms = try {
            MediaMetadataRetriever().run {
                setDataSource(this@VideoCaptureActivity, uri)
                val d = extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()
                release()
                d
            }
        } catch (e: Exception) {
            null
        }
        when {
            ms == null -> { Toast.makeText(this, R.string.video_unreadable, Toast.LENGTH_LONG).show(); return }
            ms > MAX_ALLOWED_MS -> {
                Toast.makeText(this, getString(R.string.video_too_long, ((ms + 500) / 1000).toInt()), Toast.LENGTH_LONG).show(); return
            }
            ms < MIN_MS -> { Toast.makeText(this, R.string.video_too_short, Toast.LENGTH_LONG).show(); return }
        }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(root(this), "up_$stamp").apply { mkdirs() }
        try {
            contentResolver.openInputStream(uri)!!.use { input -> videoFile(dir).outputStream().use { input.copyTo(it) } }
        } catch (e: Exception) {
            Log.e(TAG, "Could not copy the chosen video", e)
            dir.deleteRecursively()
            Toast.makeText(this, R.string.video_unreadable, Toast.LENGTH_LONG).show()
            return
        }
        openDressing(dir)
    }

    private fun openDressing(dir: File) {
        startActivity(Intent(this, VideoDressActivity::class.java).putExtra(EXTRA_DIR, dir.absolutePath))
        showChoice()
    }
}
