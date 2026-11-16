package com.shahryar.virtualtryon.phototryon

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioManager
import android.media.MediaActionSound
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.shahryar.virtualtryon.FaceTracker
import com.shahryar.virtualtryon.PoseDetector
import com.shahryar.virtualtryon.R
import com.shahryar.virtualtryon.databinding.ActivityGuidedCaptureBinding
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.roundToInt

/**
 * Photo Try-On, stage 1: takes eight views of the wearer, one every 45 degrees round them, with
 * nobody touching the phone. Dressed and shown one after another, they are a turntable made of
 * real photos.
 *
 * The phone is stood up and the wearer walks back until all of them is in view, too far away to
 * read a screen or reach a button. So the phone talks and colours the whole screen; see
 * [CaptureGuide] for what it checks and [CaptureOverlayView] for what is shown.
 *
 * ## Three parts
 *
 * - **Setting up**, at the phone: stand it upright (the tilt sensor says how far off it is), turn
 *   the volume up, look at it for a moment - it measures the eyes' spacing, which is the photos'
 *   scale - and tap Start.
 * - **Guiding**, across the room. First the front photo: posed, counted down, taken at full size.
 *   Then one slow turn, all the way round: each time the body passes one of the other seven angles,
 *   the frame in hand is kept as that view, with a tick. Eight held poses were too much to ask.
 * - **Looking over** the eight views, back at the phone: use them or retake them.
 *
 * ## Not mirrored
 *
 * The preview is mirrored, as a front camera's always is, so the wearer sees themselves as in a
 * mirror. The frames given to the pose model and the pictures saved are not: the model's left
 * shoulder is then the wearer's own left, which is what "turn to your left" is checked against,
 * and the pictures show the wearer as others see them.
 *
 * The live try-on's [PoseDetector] and [FaceTracker] are used as they are.
 */
class GuidedCaptureActivity : AppCompatActivity(), SensorEventListener, TextToSpeech.OnInitListener {

    private enum class Phase { SETUP, GUIDING, REVIEW }

    /** The guided turn (a front photo, then views picked out of one turn), or poses of the wearer's own. */
    private enum class Mode { TURN, FREE }

    companion object {
        private const val TAG = "VTO-Photo"

        /** The folder under the app's own files that holds each capture's pictures. */
        const val PHOTOS_DIR = "photo_tryon"

        /** An intent's extra: the folder a capture's pictures are in. */
        const val EXTRA_PHOTOS_DIR = "photos_dir"

        /**
         * The size frames are analysed at, as the camera's sensor lies (sideways). Far more than the
         * pose model needs: the turn's seven views are these frames, and they must be big enough to dress.
         */
        private val FRAME_SIZE = Size(1440, 1080)
        private const val FRAME_JPEG_QUALITY = 92

        /** Leaning further than this from upright, degrees, the phone cuts off feet or heads. */
        private const val MAX_TILT_DEG = 12f

        /** Below this share of full media volume, the voice may not carry across a room. */
        private const val MIN_VOLUME = 0.6f

        /** A new instruction is spoken once it has stood this long, ms; "hold still" waits longer. */
        private const val SETTLE_MS = 500L
        private const val HOLD_STILL_SETTLE_MS = 1500L

        /** The same instruction is not repeated sooner than this, ms; "turn slowly", less often still. */
        private const val REPEAT_MS = 6000L
        private const val TURN_REPEAT_MS = 14_000L

        /** One instruction is not cut off by the next sooner than this, ms. */
        private const val MIN_GAP_MS = 1800L

        /** After "Good. Now turn slowly...", nothing more is said for this long, ms. */
        private const val AFTER_PHOTO_QUIET_MS = 6500L

        /** After a view is counted out loud, no other instruction for this long, ms: the count is the guidance. */
        private const val COUNT_QUIET_MS = 2500L

        /** With no front photo after this long, the wearer is told what else to try, ms. */
        private const val HELP_AFTER_MS = 45_000L

        /** The face is too dark to use below this mean brightness, 0..255; checked every few frames. */
        private const val MIN_FACE_LUMA = 55f
        private const val LIGHT_EVERY_FRAMES = 8

        /**
         * Measuring the eye spacing while the wearer is at the phone: an iris must be at least this
         * many pixels across to be measured, and this many readings are taken, their middle used.
         */
        private const val MIN_IRIS_PX = 12f
        private const val FACE_READINGS = 15
        private const val IRIS_A = 468
        private const val IRIS_B = 473
    }

    private lateinit var binding: ActivityGuidedCaptureBinding
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var sensorManager: SensorManager
    private lateinit var audioManager: AudioManager

    private val guide = CaptureGuide()
    private val freeGuide = FreePoseGuide()
    @Volatile private var mode = Mode.TURN

    // Free poses only, analysis thread: a pose model that tells two people apart, and the object detector,
    // run every few frames as it is the slower of the two.
    private var freePose: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker? = null
    private var spotter: ObjectSpotter? = null
    private var spotted: List<Spotted> = emptyList()
    private var freeFrames = 0
    private var wasAmber = false
    private var poseDetector: PoseDetector? = null
    private var faceTracker: FaceTracker? = null
    private val faceReadings = ArrayList<Float>()
    @Volatile private var faceIpdMm = Float.NaN
    private var imageCapture: ImageCapture? = null
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private val shutter = MediaActionSound()
    private val beeper = Beeper()

    @Volatile private var phase = Phase.SETUP
    @Volatile private var sessionDir: File? = null
    private var helpSaidAtMs = 0L

    // Speaking: which instruction is waiting to be said, and which was said last.
    private var pendingKey: String? = null
    private var pendingSinceMs = 0L
    private var spokenKey: String? = null
    private var spokenAtMs = 0L
    private var quietUntilMs = 0L
    private var lastCount = 0

    // The phone's tilt, from its accelerometer, smoothed.
    private val gravity = FloatArray(3)
    private var gravityPrimed = false
    private var statusShownAtMs = 0L

    // Analysis thread only.
    private var sourceBitmap: Bitmap? = null
    private var uprightBitmap: Bitmap? = null
    private var uprightCanvas: Canvas? = null
    private val transform = Matrix()
    private val bounds = RectF()
    private val blitPaint = Paint()
    private val imagePoints = FloatArray(CaptureGuide.POINTS * CaptureGuide.STRIDE)
    private val worldPoints = FloatArray(CaptureGuide.POINTS * CaptureGuide.STRIDE)
    private var frameCount = 0
    private var tooDark = false

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else Toast.makeText(this, R.string.capture_permission_needed, Toast.LENGTH_LONG).show()
    }

    /** The phone's own photo picker: no permission is needed, and only the chosen photos are shared with the app. */
    private val photoPicker = registerForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(CaptureGuide.VIEWS)) { chosen ->
        if (chosen.isNotEmpty()) useUploaded(chosen)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityGuidedCaptureBinding.inflate(layoutInflater)
        setContentView(binding.root)
        // As bright as the screen goes: the colour has to be seen from across a room.
        window.attributes = window.attributes.apply { screenBrightness = 1f }

        analysisExecutor = Executors.newSingleThreadExecutor()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        tts = TextToSpeech(this, this)
        shutter.load(MediaActionSound.SHUTTER_CLICK)
        analysisExecutor.execute {
            poseDetector = PoseDetector.create(applicationContext)
            faceTracker = runCatching { FaceTracker.create(applicationContext) }.getOrNull()
        }

        binding.overlay.show(null, "", 0, 0, 0, outline = true)
        binding.startButton.setOnClickListener { startGuiding(Mode.TURN) }
        binding.freeButton.setOnClickListener { startGuiding(Mode.FREE) }
        binding.doneButton.setOnClickListener { finishFree() }
        binding.retakeButton.setOnClickListener { startGuiding() }
        binding.useButton.setOnClickListener { usePhotos() }
        binding.useLastButton.setOnClickListener { lastCapture()?.let { openProfile(it) } }
        binding.uploadButton.setOnClickListener { askForPhotos() }
        binding.backButton.setOnClickListener { finish() }

        if (hasCameraPermission()) startCamera() else cameraPermission.launch(Manifest.permission.CAMERA)
    }

    override fun onResume() {
        super.onResume()
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
        binding.useLastButton.visibility = if (lastCapture() != null) View.VISIBLE else View.GONE
        showSetupStatus()
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
        tts?.stop()
        // Leaving in the middle of a capture: come back to the start, not to a voice talking to nobody.
        if (phase == Phase.GUIDING) showSetup()
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.shutdown()
        shutter.release()
        beeper.release()
        analysisExecutor.execute {
            poseDetector?.close()
            faceTracker?.close()
            runCatching { freePose?.close() }
            spotter?.close()
        }
        analysisExecutor.shutdown()
    }

    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) {
            Log.w(TAG, "Text to speech unavailable: $status")
            return
        }
        val voice = tts ?: return
        if (voice.setLanguage(Locale.US) < TextToSpeech.LANG_AVAILABLE) voice.setLanguage(Locale.getDefault())
        voice.setSpeechRate(0.95f)
        ttsReady = true
    }

    // ------------------------------------------------------------------------------ the phases

    private fun showSetup() {
        phase = Phase.SETUP
        binding.setupPanel.visibility = View.VISIBLE
        binding.reviewPanel.visibility = View.GONE
        binding.doneButton.visibility = View.GONE
        binding.overlay.show(null, "", 0, 0, 0, outline = true)
        showSetupStatus()
    }

    private fun startGuiding(chosen: Mode = Mode.TURN) {
        if (!hasCameraPermission()) {
            cameraPermission.launch(Manifest.permission.CAMERA)
            return
        }
        mode = chosen
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val prefix = if (chosen == Mode.FREE) "free" else "session"
        val dir = File(getExternalFilesDir(null), "$PHOTOS_DIR/${prefix}_$stamp").apply { mkdirs() }
        // The eye spacing measured at the phone goes with the pictures: it is their scale.
        if (!faceIpdMm.isNaN()) {
            File(dir, PhotoAnalyzer.SCALE_FILE).writeText(JSONObject().put(PhotoAnalyzer.SCALE_IPD_MM, faceIpdMm.toDouble()).toString())
        }
        sessionDir = dir
        synchronized(guide) { guide.restart() }
        synchronized(freeGuide) { freeGuide.restart() }
        wasAmber = false
        pendingKey = null
        spokenKey = null
        lastCount = 0
        val now = SystemClock.uptimeMillis()
        helpSaidAtMs = now
        binding.setupPanel.visibility = View.GONE
        binding.reviewPanel.visibility = View.GONE
        binding.doneButton.visibility = if (chosen == Mode.FREE) View.VISIBLE else View.GONE
        phase = Phase.GUIDING
        say(getString(if (chosen == Mode.FREE) R.string.free_start else R.string.capture_voice_start), key = null)
        quietUntilMs = now + 3500
    }

    // ------------------------------------------------------------------------------ free poses

    /** One frame of free poses, on the analysis thread: the two-person pose model, the object detector, the rules. */
    private fun analyzeFree(upright: Bitmap) {
        val pose = freePose ?: twoPersonPose(applicationContext,
            com.google.mediapipe.tasks.vision.core.RunningMode.VIDEO).also { freePose = it }
        val reading = pose.detectForVideo(com.google.mediapipe.framework.image.BitmapImageBuilder(upright).build(),
            SystemClock.uptimeMillis()).reading()
        if (freeFrames++ % 6 == 0) spotted = (spotter ?: ObjectSpotter.create(applicationContext).also { spotter = it }).spot(upright)
        if (reading.image != null && freeFrames % LIGHT_EVERY_FRAMES == 0) {
            reading.image.copyInto(imagePoints)
            tooDark = faceLuma(upright) < MIN_FACE_LUMA
        }
        val g = synchronized(freeGuide) {
            freeGuide.update(reading.image, reading.world, reading.people, upright.width.toFloat() / upright.height,
                spotted, tooDark, SystemClock.uptimeMillis())
        }
        runOnUiThread { renderFree(g) }
    }

    private fun renderFree(g: FreeGuidance) {
        if (phase != Phase.GUIDING || mode != Mode.FREE) return
        val now = SystemClock.uptimeMillis()
        val takenBits = (1 shl g.taken) - 1
        val text = when {
            g.verdict != null -> problemWords(g.verdict)
            g.samePose -> getString(R.string.free_same_pose)
            g.level == Level.NEARLY -> getString(R.string.free_hold)
            g.count == 0 -> getString(R.string.capture_taking)
            else -> ""
        }
        binding.overlay.show(g.level, text, g.count, takenBits, FreePoseGuide.MAX_PHOTOS, outline = false)

        // Amber: said at once, as it is the instruction to hold - the countdown follows when it is finished.
        val amber = g.level == Level.NEARLY
        if (amber && !wasAmber) say(text, key = "hold")
        wasAmber = amber
        if (g.count > 0) {
            if (g.count != lastCount) {
                lastCount = g.count
                say(getString(when (g.count) { 3 -> R.string.capture_count_3; 2 -> R.string.capture_count_2; else -> R.string.capture_count_1 }), key = null)
            }
            return
        }
        lastCount = 0
        if (g.capture) {
            takeFreePhoto(g.taken)
            return
        }
        if (amber) return

        val key = g.verdict?.problem?.name ?: if (g.samePose) "same" else return
        if (key != pendingKey) {
            pendingKey = key
            pendingSinceMs = now
        }
        if (now < quietUntilMs || now - pendingSinceMs < SETTLE_MS || now - spokenAtMs < MIN_GAP_MS) return
        if (key == spokenKey && now - spokenAtMs < REPEAT_MS) return
        say(text, key)
    }

    /** The words for a broken rule, as said and shown. */
    private fun problemWords(v: PoseVerdict): String = when (v.problem) {
        PoseProblem.NO_PERSON -> getString(R.string.free_no_person)
        PoseProblem.TWO_PEOPLE -> getString(R.string.free_two_people)
        PoseProblem.MOVE_AWAY -> getString(R.string.capture_move_away)
        PoseProblem.COME_CLOSER -> getString(R.string.capture_come_closer)
        PoseProblem.NOT_STANDING -> getString(R.string.free_not_standing)
        PoseProblem.ARMS_ON_BODY -> getString(R.string.free_arms_on_body)
        PoseProblem.HANDS_HIDDEN -> getString(R.string.free_hands_hidden)
        PoseProblem.HOLDING -> getString(R.string.free_holding, heldName(v.detail))
        PoseProblem.TOO_DARK -> getString(R.string.capture_too_dark)
    }

    private fun heldName(label: String?): String = resources.getStringArray(R.array.holdable_names)
        .map { it.split('|') }.firstOrNull { it[0] == label }?.get(1) ?: (label ?: "")

    /** A free pose's photo, full size, kept under the order it was taken in; the view it is in is decided later. */
    private fun takeFreePhoto(index: Int) {
        val capture = imageCapture
        val dir = sessionDir
        if (capture == null || dir == null) {
            synchronized(freeGuide) { freeGuide.onCaptureFailed() }
            return
        }
        capture.takePicture(ImageCapture.OutputFileOptions.Builder(UploadSortActivity.pickFile(dir, index)).build(),
            ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    if (isDestroyed || phase != Phase.GUIDING) return
                    shutter.play(MediaActionSound.SHUTTER_CLICK)
                    binding.overlay.flash()
                    val (taken, done) = synchronized(freeGuide) {
                        freeGuide.onCaptured()
                        freeGuide.taken to freeGuide.done
                    }
                    beeper.progress(taken)
                    if (done) {
                        beeper.done()
                        say(getString(R.string.free_all_done), key = null)
                        finishFree()
                    } else {
                        say(getString(R.string.free_taken, taken), key = "same")
                        pendingKey = spokenKey
                        pendingSinceMs = SystemClock.uptimeMillis()
                        quietUntilMs = pendingSinceMs + AFTER_PHOTO_QUIET_MS / 2
                    }
                }

                override fun onError(e: ImageCaptureException) {
                    Log.e(TAG, "Free pose photo failed", e)
                    synchronized(freeGuide) { freeGuide.onCaptureFailed() }
                    if (phase == Phase.GUIDING) say(getString(R.string.capture_failed), key = "failed")
                }
            })
    }

    /** The free poses taken so far go on to be sorted into views and checked, as uploaded photos are. */
    private fun finishFree() {
        val dir = sessionDir
        val taken = synchronized(freeGuide) { freeGuide.taken }
        showSetup()
        if (dir == null || taken == 0) return
        startActivity(Intent(this, UploadSortActivity::class.java).putExtra(UploadSortActivity.EXTRA_DIR, dir.absolutePath))
    }

    private fun showReview() {
        phase = Phase.REVIEW
        binding.overlay.show(null, "", 0, CaptureGuide.ALL_TAKEN, CaptureGuide.VIEWS, outline = false)
        binding.reviewPanel.visibility = View.VISIBLE
        val dir = sessionDir ?: return
        val pictures = ArrayList<ImageView>()
        for (row in listOf(binding.reviewRowTop, binding.reviewRowBottom)) {
            row.removeAllViews()
            repeat(CaptureGuide.VIEWS / 2) {
                val picture = ImageView(this).apply { scaleType = ImageView.ScaleType.CENTER_CROP }
                val margin = (3 * resources.displayMetrics.density).roundToInt()
                row.addView(picture, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f).apply { setMargins(margin, margin, margin, margin) })
                pictures += picture
            }
        }
        analysisExecutor.execute {
            for ((i, view) in Photos.VIEWS.withIndex()) {
                val small = Photos.loadUpright(Photos.file(dir, view), 700)
                runOnUiThread { if (!isDestroyed) pictures[i].setImageBitmap(small) }
            }
        }
    }

    private fun usePhotos() {
        val dir = sessionDir ?: return
        openProfile(dir)
        showSetup()
    }

    private fun openProfile(dir: File) {
        startActivity(Intent(this, PhotoProfileActivity::class.java).putExtra(EXTRA_PHOTOS_DIR, dir.absolutePath))
    }

    // ------------------------------------------------------------------------- photos from the gallery

    /** Says how to choose the photos, then opens the picker. */
    private fun askForPhotos() {
        android.app.AlertDialog.Builder(this).setTitle(R.string.capture_upload_title).setMessage(R.string.capture_upload_steps)
            .setPositiveButton(R.string.capture_upload_choose) { _, _ ->
                photoPicker.launch(androidx.activity.result.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }
            .setNegativeButton(R.string.engine_cancel, null).show()
    }

    /**
     * Makes a capture from photos chosen in the gallery, in any order: each is turned upright, made no
     * bigger than a captured view and kept under the order it was picked in; [UploadSortActivity] then
     * sorts them into views by which way the person faces, to be checked. There is no eye spacing measured
     * at the phone, so measurements use an average one, and the profile screen says so.
     */
    private fun useUploaded(chosen: List<android.net.Uri>) {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(getExternalFilesDir(null), "$PHOTOS_DIR/upload_$stamp").apply { mkdirs() }
        Toast.makeText(this, R.string.capture_upload_working, Toast.LENGTH_SHORT).show()
        analysisExecutor.execute {
            val saved = try {
                var kept = 0
                for (uri in chosen) {
                    val raw = File(dir, "incoming.raw")
                    contentResolver.openInputStream(uri)?.use { input -> raw.outputStream().use { input.copyTo(it) } }
                    val upright = Photos.loadUpright(raw, 1600)
                    raw.delete()
                    if (upright != null) {
                        FileOutputStream(UploadSortActivity.pickFile(dir, kept++)).use { upright.compress(Bitmap.CompressFormat.JPEG, 92, it) }
                    }
                }
                kept > 0
            } catch (e: Exception) {
                Log.e(TAG, "Could not read the chosen photos", e)
                false
            }
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                if (saved) startActivity(Intent(this, UploadSortActivity::class.java).putExtra(UploadSortActivity.EXTRA_DIR, dir.absolutePath)) else {
                    dir.deleteRecursively()
                    Toast.makeText(this, R.string.capture_upload_failed, Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    /** The newest earlier capture that has all its pictures, or null. */
    private fun lastCapture(): File? = File(getExternalFilesDir(null), PHOTOS_DIR).listFiles()
        ?.filter { it.isDirectory && Photos.usable(it) }?.maxByOrNull { it.name }

    // ------------------------------------------------------------------------------ the camera

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val preview = Preview.Builder().build().also { it.surfaceProvider = binding.previewView.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(ResolutionSelector.Builder()
                        .setResolutionStrategy(ResolutionStrategy(FRAME_SIZE, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(analysisExecutor, ::analyze)
                val capture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build()
                // Not unbindAll(): the live try-on's own camera use is left bound, so it comes back when this screen closes.
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis, capture)
                imageCapture = capture
            } catch (e: Exception) {
                Log.e(TAG, "Camera could not be started", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** Runs on [analysisExecutor]: one frame through the pose model and the guide. */
    private fun analyze(imageProxy: ImageProxy) {
        try {
            if (phase == Phase.SETUP && faceIpdMm.isNaN()) measureFace(toUpright(imageProxy))
            if (phase != Phase.GUIDING) return
            if (mode == Mode.FREE) {
                analyzeFree(toUpright(imageProxy))
                return
            }
            val upright = toUpright(imageProxy)
            val result = poseDetector?.detect(upright, SystemClock.uptimeMillis())
            val points = result?.landmarks()?.firstOrNull()
            val world = result?.worldLandmarks()?.firstOrNull()
            val found = points != null && world != null && points.size >= CaptureGuide.POINTS && world.size >= CaptureGuide.POINTS
            if (found) {
                for (i in 0 until CaptureGuide.POINTS) {
                    val o = i * CaptureGuide.STRIDE
                    imagePoints[o] = points!![i].x()
                    imagePoints[o + 1] = points[i].y()
                    imagePoints[o + 2] = points[i].visibility().orElse(0f)
                    worldPoints[o] = world!![i].x()
                    worldPoints[o + 1] = world[i].y()
                    worldPoints[o + 2] = world[i].z()
                }
                if (frameCount++ % LIGHT_EVERY_FRAMES == 0) tooDark = faceLuma(upright) < MIN_FACE_LUMA
            }
            val aspect = upright.width.toFloat() / upright.height
            val guidance = synchronized(guide) {
                guide.update(if (found) imagePoints else null, if (found) worldPoints else null, aspect,
                    SystemClock.uptimeMillis(), tooDark)
            }
            // A view of the turn is this very frame: keep it before the next one overwrites it.
            if (guidance.grab >= 0) saveFrame(upright, guidance.grab)
            runOnUiThread { render(guidance) }
        } catch (e: Exception) {
            Log.e(TAG, "Frame analysis failed", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun saveFrame(frame: Bitmap, view: Int) {
        val dir = sessionDir ?: return
        FileOutputStream(Photos.file(dir, Photos.VIEWS[view])).use { frame.compress(Bitmap.CompressFormat.JPEG, FRAME_JPEG_QUALITY, it) }
    }

    /**
     * The camera's frame turned upright - and not mirrored: see the class comment. Both bitmaps are
     * kept and reused, as the live try-on does, so no memory is handed out per frame.
     */
    private fun toUpright(imageProxy: ImageProxy): Bitmap {
        val plane = imageProxy.planes[0]
        val rowPixels = plane.rowStride / plane.pixelStride
        val source = sourceBitmap?.takeIf { it.width == rowPixels && it.height == imageProxy.height }
            ?: Bitmap.createBitmap(rowPixels, imageProxy.height, Bitmap.Config.ARGB_8888).also { sourceBitmap = it }
        plane.buffer.rewind()
        source.copyPixelsFromBuffer(plane.buffer)

        transform.reset()
        transform.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        bounds.set(0f, 0f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
        transform.mapRect(bounds)
        transform.postTranslate(-bounds.left, -bounds.top)
        val outWidth = bounds.width().roundToInt()
        val outHeight = bounds.height().roundToInt()
        val upright = uprightBitmap?.takeIf { it.width == outWidth && it.height == outHeight }
            ?: Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888).also {
                uprightBitmap = it
                uprightCanvas = Canvas(it)
            }
        uprightCanvas?.drawBitmap(source, transform, blitPaint)
        return upright
    }

    /**
     * While the wearer is at the phone, reads how far apart their eyes are from the width of an iris
     * (see [BodyMeasure]); across the room an iris is too small to measure. Analysis thread.
     */
    private fun measureFace(frame: Bitmap) {
        val points = faceTracker?.detect(frame, SystemClock.uptimeMillis()) ?: return
        fun px(a: Int, b: Int) = hypot((points[a * 3] - points[b * 3]) * frame.width, (points[a * 3 + 1] - points[b * 3 + 1]) * frame.height)
        // Each iris has four points round it; its width is the larger of the two distances across.
        val iris = (maxOf(px(469, 471), px(470, 472)) + maxOf(px(474, 476), px(475, 477))) / 2
        if (iris < MIN_IRIS_PX) return
        val spacing = BodyMeasure.ipdMm(iris, px(IRIS_A, IRIS_B))
        if (spacing !in 50f..78f) return
        faceReadings += spacing
        if (faceReadings.size >= FACE_READINGS) faceIpdMm = faceReadings.sorted()[faceReadings.size / 2]
    }

    /** The mean brightness of the face, 0..255, from a few dozen pixels round the nose. */
    private fun faceLuma(frame: Bitmap): Float {
        val cx = imagePoints[CaptureGuide.NOSE * CaptureGuide.STRIDE] * frame.width
        val cy = imagePoints[CaptureGuide.NOSE * CaptureGuide.STRIDE + 1] * frame.height
        val shoulders = abs(imagePoints[CaptureGuide.LEFT_SHOULDER * CaptureGuide.STRIDE] -
            imagePoints[CaptureGuide.RIGHT_SHOULDER * CaptureGuide.STRIDE]) * frame.width
        val reach = (shoulders * 0.22f).coerceAtLeast(4f)
        var sum = 0f
        var n = 0
        for (j in -3..3) for (i in -3..3) {
            val px = (cx + i * reach / 3).roundToInt()
            val py = (cy + j * reach / 3).roundToInt()
            if (px < 0 || py < 0 || px >= frame.width || py >= frame.height) continue
            val c = frame.getPixel(px, py)
            sum += 0.299f * ((c shr 16) and 0xFF) + 0.587f * ((c shr 8) and 0xFF) + 0.114f * (c and 0xFF)
            n++
        }
        return if (n == 0) 255f else sum / n
    }

    // --------------------------------------------------------------- showing and saying a verdict

    private fun render(g: Guidance) {
        if (phase != Phase.GUIDING) return
        val now = SystemClock.uptimeMillis()
        val text = if (g.prompt == Prompt.COUNTING && g.count == 0) getString(R.string.capture_taking) else words(g.prompt)
        val turning = g.taken != 0
        binding.overlay.show(g.level, text, g.count, g.taken, CaptureGuide.VIEWS, outline = g.taken == 0 && g.level == Level.ADJUST,
            turning = turning)
        binding.debugText.text = if (g.turnDeg.isNaN()) "no body" else
            String.format(Locale.US, "turn %.0f°  fill %.2f  %s", g.turnDeg, g.fill, g.prompt.name.lowercase(Locale.US))

        // Each view of the turn is counted out loud, with a beep a little higher each time and a green flash:
        // a wearer facing away from the phone hears how far round they are, and "stop" at the end.
        if (g.grab >= 0) {
            val had = Integer.bitCount(g.taken)
            binding.overlay.pulse()
            if (g.prompt != Prompt.DONE) {
                beeper.progress(had)
                say(resources.getStringArray(R.array.capture_counts)[had - 1], key = null)
                // The count is the instruction for now: no reminder to keep turning just after it.
                pendingKey = null
                quietUntilMs = now + COUNT_QUIET_MS
            }
        }
        if (g.prompt == Prompt.DONE) {
            beeper.done()
            say(getString(R.string.capture_all_done), key = null)
            showReview()
            return
        }
        if (g.count > 0) {
            if (g.count != lastCount) {
                lastCount = g.count
                say(getString(when (g.count) { 3 -> R.string.capture_count_3; 2 -> R.string.capture_count_2; else -> R.string.capture_count_1 }), key = null)
            }
            return
        }
        lastCount = 0
        if (g.capture) {
            takeFrontPhoto()
            return
        }
        if (g.prompt == Prompt.COUNTING) return

        // The front photo is where the room or the phone's aim can be wrong: say what else to try.
        if (g.taken == 0 && now - helpSaidAtMs > HELP_AFTER_MS && now >= quietUntilMs &&
            (g.prompt == Prompt.NO_PERSON || g.prompt == Prompt.MOVE_AWAY)
        ) {
            helpSaidAtMs = now
            say(getString(R.string.capture_voice_help), key = "help")
            quietUntilMs = now + 5000
            return
        }

        val key = g.prompt.name
        if (key != pendingKey) {
            pendingKey = key
            pendingSinceMs = now
        }
        val settle = if (g.prompt == Prompt.HOLD_STILL) HOLD_STILL_SETTLE_MS else SETTLE_MS
        if (now < quietUntilMs || now - pendingSinceMs < settle || now - spokenAtMs < MIN_GAP_MS) return
        val repeat = if (g.prompt == Prompt.TURN_SLOWLY) TURN_REPEAT_MS else REPEAT_MS
        if (key == spokenKey && now - spokenAtMs < repeat) return
        say(if (g.prompt == Prompt.TURN_SLOWLY) getString(R.string.capture_turning) else text, key)
    }

    private fun say(text: String, key: String?) {
        spokenKey = key
        spokenAtMs = SystemClock.uptimeMillis()
        if (ttsReady) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, key ?: "line")
    }

    private fun words(prompt: Prompt): String = getString(when (prompt) {
        Prompt.NO_PERSON -> R.string.capture_no_person
        Prompt.MOVE_AWAY -> R.string.capture_move_away
        Prompt.COME_CLOSER -> R.string.capture_come_closer
        Prompt.MOVE_LEFT -> R.string.capture_move_left
        Prompt.MOVE_RIGHT -> R.string.capture_move_right
        Prompt.STEP_FORWARD -> R.string.capture_step_forward
        Prompt.STEP_BACKWARD -> R.string.capture_step_backward
        Prompt.FACE_PHONE -> R.string.capture_face_phone
        Prompt.LOOK_AHEAD -> R.string.capture_look_at_phone
        Prompt.STAND_STRAIGHT -> R.string.capture_stand_straight
        Prompt.ARMS_OUT -> R.string.capture_arms_out
        Prompt.ARMS_DOWN -> R.string.capture_arms_down
        Prompt.FEET_APART -> R.string.capture_feet_apart
        Prompt.FEET_CLOSER -> R.string.capture_feet_closer
        Prompt.TOO_DARK -> R.string.capture_too_dark
        Prompt.HOLD_STILL, Prompt.COUNTING -> R.string.capture_hold_still
        Prompt.TURN_SLOWLY -> R.string.capture_turning
        Prompt.SLOWER -> R.string.capture_slower
        Prompt.KEEP_TURNING -> R.string.capture_keep_turning
        Prompt.DONE -> R.string.capture_all_done
    })

    // ------------------------------------------------------------------------------ the front photo

    private fun takeFrontPhoto() {
        val capture = imageCapture
        val dir = sessionDir
        if (capture == null || dir == null) {
            synchronized(guide) { guide.onCaptureFailed() }
            return
        }
        val file = Photos.file(dir, Photos.FRONT)
        capture.takePicture(ImageCapture.OutputFileOptions.Builder(file).build(), ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    if (isDestroyed || phase != Phase.GUIDING) return
                    shutter.play(MediaActionSound.SHUTTER_CLICK)
                    binding.overlay.flash()
                    val now = SystemClock.uptimeMillis()
                    synchronized(guide) { guide.onCaptured(now) }
                    // "Good" and the turn in one breath; then quiet while they start it.
                    say(getString(R.string.capture_good) + " " + getString(R.string.capture_turn_slowly), key = Prompt.TURN_SLOWLY.name)
                    pendingKey = spokenKey
                    pendingSinceMs = now
                    quietUntilMs = now + AFTER_PHOTO_QUIET_MS
                }

                override fun onError(e: ImageCaptureException) {
                    Log.e(TAG, "Front photo failed", e)
                    synchronized(guide) { guide.onCaptureFailed() }
                    if (phase == Phase.GUIDING) say(getString(R.string.capture_failed), key = "failed")
                }
            })
    }

    // ------------------------------------------------------------------ the phone's tilt and volume

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ACCELEROMETER) return
        if (gravityPrimed) {
            for (i in 0..2) gravity[i] += 0.15f * (event.values[i] - gravity[i])
        } else {
            event.values.copyInto(gravity, 0, 0, 3)
            gravityPrimed = true
        }
        val now = SystemClock.uptimeMillis()
        if (phase == Phase.SETUP && now - statusShownAtMs > 300) {
            statusShownAtMs = now
            showSetupStatus()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /** The setup card's three lines: is the phone upright, is the volume up, is the face measured. */
    private fun showSetupStatus() {
        if (!::binding.isInitialized || phase != Phase.SETUP) return
        val good = CaptureOverlayView.GREEN
        val bad = CaptureOverlayView.AMBER
        if (gravityPrimed) {
            // Upright in the hand or on a stand, the pull of gravity runs down the phone's long side.
            val length = hypot(hypot(gravity[0], gravity[1]), gravity[2]).coerceAtLeast(1e-3f)
            val offUpright = Math.toDegrees(acos((gravity[1] / length).coerceIn(-1f, 1f)).toDouble()).toFloat()
            val upright = offUpright <= MAX_TILT_DEG
            binding.tiltStatus.setTextColor(if (upright) good else bad)
            binding.tiltStatus.text = when {
                upright -> getString(R.string.capture_tilt_ok)
                abs(gravity[0]) > abs(gravity[2]) -> getString(R.string.capture_tilt_sideways)
                gravity[2] > 0 -> getString(R.string.capture_tilt_back, offUpright.roundToInt())
                else -> getString(R.string.capture_tilt_forward, offUpright.roundToInt())
            }
        }
        val max = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC).coerceAtLeast(1)
        val loud = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC).toFloat() / max >= MIN_VOLUME
        binding.volumeStatus.setTextColor(if (loud) good else bad)
        binding.volumeStatus.setText(if (loud) R.string.capture_volume_ok else R.string.capture_volume_low)
        val measured = !faceIpdMm.isNaN()
        binding.faceStatus.setTextColor(if (measured) good else bad)
        binding.faceStatus.setText(if (measured) R.string.capture_face_measured else R.string.capture_face_measuring)
    }
}
