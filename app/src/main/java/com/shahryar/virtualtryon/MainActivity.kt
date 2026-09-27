package com.shahryar.virtualtryon

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.view.View
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import com.google.mediapipe.tasks.core.Delegate
import com.shahryar.virtualtryon.databinding.ActivityMainBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * M1 — pose detection over the live camera stream.
 *
 * The M0 harness is unchanged underneath: FPS and latency are still measured around the whole
 * analyzer callback, so the numbers on screen now include the real cost of inference.
 */
class MainActivity : AppCompatActivity() {

    private companion object {
        const val TAG = "VTO"
        const val STATS_INTERVAL_MS = 250L

        /**
         * Frame rate floor requested from the camera's auto-exposure controller.
         *
         * This device advertises [10,10] [15,15] [5,20] [15,20] [20,20] [5,30] [30,30] — nothing
         * between 20 and 30, so anything in 21..30 selects [30,30] and 20 selects [20,20].
         * [30,30] caps exposure at ~33 ms, which fixed a lighting-dependent 18-24.5 FPS swing on
         * the front camera at the cost of a darker image. If detection struggles in dim rooms,
         * dropping to 20 buys a 50 ms exposure but leaves no frame-rate margin at all.
         */
        const val TARGET_MIN_FPS = 24
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var analysisExecutor: ExecutorService

    private val perf = PerfMonitor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var appliedFpsRange: Range<Int>? = null

    /** Touched only on [analysisExecutor], which is single-threaded, so access is serialised. */
    private var poseDetector: PoseDetector? = null
    private var sourceBitmap: Bitmap? = null
    private var uprightBitmap: Bitmap? = null
    private var uprightCanvas: Canvas? = null
    private val transform = Matrix()
    private val transformedBounds = RectF()
    private val blitPaint = Paint().apply {
        isFilterBitmap = false
        isAntiAlias = false
    }
    /**
     * Measured on this device rather than assumed. On the Infinix Hot 40 Pro (Helio G99,
     * Mali-G57 MC2) the CPU delegate runs inference in 39.9 ms against the GPU's 46.5 ms —
     * the GPU is 14% slower, not faster. The MediaPipe log explains why: OpenCL fails to load
     * and the delegate falls back to an ICD loader path that evidently accelerates nothing.
     *
     * The on-screen toggle still switches at runtime, so the comparison stays reproducible.
     */
    private var requestedDelegate = Delegate.CPU

    private var garments: List<Garment> = emptyList()
    private val garmentBitmaps = mutableMapOf<String, Bitmap>()

    /** Index into [garments]; -1 shows no garment. */
    private var selectedGarment = -1

    @Volatile private var garmentCount = 0
    @Volatile private var activeDelegate = "loading"
    @Volatile private var poseTracked = false
    @Volatile private var frameWidth = 0
    @Volatile private var frameHeight = 0
    private var lastStatsAtMs = 0L

    private val cameraPermissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            binding.permissionPanel.visibility = View.GONE
            startCamera()
        } else {
            binding.permissionPanel.visibility = View.VISIBLE
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // OverlayView reproduces this scale type when mapping landmarks. Set explicitly rather
        // than relying on the default, because the two have to agree.
        binding.previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        ViewCompat.setOnApplyWindowInsetsListener(binding.topBar) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.updatePadding(top = bars.top, left = bars.left, right = bars.right)
            insets
        }

        analysisExecutor = Executors.newSingleThreadExecutor()

        binding.switchButton.setOnClickListener { toggleLens() }
        binding.delegateButton.setOnClickListener { toggleDelegate() }
        binding.garmentButton.setOnClickListener { cycleGarment() }
        binding.grantButton.setOnClickListener {
            cameraPermissionRequest.launch(Manifest.permission.CAMERA)
        }

        updateDelegateButton()
        updateGarmentButton()
        renderStats(0.0, 0.0, 0, 0)
        loadGarments()
        loadDetector()

        if (hasCameraPermission()) {
            startCamera()
        } else {
            binding.permissionPanel.visibility = View.VISIBLE
            cameraPermissionRequest.launch(Manifest.permission.CAMERA)
        }
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    /**
     * Reads the garment catalog off the analysis thread, since it touches asset files.
     *
     * Rejected garments are logged rather than thrown, so one bad entry cannot stop the app
     * from starting — but the count on screen makes a silent rejection visible immediately.
     */
    private fun loadGarments() {
        analysisExecutor.execute {
            val loaded = GarmentCatalog.load(this)
            garments = loaded
            garmentCount = loaded.size
            Log.i(TAG, "Garment catalog: ${loaded.size} loaded")
            runOnUiThread { updateGarmentButton() }
            loaded.forEach { g ->
                val flag = if (g.isPlaceholder) " [placeholder]" else ""
                Log.i(TAG, "  ${g.id}: ${g.category}, ${g.colourName}, ${g.gender}, " +
                    "${g.anchors.size} anchors, ${g.sizes.size} sizes$flag")
            }
        }
    }

    /**
     * Builds the detector on the analysis thread.
     *
     * Model loading takes long enough to stutter the UI, and running it on the same
     * single-threaded executor as inference means it can never overlap a detect() call.
     */
    private fun loadDetector() {
        analysisExecutor.execute {
            poseDetector?.close()
            val detector = PoseDetector.create(this, requestedDelegate)
            poseDetector = detector
            activeDelegate = detector.delegateName
            perf.reset()
            Log.i(TAG, "Pose detector ready on ${detector.delegateName}")
            runOnUiThread { updateDelegateButton() }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                cameraProvider = future.get()
                bindUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Camera provider unavailable", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun bindUseCases() {
        val provider = cameraProvider ?: return

        val previewBuilder = Preview.Builder()

        appliedFpsRange = chooseAeFpsRange(supportedAeFpsRanges())
        appliedFpsRange?.let { range ->
            Camera2Interop.Extender(previewBuilder).setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                range
            )
            Log.i(TAG, "Requested AE target FPS range: $range")
        }

        val preview = previewBuilder.build().also {
            it.surfaceProvider = binding.previewView.surfaceProvider
        }

        // RGBA_8888 so frames can be copied straight into a Bitmap. The default YUV_420_888
        // would need a colour-space conversion on every frame before MediaPipe could use it.
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        analysis.setAnalyzer(analysisExecutor, ::analyze)

        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

        try {
            provider.unbindAll()
            perf.reset()
            provider.bindToLifecycle(this, selector, preview, analysis)
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    /** Runs on [analysisExecutor]. */
    private fun analyze(imageProxy: ImageProxy) {
        perf.frameStarted()
        try {
            // Staged timing: preparing the bitmap and running the model have entirely
            // different fixes, so one combined number would say a frame is slow without
            // saying which half to attack.
            val startNs = System.nanoTime()
            val upright = toUprightBitmap(imageProxy)
            val preparedNs = System.nanoTime()

            frameWidth = upright.width
            frameHeight = upright.height

            val landmarks = poseDetector
                ?.detect(upright, SystemClock.uptimeMillis())
                ?.landmarks()
                ?.firstOrNull()

            perf.recordPrep((preparedNs - startNs) / 1_000_000.0)
            perf.recordInference((System.nanoTime() - preparedNs) / 1_000_000.0)

            poseTracked = !landmarks.isNullOrEmpty()
            binding.overlayView.setPose(landmarks, upright.width, upright.height)
        } catch (e: Exception) {
            Log.e(TAG, "Frame analysis failed", e)
        } finally {
            perf.frameFinished()
            imageProxy.close()
        }
        publishStatsThrottled()
    }

    /**
     * Converts a camera frame into the orientation the user actually sees.
     *
     * Rotation and front-camera mirroring are applied here rather than corrected later, so the
     * landmarks MediaPipe returns are already in on-screen orientation and OverlayView only has
     * to undo the preview's scaling.
     *
     * Both bitmaps are allocated once and reused. The obvious version of this — letting
     * Bitmap.createBitmap produce the rotated copy — allocates about 1.2 MB per frame, roughly
     * 36 MB/s of garbage at 30 FPS, and the resulting collection pauses land directly on the
     * frame rate this module is judged by.
     *
     * The destination rectangle is derived by mapping the source bounds through the transform
     * and translating the result back to the origin, which is what createBitmap does internally.
     * Deriving it beats hand-writing offsets for four rotations times two mirror states.
     */
    private fun toUprightBitmap(imageProxy: ImageProxy): Bitmap {
        val source = sourceBitmap?.takeIf {
            it.width == imageProxy.width && it.height == imageProxy.height
        } ?: Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
            .also { sourceBitmap = it }

        imageProxy.planes[0].buffer.rewind()
        source.copyPixelsFromBuffer(imageProxy.planes[0].buffer)

        transform.reset()
        transform.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        if (lensFacing == CameraSelector.LENS_FACING_FRONT) transform.postScale(-1f, 1f)

        transformedBounds.set(0f, 0f, source.width.toFloat(), source.height.toFloat())
        transform.mapRect(transformedBounds)
        transform.postTranslate(-transformedBounds.left, -transformedBounds.top)

        val outWidth = Math.round(transformedBounds.width())
        val outHeight = Math.round(transformedBounds.height())

        val destination = uprightBitmap?.takeIf {
            it.width == outWidth && it.height == outHeight
        } ?: Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888).also {
            uprightBitmap = it
            uprightCanvas = Canvas(it)
        }

        // The drawn image covers the destination exactly, so there is nothing to clear first.
        // Every rotation is a multiple of 90 degrees, so sampling is pixel-aligned and filtering
        // would cost time without changing a single pixel.
        uprightCanvas?.drawBitmap(source, transform, blitPaint)
        return destination
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun supportedAeFpsRanges(): List<Range<Int>> {
        val provider = cameraProvider ?: return emptyList()
        return try {
            val info: CameraInfo? = provider.availableCameraInfos.firstOrNull { candidate ->
                Camera2CameraInfo.from(candidate)
                    .getCameraCharacteristic(CameraCharacteristics.LENS_FACING) == lensFacing
            }
            val ranges = info?.let {
                Camera2CameraInfo.from(it)
                    .getCameraCharacteristic(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
            }
            ranges?.toList().orEmpty().also { Log.i(TAG, "Camera advertises AE FPS ranges: $it") }
        } catch (e: Exception) {
            Log.w(TAG, "Could not read AE FPS ranges", e)
            emptyList()
        }
    }

    private fun chooseAeFpsRange(ranges: List<Range<Int>>): Range<Int>? {
        if (ranges.isEmpty()) return null
        ranges.filter { it.lower >= TARGET_MIN_FPS }
            .minByOrNull { it.upper }
            ?.let { return it }
        return ranges.maxByOrNull { it.lower * 1000 + it.upper }
    }

    private fun toggleLens() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            CameraSelector.LENS_FACING_BACK
        } else {
            CameraSelector.LENS_FACING_FRONT
        }
        bindUseCases()
    }

    /**
     * Steps through the catalog: none, then each garment, then back to none.
     *
     * Artwork is decoded on the analysis thread and cached by id. Decoding a 512x640 PNG takes
     * long enough to drop a frame if done on the UI thread, and the cache means switching back
     * to a garment costs nothing the second time.
     */
    private fun cycleGarment() {
        if (garments.isEmpty()) return
        selectedGarment = if (selectedGarment + 1 >= garments.size) -1 else selectedGarment + 1
        updateGarmentButton()

        val chosen = garments.getOrNull(selectedGarment)
        if (chosen == null) {
            binding.overlayView.setGarment(null, null)
            return
        }

        analysisExecutor.execute {
            val bitmap = garmentBitmaps.getOrPut(chosen.id) {
                GarmentCatalog.loadBitmap(this, chosen) ?: return@execute
            }
            runOnUiThread { binding.overlayView.setGarment(bitmap, chosen) }
        }
    }

    private fun updateGarmentButton() {
        val chosen = garments.getOrNull(selectedGarment)
        binding.garmentButton.text = chosen?.colourName?.replaceFirstChar { it.uppercase() }
            ?: getString(R.string.garment_none)
        binding.garmentButton.isEnabled = garments.isNotEmpty()
    }

    /**
     * Swaps the accelerator at runtime so GPU and CPU cost can be compared on the spot — that
     * comparison is a result for the evaluation chapter, not just a tuning knob.
     */
    private fun toggleDelegate() {
        requestedDelegate =
            if (requestedDelegate == Delegate.GPU) Delegate.CPU else Delegate.GPU
        activeDelegate = "loading"
        updateDelegateButton()
        loadDetector()
    }

    private fun updateDelegateButton() {
        binding.delegateButton.text = requestedDelegate.name
    }

    private fun publishStatsThrottled() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastStatsAtMs < STATS_INTERVAL_MS) return
        lastStatsAtMs = now

        val fps = perf.fps
        val latency = perf.latencyMs
        val w = frameWidth
        val h = frameHeight
        runOnUiThread { renderStats(fps, latency, w, h) }
    }

    private fun renderStats(fps: Double, latencyMs: Double, w: Int, h: Int) {
        val lens = if (lensFacing == CameraSelector.LENS_FACING_FRONT) "FRONT" else "BACK"
        val resolution = if (w > 0 && h > 0) "${w}x$h" else "--"
        val aeRange = appliedFpsRange?.let { "${it.lower}-${it.upper}" } ?: "default"
        val tracking = if (poseTracked) "TRACKED" else "none"
        binding.statsText.text = String.format(
            Locale.US,
            "FPS      %8.1f\nLATENCY  %8.1f ms\n  prep   %8.1f ms\n  infer  %8.1f ms\n" +
                "FRAME    %8s\nCAMERA   %8s\nAE LOCK  %8s\nMODEL    %8s\n" +
                "DELEGATE %8s\nPOSE     %8s\nGARMENTS %8d",
            fps, latencyMs, perf.prepMs, perf.inferenceMs,
            resolution, lens, aeRange, "lite", activeDelegate, tracking, garmentCount
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.execute { poseDetector?.close() }
        analysisExecutor.shutdown()
    }
}
