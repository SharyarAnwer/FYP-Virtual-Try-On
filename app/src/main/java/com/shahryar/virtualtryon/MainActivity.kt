package com.shahryar.virtualtryon

import android.Manifest
import android.content.pm.PackageManager
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
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import com.shahryar.virtualtryon.databinding.ActivityMainBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * M0 — foundation and performance harness.
 *
 * Brings up the camera preview, stacks a drawing layer over it, and reports frame rate and
 * per-frame latency continuously. No detection yet: M1 adds pose inference inside the analyzer
 * callback, and the numbers on screen will immediately show what it costs.
 */
class MainActivity : AppCompatActivity() {

    private companion object {
        const val TAG = "VTO"
        const val STATS_INTERVAL_MS = 250L

        /**
         * Frame rate floor requested from the camera's auto-exposure controller.
         *
         * Measured on the test device (Infinix Hot 40 Pro, front camera), delivery ranged from
         * ~18 FPS in dim light to ~24.5 FPS in bright light, because auto-exposure lengthens the
         * shutter to gather light and the frame rate follows. Pinning a floor caps exposure time,
         * which keeps the rate steady at the cost of a darker image in dim rooms.
         *
         * The point is measurement integrity: with a moving frame rate, an FPS change in M1 could
         * be the model or could be the weather outside. Raise or lower this to re-tune the
         * brightness/frame-rate trade.
         */
        const val TARGET_MIN_FPS = 24
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var analysisExecutor: ExecutorService

    private val perf = PerfMonitor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var appliedFpsRange: Range<Int>? = null

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

        // Insets go on the chrome, not the root: the preview should run edge to edge.
        ViewCompat.setOnApplyWindowInsetsListener(binding.topBar) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.updatePadding(top = bars.top, left = bars.left, right = bars.right)
            insets
        }

        analysisExecutor = Executors.newSingleThreadExecutor()

        binding.switchButton.setOnClickListener { toggleLens() }
        binding.grantButton.setOnClickListener {
            cameraPermissionRequest.launch(Manifest.permission.CAMERA)
        }

        renderStats(0.0, 0.0, 0, 0)

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

        // Pin the auto-exposure frame-rate range so the delivered rate stops drifting with
        // ambient light. Applied to the preview builder because all use cases in a session
        // share one repeating capture request.
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

        // KEEP_ONLY_LATEST drops stale frames instead of queueing them. Without it, any frame
        // that takes longer than the capture interval builds a backlog and latency grows without
        // bound — the failure this project's 150 ms budget is most likely to hit in M1.
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analysis.setAnalyzer(analysisExecutor) { imageProxy ->
            perf.frameStarted()
            try {
                frameWidth = imageProxy.width
                frameHeight = imageProxy.height

                // M1: pose inference goes here. Everything around it already measures it.

            } finally {
                perf.frameFinished()
                imageProxy.close()
            }
            publishStatsThrottled()
        }

        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

        try {
            provider.unbindAll()
            perf.reset()
            provider.bindToLifecycle(this, selector, preview, analysis)
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    /**
     * Auto-exposure frame-rate ranges the currently selected camera advertises.
     *
     * Requesting a range the camera does not advertise is either ignored or rejected, so the
     * choice has to come from this list rather than from a hard-coded guess.
     */
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
            val list = ranges?.toList().orEmpty()
            Log.i(TAG, "Camera advertises AE FPS ranges: $list")
            list
        } catch (e: Exception) {
            Log.w(TAG, "Could not read AE FPS ranges", e)
            emptyList()
        }
    }

    /**
     * Picks the advertised range that best guarantees [TARGET_MIN_FPS].
     *
     * Among ranges that already clear the floor, the one with the lowest ceiling wins — a wider
     * ceiling invites the camera into a high-speed mode that can cost resolution or extra power
     * without improving the floor, which is the only part being relied on here.
     */
    private fun chooseAeFpsRange(ranges: List<Range<Int>>): Range<Int>? {
        if (ranges.isEmpty()) return null
        ranges.filter { it.lower >= TARGET_MIN_FPS }
            .minByOrNull { it.upper }
            ?.let { return it }
        // Nothing reaches the floor: fall back to the strongest floor on offer.
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

    /** Stats are published a few times a second — per-frame UI updates would cost frames. */
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
        binding.statsText.text = String.format(
            Locale.US,
            "FPS      %8.1f\nLATENCY  %8.1f ms\nFRAME    %8s\nCAMERA   %8s\nAE LOCK  %8s",
            fps, latencyMs, resolution, lens, aeRange
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }
}
