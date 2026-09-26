package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * Wraps MediaPipe's Pose Landmarker for the live camera stream.
 *
 * Two decisions worth knowing:
 *
 * **VIDEO mode, called synchronously.** MediaPipe also offers LIVE_STREAM, which takes a
 * callback and does its own internal frame dropping. That would make the cost of inference
 * invisible to [PerfMonitor], which measures the analyzer callback. Running synchronously means
 * the measured latency is the real inference cost, and CameraX's KEEP_ONLY_LATEST already
 * handles back-pressure — so effective frame rate is simply 1000 / max(frame interval, inference).
 * That is the model the M1 budget was calculated from, and keeping it true keeps the budget real.
 *
 * **Rotation and mirroring are baked into the bitmap before inference**, not corrected afterward.
 * The landmarks therefore come back in the same orientation the user sees on screen, so the
 * overlay only has to account for preview scaling. Correcting orientation after the fact means
 * composing rotation, mirroring, and scaling in the drawing code, which is where these apps
 * usually go wrong.
 */
class PoseDetector private constructor(
    private val landmarker: PoseLandmarker,
    /** Which accelerator actually loaded — may differ from what was asked for. */
    val delegateName: String
) {

    companion object {
        private const val TAG = "VTO"
        private const val MODEL_ASSET = "pose_landmarker_lite.task"

        /**
         * Builds a detector, preferring [preferred] but falling back to CPU if it fails to load.
         *
         * CPU is the default because it measured faster here, not by convention — see the note
         * on MainActivity.requestedDelegate. The Helio G99 has no NPU, so "GPU" means the
         * Mali-G57 MC2, and its OpenCL path does not load on this device.
         *
         * The fallback matters regardless of default: GPU delegate creation fails at load time
         * rather than at inference time, so without it a bad driver is a crash rather than a
         * slower app.
         */
        fun create(context: Context, preferred: Delegate = Delegate.CPU): PoseDetector {
            runCatching { build(context, preferred) }
                .onSuccess { return PoseDetector(it, preferred.name) }
                .onFailure { Log.w(TAG, "Pose landmarker failed on ${preferred.name}, falling back to CPU", it) }

            return PoseDetector(build(context, Delegate.CPU), Delegate.CPU.name)
        }

        private fun build(context: Context, delegate: Delegate): PoseLandmarker {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(delegate)
                .build()

            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            return PoseLandmarker.createFromOptions(context, options)
        }
    }

    private var lastTimestampMs = 0L

    /**
     * Runs detection on an already-upright, already-mirrored bitmap.
     *
     * VIDEO mode requires strictly increasing timestamps and throws otherwise. Two frames can
     * land in the same millisecond on a fast stream, so the timestamp is nudged forward rather
     * than trusted blindly.
     */
    fun detect(bitmap: Bitmap, timestampMs: Long): PoseLandmarkerResult? {
        val stamp = if (timestampMs <= lastTimestampMs) lastTimestampMs + 1 else timestampMs
        lastTimestampMs = stamp

        return try {
            landmarker.detectForVideo(BitmapImageBuilder(bitmap).build(), stamp)
        } catch (e: Exception) {
            Log.e(TAG, "Pose detection failed", e)
            null
        }
    }

    fun close() {
        runCatching { landmarker.close() }
    }
}
