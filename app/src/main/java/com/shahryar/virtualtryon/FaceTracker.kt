package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker

/**
 * Wraps MediaPipe's Face Landmarker - Face Mesh - for the live camera stream: 478 points over
 * the face, each with a depth, for placing glasses (M5).
 *
 * The pose model has a few face points, but they are rough at full-body distance and say little
 * about which way the head is turned. Face Mesh places the eye corners, nose and the outline of
 * the face precisely and in 3D, which a 3D pair of glasses needs to be turned with the head.
 *
 * Same conventions as [PoseDetector]: VIDEO mode, called synchronously so its cost shows in the
 * stats, on the same upright, mirrored bitmap, so its points share the pose landmarks' frame.
 * It is a second model on every frame, so it runs only while glasses are worn.
 */
class FaceTracker private constructor(private val landmarker: FaceLandmarker) {

    companion object {
        private const val TAG = "VTO"
        private const val MODEL_ASSET = "face_landmarker.task"

        /** Points per face: Face Mesh's 468 plus 10 around the irises. */
        const val POINTS = 478

        fun create(context: Context): FaceTracker {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .setDelegate(Delegate.CPU)
                .build()
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setOutputFaceBlendshapes(false)
                .setOutputFacialTransformationMatrixes(false)
                .build()
            return FaceTracker(FaceLandmarker.createFromOptions(context, options))
        }
    }

    private var lastTimestampMs = 0L

    /**
     * Finds the face on an already-upright, already-mirrored bitmap. Returns its points packed as
     * x, y, z per point - x and y normalised to the bitmap, z a depth on x's scale, smaller
     * nearer the camera - or null when no face is found.
     */
    fun detect(bitmap: Bitmap, timestampMs: Long): FloatArray? {
        val stamp = if (timestampMs <= lastTimestampMs) lastTimestampMs + 1 else timestampMs
        lastTimestampMs = stamp
        val face = try {
            landmarker.detectForVideo(BitmapImageBuilder(bitmap).build(), stamp).faceLandmarks().firstOrNull()
        } catch (e: Exception) {
            Log.e(TAG, "Face detection failed", e)
            null
        } ?: return null
        if (face.size < POINTS) return null
        val out = FloatArray(POINTS * 3)
        for (i in 0 until POINTS) {
            out[i * 3] = face[i].x()
            out[i * 3 + 1] = face[i].y()
            out[i * 3 + 2] = face[i].z()
        }
        return out
    }

    fun close() {
        runCatching { landmarker.close() }
    }
}
