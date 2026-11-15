package com.shahryar.virtualtryon.phototryon

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.shahryar.virtualtryon.FaceTracker
import java.io.File
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.roundToInt

/**
 * Which way the person in one photo faces: [angleDeg] round from facing the camera, growing as they turn
 * to their left - the angle the capture numbers its views by - or null if no body was found. [faceSeen]
 * says whether a face was found, which is what settled front against back.
 */
class PhotoDirection(val angleDeg: Float?, val faceSeen: Boolean)

/**
 * Sorts photos chosen from the gallery into the capture's eight views by which way the person faces in
 * each, so the order they were picked in does not matter (Photo Try-On).
 *
 * Two readings of each photo are combined:
 *
 * - **The shoulders**, from the pose model's 3D points: the line from the right shoulder to the left, seen
 *   from above, turns as the person does - the same measurement the guided capture uses. From behind, the
 *   model can swap left and right, which turns the reading round by half a circle.
 * - **The face**: found in a close crop of the head, it says the person faces the camera; not found, it is
 *   their back. This is the dependable part, and where it disagrees with the shoulders by more than a side
 *   view's width, the shoulders' reading is turned round.
 */
class UploadSorter(private val context: Context) {

    companion object {
        private const val TAG = "VTO-Photo"

        /** The shoulder reading is turned round only when it is this far past side-on on the wrong half. */
        private const val FLIP_MARGIN_DEG = 20f

        /** The head is cropped this many shoulder widths round the nose, for the face model. */
        private const val HEAD_REACH = 0.6f
        private const val FACE_SIDE = 480

        /**
         * The view each photo goes in: each to the nearest free one, nearest pairs first. Photos with no
         * reading take the free views left, in the order they were picked. Never two photos in one view.
         */
        fun assign(angles: List<Float?>): IntArray {
            val views = CaptureGuide.VIEWS
            val out = IntArray(angles.size) { -1 }
            val free = BooleanArray(views) { true }
            val pairs = ArrayList<Triple<Int, Int, Float>>()
            for ((photo, angle) in angles.withIndex()) {
                if (angle == null) continue
                for (view in 0 until views) pairs += Triple(photo, view, abs(CaptureGuide.wrap(angle - view * CaptureGuide.STEP_DEG)))
            }
            for ((photo, view, _) in pairs.sortedBy { it.third }) {
                if (out[photo] < 0 && free[view]) {
                    out[photo] = view
                    free[view] = false
                }
            }
            for (photo in angles.indices) {
                if (out[photo] >= 0) continue
                val view = (0 until views).firstOrNull { free[it] } ?: break
                out[photo] = view
                free[view] = false
            }
            return out
        }

        /** The shoulders' [rawDeg] made to agree with whether a face was seen: front half if so, back half if not. */
        fun settle(rawDeg: Float, faceSeen: Boolean): Float {
            val turned = abs(CaptureGuide.wrap(rawDeg))
            val wrongHalf = if (faceSeen) turned > 90f + FLIP_MARGIN_DEG else turned < 90f - FLIP_MARGIN_DEG
            return if (wrongHalf) CaptureGuide.wrap(rawDeg + 180f) else CaptureGuide.wrap(rawDeg)
        }
    }

    /** The direction of each photo in [files]; call off the main thread. */
    fun directions(files: List<File>): List<PhotoDirection> {
        val pose = PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.IMAGE).setNumPoses(1).build())
        try {
            return files.map { file ->
                try {
                    Photos.loadUpright(file, 1600)?.let { direction(pose, it) } ?: PhotoDirection(null, false)
                } catch (e: Exception) {
                    Log.w(TAG, "No direction for ${file.name}", e)
                    PhotoDirection(null, false)
                }
            }
        } finally {
            runCatching { pose.close() }
        }
    }

    private fun direction(pose: PoseLandmarker, photo: Bitmap): PhotoDirection {
        val result = pose.detect(BitmapImageBuilder(photo).build())
        val points = result.landmarks().firstOrNull()
        val world = result.worldLandmarks().firstOrNull()
        if (points == null || world == null || points.size < CaptureGuide.POINTS) return PhotoDirection(null, false)
        val l = world[CaptureGuide.LEFT_SHOULDER]
        val r = world[CaptureGuide.RIGHT_SHOULDER]
        val raw = Math.toDegrees(atan2(((l.z() - r.z()) * CaptureGuide.TURN_SIGN).toDouble(), (l.x() - r.x()).toDouble())).toFloat()

        // The head, cropped round the nose and enlarged: a whole-body photo's face is too small to find otherwise.
        val nose = points[CaptureGuide.NOSE]
        val shoulders = abs(points[CaptureGuide.LEFT_SHOULDER].x() - points[CaptureGuide.RIGHT_SHOULDER].x()) * photo.width
        val shoulderSpan = maxOf(shoulders, 0.12f * photo.width)
        val half = (shoulderSpan * HEAD_REACH).roundToInt().coerceAtLeast(24)
        val left = (nose.x() * photo.width - half).roundToInt().coerceIn(0, photo.width - 2)
        val top = (nose.y() * photo.height - half).roundToInt().coerceIn(0, photo.height - 2)
        val side = minOf(2 * half, photo.width - left, photo.height - top)
        val head = Bitmap.createScaledBitmap(Bitmap.createBitmap(photo, left, top, side, side), FACE_SIDE, FACE_SIDE, true)
        // A tracker of its own for each photo, so one photo's face is never carried into the next.
        val faces = FaceTracker.create(context)
        val faceSeen = try {
            faces.detect(head, 1L) != null || faces.detect(head, 2L) != null
        } finally {
            faces.close()
        }
        return PhotoDirection(settle(raw, faceSeen), faceSeen)
    }
}
