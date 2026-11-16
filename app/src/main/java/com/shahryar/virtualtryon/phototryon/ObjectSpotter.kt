package com.shahryar.virtualtryon.phototryon

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.io.File

/**
 * Finds things a wearer might be holding - a phone, a bag, a bottle - with Google's small EfficientDet
 * model (assets/efficientdet_lite0.tflite), on the phone. Without the model it finds nothing, and the
 * "holding something" rule simply never fires.
 */
class ObjectSpotter private constructor(private val detector: ObjectDetector?) {

    companion object {
        private const val TAG = "VTO-Photo"
        private const val MODEL = "efficientdet_lite0.tflite"

        fun create(context: Context): ObjectSpotter = try {
            ObjectSpotter(ObjectDetector.createFromOptions(context, ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL).setDelegate(Delegate.CPU).build())
                .setRunningMode(RunningMode.IMAGE).setMaxResults(8).setScoreThreshold(0.35f)
                .setCategoryAllowlist(PoseRules.HOLDABLE.toList()).build()))
        } catch (e: Exception) {
            Log.w(TAG, "Object detector unavailable; the holding rule is off", e)
            ObjectSpotter(null)
        }
    }

    /** The things found in [picture], boxes as shares of its width and height. */
    fun spot(picture: Bitmap): List<Spotted> {
        val d = detector ?: return emptyList()
        return try {
            d.detect(BitmapImageBuilder(picture).build()).detections().mapNotNull { det ->
                val name = det.categories().firstOrNull()?.categoryName() ?: return@mapNotNull null
                val b = det.boundingBox()
                Spotted(name, b.left / picture.width, b.top / picture.height, b.right / picture.width, b.bottom / picture.height)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Object detection failed", e)
            emptyList()
        }
    }

    fun close() {
        runCatching { detector?.close() }
    }
}

/** The pose model's answer as [PoseRules] takes it: the first person's points, in the picture and in metres, and how many people. */
class PoseReading(val image: FloatArray?, val world: FloatArray?, val people: Int)

/** Packs a pose result for [PoseRules] and [FreePoseGuide]. */
fun PoseLandmarkerResult.reading(): PoseReading {
    val points = landmarks().firstOrNull()
    val world = worldLandmarks().firstOrNull()
    if (points == null || world == null || points.size < CaptureGuide.POINTS || world.size < CaptureGuide.POINTS) {
        return PoseReading(null, null, landmarks().size)
    }
    val image = FloatArray(CaptureGuide.POINTS * 3)
    val metres = FloatArray(CaptureGuide.POINTS * 3)
    for (i in 0 until CaptureGuide.POINTS) {
        image[i * 3] = points[i].x()
        image[i * 3 + 1] = points[i].y()
        image[i * 3 + 2] = points[i].visibility().orElse(0f)
        metres[i * 3] = world[i].x()
        metres[i * 3 + 1] = world[i].y()
        metres[i * 3 + 2] = world[i].z()
    }
    return PoseReading(image, metres, landmarks().size)
}

/** A pose model that looks for up to two people, so a second person in the picture can be told apart. */
fun twoPersonPose(context: Context, mode: RunningMode): PoseLandmarker = PoseLandmarker.createFromOptions(context,
    PoseLandmarker.PoseLandmarkerOptions.builder()
        .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").setDelegate(Delegate.CPU).build())
        .setRunningMode(mode).setNumPoses(2)
        .setMinPoseDetectionConfidence(0.5f).setMinPosePresenceConfidence(0.5f).setMinTrackingConfidence(0.5f).build())

/**
 * The same rules for photos chosen from the gallery: each photo's first broken rule, or null. Brightness is
 * not judged here - a photo the person chose themselves may be dim and still fine.
 */
class PhotoChecker(private val context: Context) {
    fun check(files: List<File>): List<PoseVerdict?> {
        val pose = twoPersonPose(context, RunningMode.IMAGE)
        val spotter = ObjectSpotter.create(context)
        try {
            return files.map { file ->
                try {
                    val photo = Photos.loadUpright(file, 1280) ?: return@map PoseVerdict(PoseProblem.NO_PERSON)
                    val reading = pose.detect(BitmapImageBuilder(photo).build()).reading()
                    PoseRules.check(reading.image, reading.world, reading.people, photo.width.toFloat() / photo.height,
                        spotter.spot(photo), tooDark = false)
                } catch (e: Exception) {
                    Log.w("VTO-Photo", "Could not check ${file.name}", e)
                    null
                }
            }
        } finally {
            runCatching { pose.close() }
            spotter.close()
        }
    }
}
