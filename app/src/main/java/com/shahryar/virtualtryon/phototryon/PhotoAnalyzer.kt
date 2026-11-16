package com.shahryar.virtualtryon.phototryon

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.shahryar.virtualtryon.FaceShape
import com.shahryar.virtualtryon.FaceShapeMeter
import com.shahryar.virtualtryon.FaceTracker
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.roundToInt

/** Reading and shrinking the capture's photos. */
object Photos {

    /**
     * A capture's eight views, by the names the engine knows them by: view_0 is the front, and each
     * one after is another 45 degrees to the wearer's left, so view_2 shows their right side and
     * view_4 their back.
     */
    val VIEWS = List(CaptureGuide.VIEWS) { "view_$it" }
    val FRONT = VIEWS[0]
    val SIDE = VIEWS[2]

    fun file(dir: File, view: String) = File(dir, "$view.jpg")

    /** True if [dir] holds all eight views. */
    fun complete(dir: File) = VIEWS.all { file(dir, it).exists() }

    /** True if [dir] has a front view: enough to measure and dress. Photos chosen from the gallery may be fewer than eight. */
    fun usable(dir: File) = file(dir, FRONT).exists()

    /** The views [dir] has photos for. */
    fun present(dir: File) = VIEWS.filter { file(dir, it).exists() }

    /**
     * Which view each of [count] photos chosen from the gallery stands for, taken in the order the wearer
     * turned: spread evenly round the eight, so four are the front, the right side, the back and the left
     * side, and two are the front and the back.
     */
    fun viewsFor(count: Int): List<String> = List(count.coerceIn(0, VIEWS.size)) { k -> VIEWS[(k * VIEWS.size + count / 2) / count] }

    /** [file] decoded the right way up - its EXIF tag obeyed - with its longer side at most [longSide] pixels. */
    fun loadUpright(file: File, longSide: Int): Bitmap? {
        if (!file.exists()) return null
        val size = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, size)
        var sample = 1
        while (maxOf(size.outWidth, size.outHeight) / (sample * 2) >= longSide) sample *= 2
        val decoded = BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample }) ?: return null
        val turn = when (ExifInterface(file.path).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        val scale = (longSide.toFloat() / maxOf(decoded.width, decoded.height)).coerceAtMost(1f)
        if (turn == 0f && scale == 1f) return decoded
        val m = Matrix().apply { postScale(scale, scale); postRotate(turn) }
        return Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, m, true)
    }

    /** [file] as a JPEG small enough to send: upright, longer side at most [longSide]. */
    fun jpeg(file: File, longSide: Int = 1600, quality: Int = 90): ByteArray? {
        val bitmap = loadUpright(file, longSide) ?: return null
        return ByteArrayOutputStream().also { bitmap.compress(Bitmap.CompressFormat.JPEG, quality, it) }.toByteArray()
    }
}

/**
 * What the photos say about the wearer. [scaleKnown] is false when no face could be measured in the
 * front photo: the measurements are then to an assumed height and only their proportions mean
 * anything.
 */
class PhotoAnalysis(val measurements: Measurements?, val skin: SkinReading?, val faceShape: FaceShape?, val scaleKnown: Boolean)

/**
 * Runs the models over a capture's photos and hands the results to [BodyMeasure], [SkinTone] and
 * the live try-on's [FaceShapeMeter] (Photo Try-On, stage 2). Everything here runs on the phone.
 *
 * - The pose model, in its still-image mode with the person mask on, gives the joints and the
 *   silhouette of the front view and of the side view (taken mid-turn, so its depths are rougher
 *   than a posed photo's would be).
 * - The face model runs on the head cut from the front photo at high resolution, where the face is
 *   a few hundred pixels across: it gives the eye spacing in pixels (the scale), the cheeks and
 *   forehead to sample skin from, and the points face shape is judged by.
 *
 * Call [analyse] off the main thread; it takes a few seconds.
 */
class PhotoAnalyzer(private val context: Context) {

    companion object {
        private const val TAG = "VTO-Photo"

        /** The file in a capture's folder holding the wearer's eye spacing, measured at the phone. */
        const val SCALE_FILE = "scale.json"
        const val SCALE_IPD_MM = "ipdMm"

        /** Sizes the photos are read at: the body for the pose model, the head for the face model. */
        private const val BODY_LONG_SIDE = 1280
        private const val HEAD_LONG_SIDE = 3200
        private const val FACE_SIDE = 640

        /** With no face found, the scale is set as if the wearer were this tall, centimetres. */
        private const val ASSUMED_HEIGHT_CM = 170f

        // Face Mesh points: the two iris centres, then where skin is sampled.
        private const val IRIS_A = 468
        private const val IRIS_B = 473
        private val SKIN_POINTS = intArrayOf(50, 280, 205, 425, 151)
        private const val SKIN_REACH = 7
    }

    fun analyse(dir: File): PhotoAnalysis {
        val pose = PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.IMAGE).setNumPoses(1).setOutputSegmentationMasks(true).build())
        val faces = FaceTracker.create(context)
        try {
            val frontFile = Photos.file(dir, Photos.FRONT)
            val frontPhoto = Photos.loadUpright(frontFile, BODY_LONG_SIDE) ?: return PhotoAnalysis(null, null, null, false)
            val front = body(pose, frontPhoto) ?: return PhotoAnalysis(null, null, null, false)
            val side = Photos.loadUpright(Photos.file(dir, Photos.SIDE), BODY_LONG_SIDE)?.let { body(pose, it) }
            val aspect = frontPhoto.width.toFloat() / frontPhoto.height

            // The head, from the photo at high resolution.
            var skin: SkinReading? = null
            var shape: FaceShape? = null
            var mmPerHeight = Float.NaN
            var ownSpacing = false
            val big = Photos.loadUpright(frontFile, HEAD_LONG_SIDE)
            if (big != null) {
                val p = front.first
                val noseX = p[CaptureGuide.NOSE * 3] * big.width
                val noseY = p[CaptureGuide.NOSE * 3 + 1] * big.height
                val shoulders = abs(p[CaptureGuide.LEFT_SHOULDER * 3] - p[CaptureGuide.RIGHT_SHOULDER * 3]) * big.width
                val half = (shoulders * 0.6f).roundToInt().coerceAtLeast(40)
                val left = (noseX - half).roundToInt().coerceIn(0, big.width - 2)
                val top = (noseY - half).roundToInt().coerceIn(0, big.height - 2)
                val side2 = minOf(2 * half, big.width - left, big.height - top)
                val head = Bitmap.createScaledBitmap(Bitmap.createBitmap(big, left, top, side2, side2), FACE_SIDE, FACE_SIDE, true)
                // Twice: the tracker settles on a new picture at the second look.
                faces.detect(head, SystemClock.uptimeMillis())
                val points = faces.detect(head, SystemClock.uptimeMillis())
                if (points != null) {
                    val spacing = hypot(points[IRIS_A * 3] - points[IRIS_B * 3], points[IRIS_A * 3 + 1] - points[IRIS_B * 3 + 1]) * side2
                    val measured = spacingFromFile(dir)
                    ownSpacing = measured != null
                    if (spacing > 1f) mmPerHeight = (measured ?: BodyMeasure.AVERAGE_IPD_MM) / spacing * big.height
                    skin = SkinTone.read(skinPixels(head, points))
                    shape = FaceShapeMeter.proportions(points, FACE_SIDE, FACE_SIDE)?.let { FaceShapeMeter.classify(it) }
                }
            }
            val scaleKnown = !mmPerHeight.isNaN()
            if (!scaleKnown) {
                val bodyShare = (front.second.bottom - front.second.top + 1f) / front.second.height
                mmPerHeight = ASSUMED_HEIGHT_CM * 10f / bodyShare.coerceAtLeast(0.1f)
            }
            val measurements = BodyMeasure.measure(front.second, front.first, side?.second, side?.first, aspect, mmPerHeight, ownSpacing)
            return PhotoAnalysis(measurements, skin, shape, scaleKnown)
        } catch (e: Exception) {
            Log.e(TAG, "Photo analysis failed", e)
            return PhotoAnalysis(null, null, null, false)
        } finally {
            runCatching { pose.close() }
            faces.close()
        }
    }

    /** The pose points (x, y, visibility) and the silhouette of the person in [photo], or null if nobody is found. */
    private fun body(pose: PoseLandmarker, photo: Bitmap): Pair<FloatArray, Silhouette>? {
        val result = pose.detect(BitmapImageBuilder(photo).build())
        val found = result.landmarks().firstOrNull() ?: return null
        if (found.size < CaptureGuide.POINTS) return null
        val points = FloatArray(CaptureGuide.POINTS * 3)
        for (i in 0 until CaptureGuide.POINTS) {
            points[i * 3] = found[i].x()
            points[i * 3 + 1] = found[i].y()
            points[i * 3 + 2] = found[i].visibility().orElse(0f)
        }
        val mask = result.segmentationMasks().orElse(null)?.firstOrNull() ?: return null
        val values = ByteBufferExtractor.extract(mask).duplicate().order(ByteOrder.nativeOrder()).asFloatBuffer()
        val n = mask.width * mask.height
        if (values.remaining() < n) return null
        return points to Silhouette(mask.width, mask.height, BooleanArray(n) { values.get(it) >= 0.5f })
    }

    private fun spacingFromFile(dir: File): Float? = try {
        File(dir, SCALE_FILE).takeIf { it.exists() }?.let { JSONObject(it.readText()).optDouble(SCALE_IPD_MM, Double.NaN) }
            ?.takeIf { !it.isNaN() && it in 50.0..78.0 }?.toFloat()
    } catch (e: Exception) {
        null
    }

    /** Pixels of both cheeks and the forehead: a small square round each of [SKIN_POINTS]. */
    private fun skinPixels(head: Bitmap, points: FloatArray): IntArray {
        val out = ArrayList<Int>()
        for (index in SKIN_POINTS) {
            val cx = (points[index * 3] * head.width).roundToInt()
            val cy = (points[index * 3 + 1] * head.height).roundToInt()
            for (y in cy - SKIN_REACH..cy + SKIN_REACH) for (x in cx - SKIN_REACH..cx + SKIN_REACH) {
                if (x in 0 until head.width && y in 0 until head.height) out += head.getPixel(x, y)
            }
        }
        return out.toIntArray()
    }
}
