package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.json.JSONObject

/** A point on the garment artwork, in that artwork's own pixel coordinates. */
data class Anchor(val x: Float, val y: Float)

/**
 * One size of a garment, expressed as body *proportions* rather than centimetres.
 *
 * A single camera with no depth sensor cannot recover real-world measurements, so M6 compares
 * the ratio of shoulder width to torso height against these bands instead. This is a declared
 * simplification rather than a hidden one — the honest version of a hard problem.
 *
 * [heightMinCm] and [heightMaxCm] are different: height is typed in by the user on their
 * profile (M7), never measured, so real units are meaningful there.
 */
data class SizeBand(
    val label: String,
    val shoulderToTorsoMin: Float,
    val shoulderToTorsoMax: Float,
    val heightMinCm: Int,
    val heightMaxCm: Int
)

/**
 * A garment and everything the rest of the project needs to know about it.
 *
 * The schema is wider than the overlay alone requires, on purpose. M3 needs only [anchors];
 * M6 reads [sizes]; M7 reads [colourName], [gender], [category] and [suitableBodyTypes].
 * Defining all of it once means asset files are not rewritten twice later.
 */
data class Garment(
    val id: String,
    val name: String,
    val category: String,
    val assetPath: String,
    val imageWidth: Int,
    val imageHeight: Int,
    /** Keyed by MediaPipe pose landmark name, so M3 maps them without a lookup table. */
    val anchors: Map<String, Anchor>,
    val colourName: String,
    val colourHex: String,
    val gender: String,
    val fitStyle: String,
    val suitableBodyTypes: List<String>,
    val sizes: List<SizeBand>,
    val isPlaceholder: Boolean
) {
    fun anchor(landmark: String): Anchor? = anchors[landmark]
}

/**
 * Loads and validates the garment catalog from assets.
 *
 * Validation is the point of this class. An anchor is four numbers in a text file: if it drifts
 * away from the artwork, nothing throws — the garment simply renders in the wrong place, which
 * is indistinguishable from a broken transform in M3. Catching it at load time turns a confusing
 * visual bug into a log line naming the garment and the field.
 */
object GarmentCatalog {

    private const val TAG = "VTO"
    private const val DIR = "garments"
    private const val CATALOG = "$DIR/catalog.json"

    private val REQUIRED_ANCHORS = mapOf(
        "upper_body" to listOf("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
        "lower_body" to listOf("left_hip", "right_hip", "left_knee", "right_knee")
    )

    /** Returns only garments that passed validation. Rejects are logged, never thrown. */
    fun load(context: Context): List<Garment> {
        val raw = try {
            context.assets.open(CATALOG).bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            Log.e(TAG, "Could not read $CATALOG", e)
            return emptyList()
        }

        return try {
            val root = JSONObject(raw)
            val array = root.getJSONArray("garments")
            (0 until array.length())
                .mapNotNull { parse(array.getJSONObject(it)) }
                .filter { validate(context, it) }
        } catch (e: Exception) {
            Log.e(TAG, "Malformed $CATALOG", e)
            emptyList()
        }
    }

    /** Decodes a garment's artwork. Callers should cache — this allocates. */
    fun loadBitmap(context: Context, garment: Garment): Bitmap? = try {
        context.assets.open(garment.assetPath).use { BitmapFactory.decodeStream(it) }
    } catch (e: Exception) {
        Log.e(TAG, "Could not decode ${garment.assetPath}", e)
        null
    }

    private fun parse(json: JSONObject): Garment? = try {
        val anchorsJson = json.getJSONObject("anchors")
        val anchors = anchorsJson.keys().asSequence().associateWith { key ->
            val point = anchorsJson.getJSONObject(key)
            Anchor(point.getDouble("x").toFloat(), point.getDouble("y").toFloat())
        }

        val sizesJson = json.getJSONArray("sizes")
        val sizes = (0 until sizesJson.length()).map { i ->
            val size = sizesJson.getJSONObject(i)
            val ratio = size.getJSONObject("shoulderToTorsoRatio")
            val height = size.getJSONObject("heightRangeCm")
            SizeBand(
                label = size.getString("label"),
                shoulderToTorsoMin = ratio.getDouble("min").toFloat(),
                shoulderToTorsoMax = ratio.getDouble("max").toFloat(),
                heightMinCm = height.getInt("min"),
                heightMaxCm = height.getInt("max")
            )
        }

        val bodyTypesJson = json.getJSONArray("suitableBodyTypes")
        val colour = json.getJSONObject("colour")

        Garment(
            id = json.getString("id"),
            name = json.getString("name"),
            category = json.getString("category"),
            assetPath = "$DIR/${json.getString("asset")}",
            imageWidth = json.getInt("imageWidth"),
            imageHeight = json.getInt("imageHeight"),
            anchors = anchors,
            colourName = colour.getString("name"),
            colourHex = colour.getString("hex"),
            gender = json.getString("gender"),
            fitStyle = json.getString("fitStyle"),
            suitableBodyTypes = (0 until bodyTypesJson.length()).map { bodyTypesJson.getString(it) },
            sizes = sizes,
            isPlaceholder = json.optBoolean("placeholder", false)
        )
    } catch (e: Exception) {
        Log.e(TAG, "Skipping malformed garment entry", e)
        null
    }

    private fun validate(context: Context, garment: Garment): Boolean {
        var ok = true

        fun reject(reason: String) {
            Log.e(TAG, "Garment '${garment.id}' rejected: $reason")
            ok = false
        }

        REQUIRED_ANCHORS[garment.category]?.forEach { required ->
            if (!garment.anchors.containsKey(required)) {
                reject("category '${garment.category}' needs anchor '$required'")
            }
        } ?: Log.w(TAG, "Garment '${garment.id}' has unrecognised category '${garment.category}'")

        garment.anchors.forEach { (name, anchor) ->
            val inside = anchor.x >= 0f && anchor.y >= 0f &&
                anchor.x < garment.imageWidth && anchor.y < garment.imageHeight
            if (!inside) {
                reject("anchor '$name' at (${anchor.x}, ${anchor.y}) falls outside " +
                    "${garment.imageWidth}x${garment.imageHeight}")
            }
        }

        // Decode bounds only — cheap, and it catches artwork edited without updating the
        // catalog, which would otherwise scale every anchor wrongly in M3.
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        try {
            context.assets.open(garment.assetPath).use { BitmapFactory.decodeStream(it, null, bounds) }
        } catch (e: Exception) {
            reject("artwork '${garment.assetPath}' could not be opened")
            return false
        }

        if (bounds.outWidth != garment.imageWidth || bounds.outHeight != garment.imageHeight) {
            reject("catalog says ${garment.imageWidth}x${garment.imageHeight} but artwork is " +
                "${bounds.outWidth}x${bounds.outHeight}")
        }

        return ok
    }
}
