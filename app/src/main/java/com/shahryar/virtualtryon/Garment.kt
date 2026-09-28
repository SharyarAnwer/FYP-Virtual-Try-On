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

enum class PartKind {
    /** Follows both shoulders and the hip midpoint with one affine transform (M3). */
    TORSO,

    /** Follows the two landmarks named in [GarmentPart.from] and [GarmentPart.to] (M3.5). */
    SEGMENT
}

/**
 * One layer of a garment.
 *
 * Every layer is a full-size canvas with its part drawn where it sits when the arms hang at
 * rest. That shared frame is what makes the fallback free: a sleeve whose elbow is out of sight
 * can simply be drawn with its [parent]'s transform and still land on the body, rather than
 * vanishing or being guessed at.
 */
data class GarmentPart(
    val name: String,
    val kind: PartKind,
    val assetPath: String,
    /**
     * Outline layer for a limb: its silhouette grown by a few px in the edge colour. It is drawn
     * beneath the torso and every fill, so it only survives where the limb is the garment's outer
     * edge — never as a line across the chest where a sleeve overlaps the torso.
     */
    val outlinePath: String?,
    /**
     * Line layer drawn over every fill — for a long sleeve, the underarm edge from armpit to cuff.
     * At rest that edge lies over the shirt's body, where the outline pass hides it, and without it
     * the sleeve fuses into the body like a batwing.
     */
    val seamPath: String?,
    /** Landmark at the part's pivot — the shoulder for a sleeve, the elbow for a forearm. */
    val from: String?,
    /** Landmark at the far end of the part's axis. */
    val to: String?,
    /** Part whose transform this one borrows when [from] or [to] is not visible. */
    val parent: String?
)

/**
 * A garment and everything the rest of the project needs to know about it.
 *
 * The schema is wider than the overlay alone requires, on purpose. The overlay needs [anchors]
 * and [parts]; M6 reads [sizes]; M7 reads [colourName], [gender], [category] and
 * [suitableBodyTypes]. Defining it all once means asset files are not rewritten later.
 */
data class Garment(
    val id: String,
    val name: String,
    val category: String,
    val imageWidth: Int,
    val imageHeight: Int,
    /** Rest-pose positions keyed by MediaPipe landmark name, shared by every layer. */
    val anchors: Map<String, Anchor>,
    /** Layers in back-to-front draw order; a part's parent always comes before it. */
    val parts: List<GarmentPart>,
    val colourName: String,
    val colourHex: String,
    val gender: String,
    val fitStyle: String,
    val suitableBodyTypes: List<String>,
    val sizes: List<SizeBand>,
    val isPlaceholder: Boolean
) {
    fun anchor(landmark: String?): Anchor? = landmark?.let { anchors[it] }
}

/** Decoded artwork for one garment, index-aligned with [Garment.parts]. */
class GarmentLayers(val fills: List<Bitmap>, val outlines: List<Bitmap?>, val seams: List<Bitmap?>)

/**
 * Loads and validates the garment catalog from assets.
 *
 * Validation is the point of this class. An anchor is two numbers in a text file: if it drifts
 * away from the artwork, nothing throws — the garment simply renders in the wrong place, which
 * looks exactly like a broken transform. Catching it at load time turns a confusing visual bug
 * into a log line naming the garment and the field.
 */
object GarmentCatalog {

    private const val TAG = "VTO"
    private const val DIR = "garments"
    private const val CATALOG = "$DIR/catalog.json"
    private const val SCHEMA_VERSION = 2

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
            val version = root.optInt("schemaVersion", 1)
            if (version != SCHEMA_VERSION) {
                Log.e(TAG, "$CATALOG is schema v$version; this build reads v$SCHEMA_VERSION")
                return emptyList()
            }
            val array = root.getJSONArray("garments")
            (0 until array.length())
                .mapNotNull { parse(array.getJSONObject(it)) }
                .filter { validate(context, it) }
        } catch (e: Exception) {
            Log.e(TAG, "Malformed $CATALOG", e)
            emptyList()
        }
    }

    /**
     * Decodes every layer of a garment, fills and outlines. Returns null if any layer fails, since
     * a garment missing a sleeve is worse than no garment. Callers should cache — this allocates.
     */
    fun loadLayers(context: Context, garment: Garment): GarmentLayers? {
        val fills = ArrayList<Bitmap>(garment.parts.size)
        val outlines = ArrayList<Bitmap?>(garment.parts.size)
        val seams = ArrayList<Bitmap?>(garment.parts.size)
        for (part in garment.parts) {
            fills.add(decode(context, part.assetPath) ?: return null)
            outlines.add(part.outlinePath?.let { decode(context, it) ?: return null })
            seams.add(part.seamPath?.let { decode(context, it) ?: return null })
        }
        return GarmentLayers(fills, outlines, seams)
    }

    private fun decode(context: Context, path: String): Bitmap? = try {
        context.assets.open(path).use { BitmapFactory.decodeStream(it) }
    } catch (e: Exception) {
        Log.e(TAG, "Could not decode $path", e)
        null
    }

    private fun parse(json: JSONObject): Garment? = try {
        val anchorsJson = json.getJSONObject("anchors")
        val anchors = anchorsJson.keys().asSequence().associateWith { key ->
            val point = anchorsJson.getJSONObject(key)
            Anchor(point.getDouble("x").toFloat(), point.getDouble("y").toFloat())
        }

        val partsJson = json.getJSONArray("parts")
        val parts = (0 until partsJson.length()).map { i ->
            val part = partsJson.getJSONObject(i)
            GarmentPart(
                name = part.getString("name"),
                kind = when (val kind = part.getString("kind")) {
                    "torso" -> PartKind.TORSO
                    "segment" -> PartKind.SEGMENT
                    else -> throw IllegalArgumentException("unknown part kind '$kind'")
                },
                assetPath = "$DIR/${part.getString("asset")}",
                outlinePath = part.optString("outline").ifEmpty { null }?.let { "$DIR/$it" },
                seamPath = part.optString("seam").ifEmpty { null }?.let { "$DIR/$it" },
                from = part.optString("from").ifEmpty { null },
                to = part.optString("to").ifEmpty { null },
                parent = part.optString("parent").ifEmpty { null }
            )
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
            imageWidth = json.getInt("imageWidth"),
            imageHeight = json.getInt("imageHeight"),
            anchors = anchors,
            parts = parts,
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
            if (PoseLandmarks.indexOf(name) < 0) {
                reject("anchor '$name' is not a MediaPipe pose landmark")
            }
            val inside = anchor.x >= 0f && anchor.y >= 0f &&
                anchor.x < garment.imageWidth && anchor.y < garment.imageHeight
            if (!inside) {
                reject("anchor '$name' at (${anchor.x}, ${anchor.y}) falls outside " +
                    "${garment.imageWidth}x${garment.imageHeight}")
            }
        }

        // The torso must be drawn first: it is the back layer, and every other part's fallback
        // chain ends at it.
        val torsoCount = garment.parts.count { it.kind == PartKind.TORSO }
        if (torsoCount != 1 || garment.parts.firstOrNull()?.kind != PartKind.TORSO) {
            reject("needs exactly one torso part, listed first")
        }

        val seen = mutableSetOf<String>()
        for (part in garment.parts) {
            if (part.kind == PartKind.SEGMENT) {
                listOf(part.from, part.to).forEach { landmark ->
                    if (garment.anchor(landmark) == null) {
                        reject("part '${part.name}' uses anchor '$landmark', which is not defined")
                    }
                }
                // A parent's transform is computed before its children each frame, so it must
                // appear earlier in draw order or the fallback would read last frame's matrix.
                if (part.parent == null || part.parent !in seen) {
                    reject("part '${part.name}' needs a parent listed before it")
                }
            }
            if (!seen.add(part.name)) reject("part name '${part.name}' is used twice")

            // Decode bounds only — cheap, and it catches artwork edited without updating the
            // catalog, which would otherwise scale every anchor wrongly.
            for (path in listOfNotNull(part.assetPath, part.outlinePath, part.seamPath)) {
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                try {
                    context.assets.open(path).use { BitmapFactory.decodeStream(it, null, bounds) }
                } catch (e: Exception) {
                    reject("artwork '$path' could not be opened")
                    continue
                }
                if (bounds.outWidth != garment.imageWidth || bounds.outHeight != garment.imageHeight) {
                    reject("'$path' is ${bounds.outWidth}x${bounds.outHeight}, but the garment " +
                        "canvas is ${garment.imageWidth}x${garment.imageHeight}")
                }
            }
        }

        return ok
    }
}
