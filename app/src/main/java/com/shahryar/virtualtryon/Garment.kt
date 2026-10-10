package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
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

enum class RenderKind {
    /** Generated artwork cut into torso and limb layers, each moved rigidly (M3.5). */
    LAYERS,

    /** One photo per view, bent over a deformable grid (M3.6). */
    MESH
}

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
    /** Layers in back-to-front draw order; a part's parent always comes before it. Empty for
     *  mesh garments. */
    val parts: List<GarmentPart>,
    val colourName: String,
    val colourHex: String,
    val gender: String,
    val fitStyle: String,
    val suitableBodyTypes: List<String>,
    val sizes: List<SizeBand>,
    val isPlaceholder: Boolean,
    val render: RenderKind = RenderKind.LAYERS,
    /** Mesh garments: the front and (for M5.5) back photos, on one shared canvas. */
    val frontPath: String? = null,
    val backPath: String? = null,
    /** Mesh garments: where each sleeve meets the body, keyed "left" and "right". */
    val armholes: Map<String, Armhole> = emptyMap(),
    /**
     * Mesh garments: the collar's inside back, as a polygon on the front photo. A product photo
     * shows it because the shirt was shot on an invisible mannequin; on a person the neck fills
     * that opening, so it is cleared at load time and the wearer's own neck shows through.
     */
    val neckOpening: List<Anchor> = emptyList()
) {
    fun anchor(landmark: String?): Anchor? = landmark?.let { anchors[it] }
}

/**
 * Decoded artwork for one garment. Layered garments fill [fills], [outlines] and [seams],
 * index-aligned with [Garment.parts]; mesh garments fill [meshFront], [meshSleeves] and [mesh].
 */
class GarmentLayers(
    val fills: List<Bitmap>,
    val outlines: List<Bitmap?>,
    val seams: List<Bitmap?>,
    val meshFront: Bitmap? = null,
    val meshSleeves: Bitmap? = null,
    val mesh: GarmentMesh? = null
)

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

    /**
     * Each catalog lives in its own folder with its assets. The generated placeholders and the
     * real garments are written by different tools, so each tool owns one catalog outright and
     * neither can overwrite the other's entries.
     */
    private val CATALOG_DIRS = listOf("garments", "garments/real")
    private val SCHEMA_VERSIONS = 2..3

    private val REQUIRED_ANCHORS = mapOf(
        "upper_body" to listOf("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
        "lower_body" to listOf("left_hip", "right_hip", "left_knee", "right_knee")
    )

    /** Returns only garments that passed validation. Rejects are logged, never thrown. */
    fun load(context: Context): List<Garment> = CATALOG_DIRS.flatMap { loadCatalog(context, it) }

    private fun loadCatalog(context: Context, dir: String): List<Garment> {
        val path = "$dir/catalog.json"
        val raw = try {
            context.assets.open(path).bufferedReader().use { it.readText() }
        } catch (e: Exception) {
            Log.i(TAG, "No catalog at $path")
            return emptyList()
        }

        return try {
            val root = JSONObject(raw)
            val version = root.optInt("schemaVersion", 1)
            if (version !in SCHEMA_VERSIONS) {
                Log.e(TAG, "$path is schema v$version; this build reads v$SCHEMA_VERSIONS")
                return emptyList()
            }
            val array = root.getJSONArray("garments")
            (0 until array.length())
                .mapNotNull { parse(array.getJSONObject(it), dir) }
                .filter { validate(context, it) }
        } catch (e: Exception) {
            Log.e(TAG, "Malformed $path", e)
            emptyList()
        }
    }

    /**
     * Decodes every layer of a garment, fills and outlines. Returns null if any layer fails, since
     * a garment missing a sleeve is worse than no garment. Callers should cache — this allocates.
     */
    fun loadLayers(context: Context, garment: Garment): GarmentLayers? {
        if (garment.render == RenderKind.MESH) {
            val photo = decode(context, garment.frontPath ?: return null) ?: return null
            val front = clearNeckOpening(photo, garment.neckOpening)
            // Weights and the sleeves-only copy are built here, off the UI thread, once.
            val mesh = GarmentMesh(garment)
            return GarmentLayers(emptyList(), emptyList(), emptyList(),
                meshFront = front, meshSleeves = mesh.sleevesOnly(front), mesh = mesh)
        }
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

    /**
     * Clears the collar's inside back from the photo, with an anti-aliased edge so the collar
     * meets the neck softly rather than along a stair-stepped cut. The photo file is untouched:
     * the opening is data in the catalog, adjusted without re-exporting any artwork.
     */
    private fun clearNeckOpening(photo: Bitmap, polygon: List<Anchor>): Bitmap {
        if (polygon.size < 3) return photo
        val out = photo.copy(Bitmap.Config.ARGB_8888, true)
        val path = Path().apply {
            moveTo(polygon[0].x, polygon[0].y)
            for (p in polygon.drop(1)) lineTo(p.x, p.y)
            close()
        }
        val eraser = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
        }
        Canvas(out).drawPath(path, eraser)
        return out
    }

    private fun decode(context: Context, path: String): Bitmap? = try {
        context.assets.open(path).use { BitmapFactory.decodeStream(it) }
    } catch (e: Exception) {
        Log.e(TAG, "Could not decode $path", e)
        null
    }

    private fun point(json: JSONObject): Anchor =
        Anchor(json.getDouble("x").toFloat(), json.getDouble("y").toFloat())

    private fun parse(json: JSONObject, dir: String): Garment? = try {
        val anchorsJson = json.getJSONObject("anchors")
        val anchors = anchorsJson.keys().asSequence().associateWith { key ->
            point(anchorsJson.getJSONObject(key))
        }

        val render = when (val kind = json.optString("render", "layers")) {
            "layers" -> RenderKind.LAYERS
            "mesh" -> RenderKind.MESH
            else -> throw IllegalArgumentException("unknown render kind '$kind'")
        }

        val armholes = json.optJSONObject("armholes")?.let { holes ->
            holes.keys().asSequence().associateWith { side ->
                val h = holes.getJSONObject(side)
                Armhole(point(h.getJSONObject("shoulder")), point(h.getJSONObject("armpit")),
                    point(h.getJSONObject("below")))
            }
        } ?: emptyMap()

        val partsJson = json.optJSONArray("parts") ?: org.json.JSONArray()
        val parts = (0 until partsJson.length()).map { i ->
            val part = partsJson.getJSONObject(i)
            GarmentPart(
                name = part.getString("name"),
                kind = when (val kind = part.getString("kind")) {
                    "torso" -> PartKind.TORSO
                    "segment" -> PartKind.SEGMENT
                    else -> throw IllegalArgumentException("unknown part kind '$kind'")
                },
                assetPath = "$dir/${part.getString("asset")}",
                outlinePath = part.optString("outline").ifEmpty { null }?.let { "$dir/$it" },
                seamPath = part.optString("seam").ifEmpty { null }?.let { "$dir/$it" },
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
            isPlaceholder = json.optBoolean("placeholder", false),
            render = render,
            frontPath = json.optString("front").ifEmpty { null }?.let { "$dir/$it" },
            backPath = json.optString("back").takeUnless { it.isEmpty() || it == "null" }?.let { "$dir/$it" },
            armholes = armholes,
            neckOpening = json.optJSONArray("neckOpening")?.let { pts ->
                (0 until pts.length()).map { point(pts.getJSONObject(it)) }
            } ?: emptyList()
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
            // Elbow and wrist anchors are extrapolated along the sleeve, past the end of a short
            // sleeve, and can legitimately fall beyond the canvas.
            val extrapolated = name.endsWith("_elbow") || name.endsWith("_wrist")
            if (!inside && !extrapolated) {
                reject("anchor '$name' at (${anchor.x}, ${anchor.y}) falls outside " +
                    "${garment.imageWidth}x${garment.imageHeight}")
            }
        }

        if (garment.render == RenderKind.MESH) return ok && validateMesh(context, garment, ::reject)

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

    private fun validateMesh(context: Context, garment: Garment, reject: (String) -> Unit): Boolean {
        var ok = true
        val fail = { reason: String -> reject(reason); ok = false }

        listOf("left_elbow", "right_elbow").forEach { required ->
            if (!garment.anchors.containsKey(required)) fail("mesh garment needs anchor '$required'")
        }
        listOf("left", "right").forEach { side ->
            if (garment.armholes[side] == null) fail("mesh garment needs armhole '$side'")
        }
        val front = garment.frontPath ?: run { fail("mesh garment has no front photo"); return false }
        if (garment.neckOpening.isNotEmpty() && garment.neckOpening.size < 3) {
            fail("neck opening needs at least 3 points, has ${garment.neckOpening.size}")
        }

        for (path in listOfNotNull(front, garment.backPath)) {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            try {
                context.assets.open(path).use { BitmapFactory.decodeStream(it, null, bounds) }
            } catch (e: Exception) {
                fail("photo '$path' could not be opened")
                continue
            }
            if (bounds.outWidth != garment.imageWidth || bounds.outHeight != garment.imageHeight) {
                fail("'$path' is ${bounds.outWidth}x${bounds.outHeight}, but the garment canvas is " +
                    "${garment.imageWidth}x${garment.imageHeight}")
            }
        }
        return ok
    }
}
