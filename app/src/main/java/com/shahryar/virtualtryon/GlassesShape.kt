package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Color
import android.util.Log
import org.json.JSONObject
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * A pair the app offers, from assets/glasses/catalog.json: its [name] on the Glasses button, its
 * model [asset], and how far behind the lenses' front its arms bend down behind the ears.
 */
data class GlassesPair(val id: String, val name: String, val asset: String, val earFromMm: Float)

/**
 * A pair of glasses ready for [Glasses3D] to wear: triangles in millimetres in the glasses' own
 * frame, with a normal and a colour at every corner and, per triangle, the part it belongs to.
 *
 * The frame: x across the face, y down, z back towards the ears. The origin is on the line through
 * the wearer's outer eye corners, centred between them, at the front of the lenses; [Glasses3D]
 * puts it [Glasses3D.LENS_FORWARD_MM] in front of the eye corners.
 *
 * Two kinds: [fromGlb], a real 3D model read by [GlbModel] and worn by default, and [builtIn],
 * plain black glasses built here from square tubes - the first trial, kept as the fallback when a
 * model cannot be read.
 */
class GlassesShape(
    /** Triangle corners, x y z mm, 9 per triangle. */
    val positions: FloatArray,
    /** Unit normal per corner, 9 per triangle. */
    val normals: FloatArray,
    /** ARGB per corner, 3 per triangle. */
    val colours: IntArray,
    /** Per triangle: [PART_FRAME], [PART_ARM] or [PART_EAR]. */
    val parts: ByteArray,
    /** Per triangle: 0 not metal, 1 metal - shaded by reflection rather than by its colour alone. */
    val metallic: FloatArray,
    /** Per triangle: 0 mirror-smooth, sharp highlights, to 1 matt, none. */
    val roughness: FloatArray,
    /**
     * How much the shading is lifted: 1 for a real model, so a surface facing the camera shows
     * exactly its own colour; more for the built-in black pair, whose shape is lost on a face
     * otherwise.
     */
    val brighten: Float,
) {

    val triangles: Int get() = parts.size

    companion object {
        private const val TAG = "VTO"

        // What each triangle is part of, for hiding inside the face's outline (see Glasses3D).
        const val PART_FRAME: Byte = 0
        const val PART_ARM: Byte = 1
        const val PART_EAR: Byte = 2

        /** How far the lens centres sit below the line through the outer eye corners, mm. */
        const val LENS_DROP = 3f

        /** The pairs on offer, written by tools/prepare_glasses.py; see assets/glasses/SOURCES.md. */
        const val CATALOG = "glasses/catalog.json"

        /** A pair to fall back on when the catalog cannot be read: the Khronos sunglasses. */
        const val DEFAULT_MODEL = "glasses/sunglasses_khronos.glb"

        /**
         * How far behind the front of the lenses a model's arms start to bend down behind the
         * ears, mm, when the catalog does not say: past it an arm is the ear end. Measured by eye
         * on the Khronos sunglasses, whose arms run back almost level and bend from about
         * 110-115 mm; tools/prepare_glasses.py measures each pair.
         */
        const val EAR_FROM_MM = 112f

        // A model's parts, told apart by their mesh names.
        private val LENS = Regex("lens", RegexOption.IGNORE_CASE)
        private val ARM = Regex("temple|earhook|arm", RegexOption.IGNORE_CASE)

        /**
         * A lens's inner surface. Drawn as a flat tint it only darkens the lens a second time, so
         * it is left out, and its triangles with it.
         */
        private val LENS_INNER = Regex("lens.*(interior|inner|inside)", RegexOption.IGNORE_CASE)

        // The built-in pair's shape, mm: lens width and height, bridge, how much the front wraps back.
        private const val LENS_W = 52f
        private const val LENS_H = 40f
        private const val LENS_RADIUS = 11f
        private const val BRIDGE = 18f
        private const val WRAP = 0.0022f
        private val FRAME_COLOUR = Color.rgb(30, 30, 34)
        private val ARM_COLOUR = Color.rgb(42, 42, 48)
        private const val BUILT_IN_BRIGHTEN = 1.6f

        /** The pairs listed in [CATALOG], in button order; empty if it cannot be read. */
        fun catalog(context: Context): List<GlassesPair> = try {
            val json = JSONObject(context.assets.open(CATALOG).use { it.readBytes() }.toString(Charsets.UTF_8))
            val list = json.getJSONArray("glasses")
            List(list.length()) { i ->
                val g = list.getJSONObject(i)
                GlassesPair(g.getString("id"), g.getString("name"), "glasses/" + g.getString("file"),
                    if (g.isNull("earFromMm")) EAR_FROM_MM else g.getDouble("earFromMm").toFloat())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Glasses catalog $CATALOG could not be read", e)
            emptyList()
        }

        /** Reads [pair]'s model; the built-in pair if it cannot be read. */
        fun load(context: Context, pair: GlassesPair): GlassesShape = load(context, pair.asset, pair.earFromMm)

        /** Reads the model [asset]; the built-in pair if it cannot be read. */
        fun load(context: Context, asset: String = DEFAULT_MODEL, earFromMm: Float = EAR_FROM_MM): GlassesShape = try {
            fromGlb(GlbModel.load(context, asset), earFromMm).also {
                Log.i(TAG, "Glasses model $asset: ${it.triangles} triangles")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Glasses model $asset could not be read; wearing the built-in pair", e)
            builtIn()
        }

        /**
         * Turns a model in glTF's frame - metres, +Y up, the front facing +Z - into this one. It is
         * placed by its lenses: centred across them, their middle [LENS_DROP] below the eye line,
         * their front at z = 0. Turning y and z round together is a rotation, not a mirror, so left
         * and right stay as the model has them. Arm triangles more than [earFromMm] behind the
         * lenses' front are the ear ends.
         */
        fun fromGlb(model: GlbModel, earFromMm: Float = EAR_FROM_MM): GlassesShape {
            val n = model.triangles
            val p = model.positions
            val keep = BooleanArray(n) { !LENS_INNER.containsMatchIn(model.meshNames[it]) }
            val lens = BooleanArray(n) { keep[it] && LENS.containsMatchIn(model.meshNames[it]) }
            val byLens = lens.any { it }
            var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
            var front = -Float.MAX_VALUE
            for (t in 0 until n) {
                if (!(if (byLens) lens[t] else keep[t])) continue
                for (c in 0 until 3) {
                    val i = t * 9 + c * 3
                    minX = minOf(minX, p[i]); maxX = maxOf(maxX, p[i])
                    minY = minOf(minY, p[i + 1]); maxY = maxOf(maxY, p[i + 1])
                    front = maxOf(front, p[i + 2])
                }
            }
            require(front > -Float.MAX_VALUE) { "model has no triangles" }
            val cx = (minX + maxX) / 2f
            val cy = (minY + maxY) / 2f

            val count = keep.count { it }
            val positions = FloatArray(count * 9)
            val normals = FloatArray(count * 9)
            val colours = IntArray(count * 3)
            val parts = ByteArray(count)
            val metallic = FloatArray(count)
            val roughness = FloatArray(count)
            var k = 0
            for (t in 0 until n) {
                if (!keep[t]) continue
                metallic[k] = model.metallic[t]
                roughness[k] = model.roughness[t]
                var depth = 0f
                for (c in 0 until 3) {
                    val i = t * 9 + c * 3
                    val o = k * 9 + c * 3
                    positions[o] = (p[i] - cx) * 1000f
                    positions[o + 1] = (cy - p[i + 1]) * 1000f + LENS_DROP
                    positions[o + 2] = (front - p[i + 2]) * 1000f
                    normals[o] = model.normals[i]
                    normals[o + 1] = -model.normals[i + 1]
                    normals[o + 2] = -model.normals[i + 2]
                    colours[k * 3 + c] = model.colours[t * 3 + c]
                    depth += positions[o + 2]
                }
                parts[k] = when {
                    !ARM.containsMatchIn(model.meshNames[t]) -> PART_FRAME
                    depth / 3f > earFromMm -> PART_EAR
                    else -> PART_ARM
                }
                k++
            }
            return GlassesShape(positions, normals, colours, parts, metallic, roughness, brighten = 1f)
        }

        /**
         * Plain black glasses built from simple parts - two rims, a bridge, two arms - as square
         * tubes cut into triangles. Flat-shaded: every corner takes its triangle's normal. Matt,
         * not metal, and lifted by [BUILT_IN_BRIGHTEN], so it looks as it did in the first trial.
         */
        fun builtIn(): GlassesShape {
            val tris = ArrayList<Float>()
            val cols = ArrayList<Int>()
            val tags = ArrayList<Byte>()
            val centreY = LENS_DROP
            for (side in intArrayOf(-1, 1)) {
                val cx = side * (BRIDGE / 2f + LENS_W / 2f)
                tube(rim(cx, centreY), closed = true, halfWide = 2.6f, halfDeep = 2f, refZ = true, colour = FRAME_COLOUR, tris, cols, tags, PART_FRAME)
                // End piece from the rim's outer top corner back to the hinge, then the arm.
                val outer = side * (BRIDGE / 2f + LENS_W)
                val hingeY = centreY - LENS_H / 2f + 8f
                tube(listOf(wrapped(outer - side * 1f, hingeY, 0f), wrapped(outer + side * 4f, hingeY, 0f)),
                    closed = false, halfWide = 2.6f, halfDeep = 2f, refZ = true, colour = FRAME_COLOUR, tris, cols, tags, PART_FRAME)
                val hinge = wrapped(outer + side * 4f, hingeY, 0f)
                val arm = ArrayList<FloatArray>()
                for (k in 0..12) {
                    val t = k / 12f
                    val z = hinge[2] + 4f + t * 96f
                    arm += floatArrayOf(hinge[0] + side * 3f * t, hingeY, z)
                }
                // The end curls down behind the ear, and in towards the head as it does.
                val earFrom = arm.size - 1
                for (k in 1..5) {
                    val a = k / 5f * 0.9f
                    arm += floatArrayOf(hinge[0] + side * (3f - 10f * k / 5f), hingeY + 30f * sin(a),
                        hinge[2] + 100f + 24f * sin(a) * cos(a))
                }
                tube(arm, closed = false, halfWide = 2.4f, halfDeep = 1.3f, refZ = false, colour = ARM_COLOUR, tris, cols, tags, PART_ARM, earFrom)
            }
            // Bridge: an arch between the two rims' inner tops.
            val bridge = ArrayList<FloatArray>()
            for (k in 0..8) {
                val x = -BRIDGE / 2f - 1f + (BRIDGE + 2f) * k / 8f
                bridge += wrapped(x, centreY - 9f - 3f * sin(Math.PI.toFloat() * k / 8f), 0f)
            }
            tube(bridge, closed = false, halfWide = 1.8f, halfDeep = 1.8f, refZ = true, colour = FRAME_COLOUR, tris, cols, tags, PART_FRAME)

            val positions = tris.toFloatArray()
            val count = cols.size
            val normals = FloatArray(count * 9)
            for (t in 0 until count) {
                val i = t * 9
                val ux = positions[i + 3] - positions[i]; val uy = positions[i + 4] - positions[i + 1]; val uz = positions[i + 5] - positions[i + 2]
                val vx = positions[i + 6] - positions[i]; val vy = positions[i + 7] - positions[i + 1]; val vz = positions[i + 8] - positions[i + 2]
                var nx = uy * vz - uz * vy
                var ny = uz * vx - ux * vz
                var nz = ux * vy - uy * vx
                val nl = sqrt(nx * nx + ny * ny + nz * nz)
                if (nl > 0f) {
                    nx /= nl; ny /= nl; nz /= nl
                }
                for (c in 0 until 3) {
                    normals[i + c * 3] = nx; normals[i + c * 3 + 1] = ny; normals[i + c * 3 + 2] = nz
                }
            }
            return GlassesShape(positions, normals, IntArray(count * 3) { cols[it / 3] }, tags.toByteArray(),
                metallic = FloatArray(count), roughness = FloatArray(count) { 1f }, brighten = BUILT_IN_BRIGHTEN)
        }

        /** A point on the frame's front, pushed back as the front wraps round towards the ears. */
        private fun wrapped(x: Float, y: Float, z: Float) = floatArrayOf(x, y, z + WRAP * x * x)

        /** One lens rim, a rounded rectangle centred on ([cx], [cy]), as a closed path. */
        private fun rim(cx: Float, cy: Float): List<FloatArray> {
            val out = ArrayList<FloatArray>()
            val hw = LENS_W / 2f - LENS_RADIUS
            val hh = LENS_H / 2f - LENS_RADIUS
            val perCorner = 10
            for (corner in 0 until 4) {
                // Clockwise from the top right: top right, bottom right, bottom left, top left.
                val ox = if (corner < 2) hw else -hw
                val oy = if (corner == 0 || corner == 3) -hh else hh
                for (k in 0 until perCorner) {
                    val a = (corner * 90f + 90f * k / perCorner - 90f) * Math.PI.toFloat() / 180f
                    out += wrapped(cx + ox + LENS_RADIUS * cos(a), cy + oy + LENS_RADIUS * sin(a), 0f)
                }
            }
            return out
        }

        /**
         * Adds a square tube along [path] to [tris]: at each point a rectangle across the path,
         * [halfWide] one way and [halfDeep] the other, joined to the next by four quads. The deep
         * direction is the depth axis for a part in the frame's front ([refZ]), the side-to-side
         * axis for an arm. Each triangle is tagged [part] in [tags], or [PART_EAR] from segment
         * [earFrom] on: the arm's end curling behind the ear.
         */
        private fun tube(
            path: List<FloatArray>, closed: Boolean, halfWide: Float, halfDeep: Float, refZ: Boolean,
            colour: Int, tris: MutableList<Float>, cols: MutableList<Int>, tags: MutableList<Byte>, part: Byte,
            earFrom: Int = Int.MAX_VALUE
        ) {
            val n = path.size
            val rings = ArrayList<Array<FloatArray>>()
            for (i in 0 until n) {
                val prev = path[if (i > 0) i - 1 else if (closed) n - 1 else 0]
                val next = path[if (i < n - 1) i + 1 else if (closed) 0 else n - 1]
                var tx = next[0] - prev[0]
                var ty = next[1] - prev[1]
                var tz = next[2] - prev[2]
                val tl = sqrt(tx * tx + ty * ty + tz * tz)
                tx /= tl; ty /= tl; tz /= tl
                val rx = if (refZ) 0f else 1f
                val rz = if (refZ) 1f else 0f
                // across = t x ref, deep = across x t
                var ax = ty * rz
                var ay = tz * rx - tx * rz
                var az = -ty * rx
                val al = sqrt(ax * ax + ay * ay + az * az)
                ax /= al; ay /= al; az /= al
                val dx = ay * tz - az * ty
                val dy = az * tx - ax * tz
                val dz = ax * ty - ay * tx
                val p = path[i]
                rings += Array(4) { c ->
                    val a = if (c == 0 || c == 3) halfWide else -halfWide
                    val d = if (c < 2) halfDeep else -halfDeep
                    floatArrayOf(p[0] + ax * a + dx * d, p[1] + ay * a + dy * d, p[2] + az * a + dz * d)
                }
            }
            val segments = if (closed) n else n - 1
            for (i in 0 until segments) {
                val r0 = rings[i]
                val r1 = rings[(i + 1) % n]
                for (c in 0 until 4) {
                    quad(r0[c], r0[(c + 1) % 4], r1[(c + 1) % 4], r1[c], colour, tris, cols)
                }
                val tag = if (i >= earFrom) PART_EAR else part
                repeat(8) { tags += tag }
            }
            if (!closed) {
                quad(rings[0][0], rings[0][1], rings[0][2], rings[0][3], colour, tris, cols)
                quad(rings[n - 1][0], rings[n - 1][1], rings[n - 1][2], rings[n - 1][3], colour, tris, cols)
                val endTag = if (n - 2 >= earFrom) PART_EAR else part
                tags += part; tags += part
                tags += endTag; tags += endTag
            }
        }

        private fun quad(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray, colour: Int,
                         tris: MutableList<Float>, cols: MutableList<Int>) {
            for (v in arrayOf(a, b, c, a, c, d)) tris += v.asList()
            cols += colour
            cols += colour
        }
    }
}
