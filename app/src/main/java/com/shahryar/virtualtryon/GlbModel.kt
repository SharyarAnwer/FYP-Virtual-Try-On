package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Color
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.floor
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Reads a glTF 2.0 binary file (.glb) into plain triangles: what [GlassesShape] needs to draw a
 * real 3D model of a pair of glasses (M5), with no 3D library.
 *
 * Every mesh in the default scene is placed by its nodes' transforms and unrolled into separate
 * triangles - three corners each, none shared - in the file's own frame: metres, +Y up, the
 * front facing +Z. Each corner gets its normal and a colour: the material's base colour, times
 * its texture at that corner if it has one, times the corner's own colour (COLOR_0) if the file
 * stores one - as tools/prepare_glasses.py does - in sRGB. Each triangle also keeps its material's
 * metallic and roughness factors - whether it is metal, and how polished - so it can be shaded
 * as the file describes it; textures for those two are not read.
 *
 * Materials that let light through (KHR_materials_transmission, or alpha blending) get an alpha.
 * Drawing flat colours cannot bend light through glass, so a see-through part is drawn partly
 * transparent instead: a dark tint like a sunglass lens mostly hides what is behind it, a clear
 * part like a nose pad barely does.
 *
 * Only what simple models use is read: triangle lists, float positions and normals, float or
 * normalised texture coordinates, node matrices or translation-rotation-scale, and embedded PNG
 * or JPEG textures. Normals are turned by each node's 3x3 and renormalised, which is right for
 * rotations and uniform scales.
 */
class GlbModel private constructor(
    /** Triangle corners, x y z, 9 per triangle. */
    val positions: FloatArray,
    /** Unit normal per corner, 9 per triangle. */
    val normals: FloatArray,
    /** ARGB per corner, 3 per triangle. */
    val colours: IntArray,
    /** Per triangle, the name of the mesh it came from. */
    val meshNames: Array<String>,
    /** Per triangle, its material's metallic factor: 0 not metal, 1 metal. */
    val metallic: FloatArray,
    /** Per triangle, its material's roughness factor: 0 mirror-smooth, 1 matt. */
    val roughness: FloatArray,
) {

    val triangles: Int get() = meshNames.size

    companion object {
        private const val MAGIC_GLTF = 0x46546C67 // "glTF"
        private const val CHUNK_JSON = 0x4E4F534A // "JSON"
        private const val CHUNK_BIN = 0x004E4942 // "BIN\0"
        private const val TRIANGLES = 4

        /** Alpha of a fully see-through material: dark tints, then clear ones. */
        private const val SEE_THROUGH_DARK = 0.7f
        private const val SEE_THROUGH_CLEAR = 0.25f

        fun load(context: Context, assetPath: String): GlbModel =
            parse(context.assets.open(assetPath).use { it.readBytes() })

        fun parse(bytes: ByteArray): GlbModel {
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            require(bytes.size >= 20 && buf.getInt(0) == MAGIC_GLTF) { "not a GLB file" }
            require(buf.getInt(4) == 2) { "not glTF 2.0" }
            var json: JSONObject? = null
            var binStart = -1
            var offset = 12
            while (offset + 8 <= bytes.size) {
                val length = buf.getInt(offset)
                when (buf.getInt(offset + 4)) {
                    CHUNK_JSON -> json = JSONObject(String(bytes, offset + 8, length, Charsets.UTF_8))
                    CHUNK_BIN -> binStart = offset + 8
                }
                offset += 8 + length
            }
            requireNotNull(json) { "GLB has no JSON chunk" }
            require(binStart >= 0) { "GLB has no binary chunk" }
            return Reader(json, bytes, buf, binStart).read()
        }

        private fun linear(c: Float): Float =
            if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).toDouble().pow(2.4).toFloat()

        private fun srgb(c: Float): Float {
            val v = c.coerceIn(0f, 1f)
            return if (v <= 0.0031308f) v * 12.92f else (1.055 * v.toDouble().pow(1 / 2.4) - 0.055).toFloat()
        }

        /** Column-major 4x4 product [a] x [b]. */
        private fun multiply(a: FloatArray, b: FloatArray): FloatArray {
            val out = FloatArray(16)
            for (c in 0 until 4) for (r in 0 until 4) {
                var s = 0f
                for (k in 0 until 4) s += a[k * 4 + r] * b[c * 4 + k]
                out[c * 4 + r] = s
            }
            return out
        }

        private val IDENTITY = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)
    }

    private class Material(
        /** Base colour factor, linear RGBA. */
        val factor: FloatArray,
        /** Base colour texture as ARGB pixels, or null. */
        val pixels: IntArray?,
        val width: Int,
        val height: Int,
        val texCoord: Int,
        /** 0 opaque to 1 fully see-through. */
        val transmission: Float,
        val blend: Boolean,
        val metallic: Float,
        val roughness: Float,
    )

    private class Reader(val gltf: JSONObject, val bytes: ByteArray, val buf: ByteBuffer, val binStart: Int) {
        val accessors: JSONArray = gltf.getJSONArray("accessors")
        val views: JSONArray = gltf.getJSONArray("bufferViews")
        val materials: List<Material> = gltf.optJSONArray("materials")?.let { list ->
            List(list.length()) { material(list.getJSONObject(it)) }
        } ?: emptyList()
        /** glTF's default material: white, metal, matt. */
        val defaultMaterial = Material(floatArrayOf(1f, 1f, 1f, 1f), null, 0, 0, 0, 0f, false, 1f, 1f)

        val positions = ArrayList<Float>()
        val normals = ArrayList<Float>()
        val colours = ArrayList<Int>()
        val names = ArrayList<String>()
        val metallic = ArrayList<Float>()
        val roughness = ArrayList<Float>()

        fun read(): GlbModel {
            val scenes = gltf.optJSONArray("scenes")
            val roots: List<Int> = if (scenes != null && scenes.length() > 0) {
                val scene = scenes.getJSONObject(gltf.optInt("scene", 0))
                val nodes = scene.optJSONArray("nodes") ?: JSONArray()
                List(nodes.length()) { nodes.getInt(it) }
            } else {
                List(gltf.optJSONArray("nodes")?.length() ?: 0) { it }
            }
            for (root in roots) walk(root, IDENTITY)
            return GlbModel(positions.toFloatArray(), normals.toFloatArray(), colours.toIntArray(), names.toTypedArray(),
                metallic.toFloatArray(), roughness.toFloatArray())
        }

        private fun walk(index: Int, parent: FloatArray) {
            val node = gltf.getJSONArray("nodes").getJSONObject(index)
            val world = multiply(parent, local(node))
            if (node.has("mesh")) addMesh(gltf.getJSONArray("meshes").getJSONObject(node.getInt("mesh")), world)
            node.optJSONArray("children")?.let { children ->
                for (i in 0 until children.length()) walk(children.getInt(i), world)
            }
        }

        private fun local(node: JSONObject): FloatArray {
            node.optJSONArray("matrix")?.let { m -> return FloatArray(16) { m.getDouble(it).toFloat() } }
            val t = node.optJSONArray("translation")
            val q = node.optJSONArray("rotation")
            val s = node.optJSONArray("scale")
            fun at(a: JSONArray?, i: Int, default: Float) = a?.getDouble(i)?.toFloat() ?: default
            val qx = at(q, 0, 0f); val qy = at(q, 1, 0f); val qz = at(q, 2, 0f); val qw = at(q, 3, 1f)
            val sx = at(s, 0, 1f); val sy = at(s, 1, 1f); val sz = at(s, 2, 1f)
            return floatArrayOf(
                (1 - 2 * (qy * qy + qz * qz)) * sx, 2 * (qx * qy + qz * qw) * sx, 2 * (qx * qz - qy * qw) * sx, 0f,
                2 * (qx * qy - qz * qw) * sy, (1 - 2 * (qx * qx + qz * qz)) * sy, 2 * (qy * qz + qx * qw) * sy, 0f,
                2 * (qx * qz + qy * qw) * sz, 2 * (qy * qz - qx * qw) * sz, (1 - 2 * (qx * qx + qy * qy)) * sz, 0f,
                at(t, 0, 0f), at(t, 1, 0f), at(t, 2, 0f), 1f,
            )
        }

        private fun addMesh(mesh: JSONObject, m: FloatArray) {
            val name = mesh.optString("name", "")
            val primitives = mesh.getJSONArray("primitives")
            for (p in 0 until primitives.length()) {
                val prim = primitives.getJSONObject(p)
                if (prim.optInt("mode", TRIANGLES) != TRIANGLES) continue
                val attributes = prim.getJSONObject("attributes")
                val pos = floats(attributes.getInt("POSITION"))
                val nrm = if (attributes.has("NORMAL")) floats(attributes.getInt("NORMAL")) else null
                val material = if (prim.has("material")) materials[prim.getInt("material")] else defaultMaterial
                val uvKey = "TEXCOORD_${material.texCoord}"
                val uv = if (material.pixels != null && attributes.has(uvKey)) floats(attributes.getInt(uvKey)) else null
                val tint = if (attributes.has("COLOR_0")) floats(attributes.getInt("COLOR_0")) else null
                val tintSize = if (attributes.has("COLOR_0")) components(accessors.getJSONObject(attributes.getInt("COLOR_0")).getString("type")) else 0
                val corners = pos.size / 3
                val index = if (prim.has("indices")) ints(prim.getInt("indices")) else IntArray(corners) { it }
                for (t in 0 until index.size / 3) {
                    val start = positions.size
                    for (c in 0 until 3) {
                        val i = index[t * 3 + c]
                        val x = pos[i * 3]; val y = pos[i * 3 + 1]; val z = pos[i * 3 + 2]
                        positions += m[0] * x + m[4] * y + m[8] * z + m[12]
                        positions += m[1] * x + m[5] * y + m[9] * z + m[13]
                        positions += m[2] * x + m[6] * y + m[10] * z + m[14]
                        colours += colour(material, uv, tint, tintSize, i)
                    }
                    if (nrm != null) {
                        for (c in 0 until 3) {
                            val i = index[t * 3 + c]
                            val x = nrm[i * 3]; val y = nrm[i * 3 + 1]; val z = nrm[i * 3 + 2]
                            addUnit(m[0] * x + m[4] * y + m[8] * z, m[1] * x + m[5] * y + m[9] * z, m[2] * x + m[6] * y + m[10] * z)
                        }
                    } else {
                        // No normals: the triangle's own, the same at all three corners.
                        val ax = positions[start]; val ay = positions[start + 1]; val az = positions[start + 2]
                        val ux = positions[start + 3] - ax; val uy = positions[start + 4] - ay; val uz = positions[start + 5] - az
                        val vx = positions[start + 6] - ax; val vy = positions[start + 7] - ay; val vz = positions[start + 8] - az
                        repeat(3) { addUnit(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) }
                    }
                    names += name
                    metallic += material.metallic
                    roughness += material.roughness
                }
            }
        }

        private fun addUnit(x: Float, y: Float, z: Float) {
            val l = sqrt(x * x + y * y + z * z)
            if (l > 0f) {
                normals += x / l; normals += y / l; normals += z / l
            } else {
                normals += 0f; normals += 0f; normals += 1f
            }
        }

        /**
         * The colour of corner [i]: base factor times texture times the corner's own colour
         * ([tint], COLOR_0, [tintSize] values per corner), in sRGB, with the material's alpha.
         */
        private fun colour(material: Material, uv: FloatArray?, tint: FloatArray?, tintSize: Int, i: Int): Int {
            var r = material.factor[0]
            var g = material.factor[1]
            var b = material.factor[2]
            var tintAlpha = 1f
            if (tint != null) {
                r *= tint[i * tintSize]
                g *= tint[i * tintSize + 1]
                b *= tint[i * tintSize + 2]
                if (tintSize == 4) tintAlpha = tint[i * tintSize + 3]
            }
            val pixels = material.pixels
            if (pixels != null && uv != null) {
                val u = uv[i * 2] - floor(uv[i * 2])
                val v = uv[i * 2 + 1] - floor(uv[i * 2 + 1])
                val px = pixels[(v * (material.height - 1)).roundToInt() * material.width + (u * (material.width - 1)).roundToInt()]
                r *= linear(Color.red(px) / 255f)
                g *= linear(Color.green(px) / 255f)
                b *= linear(Color.blue(px) / 255f)
            }
            val sr = srgb(r); val sg = srgb(g); val sb = srgb(b)
            var alpha = if (material.blend) material.factor[3] * tintAlpha else 1f
            if (material.transmission > 0f) {
                val lightness = 0.299f * sr + 0.587f * sg + 0.114f * sb
                val seeThrough = SEE_THROUGH_DARK + (SEE_THROUGH_CLEAR - SEE_THROUGH_DARK) * lightness
                alpha *= 1f - material.transmission * (1f - seeThrough)
            }
            return Color.argb((alpha * 255).roundToInt(), (sr * 255).roundToInt(), (sg * 255).roundToInt(), (sb * 255).roundToInt())
        }

        private fun material(json: JSONObject): Material {
            val pbr = json.optJSONObject("pbrMetallicRoughness") ?: JSONObject()
            val factor = pbr.optJSONArray("baseColorFactor")?.let { f -> FloatArray(4) { f.getDouble(it).toFloat() } }
                ?: floatArrayOf(1f, 1f, 1f, 1f)
            var pixels: IntArray? = null
            var width = 0
            var height = 0
            var texCoord = 0
            pbr.optJSONObject("baseColorTexture")?.let { ref ->
                texCoord = ref.optInt("texCoord", 0)
                val texture = gltf.getJSONArray("textures").getJSONObject(ref.getInt("index"))
                val image = gltf.getJSONArray("images").getJSONObject(texture.getInt("source"))
                if (image.has("bufferView")) {
                    val view = views.getJSONObject(image.getInt("bufferView"))
                    val bitmap = BitmapFactory.decodeByteArray(bytes, binStart + view.optInt("byteOffset", 0), view.getInt("byteLength"))
                    if (bitmap != null) {
                        width = bitmap.width
                        height = bitmap.height
                        pixels = IntArray(width * height).also { bitmap.getPixels(it, 0, width, 0, 0, width, height) }
                        bitmap.recycle()
                    }
                }
            }
            val transmission = json.optJSONObject("extensions")?.optJSONObject("KHR_materials_transmission")
                ?.optDouble("transmissionFactor", 0.0)?.toFloat() ?: 0f
            return Material(factor, pixels, width, height, texCoord, transmission, json.optString("alphaMode") == "BLEND",
                pbr.optDouble("metallicFactor", 1.0).toFloat(), pbr.optDouble("roughnessFactor", 1.0).toFloat())
        }

        /** Accessor [index] as floats, normalised integers scaled to 0..1 or -1..1. */
        private fun floats(index: Int): FloatArray {
            val a = accessors.getJSONObject(index)
            require(!a.has("sparse")) { "sparse accessors are not supported" }
            val n = components(a.getString("type"))
            val type = a.getInt("componentType")
            val size = componentSize(type)
            val count = a.getInt("count")
            val (base, stride) = layout(a, n * size)
            val normalised = a.optBoolean("normalized", false)
            return FloatArray(count * n) { k ->
                val at = base + (k / n) * stride + (k % n) * size
                when (type) {
                    5126 -> buf.getFloat(at)
                    5121 -> (buf.get(at).toInt() and 0xFF).let { if (normalised) it / 255f else it.toFloat() }
                    5123 -> (buf.getShort(at).toInt() and 0xFFFF).let { if (normalised) it / 65535f else it.toFloat() }
                    5120 -> buf.get(at).toInt().let { if (normalised) maxOf(it / 127f, -1f) else it.toFloat() }
                    5122 -> buf.getShort(at).toInt().let { if (normalised) maxOf(it / 32767f, -1f) else it.toFloat() }
                    else -> buf.getInt(at).toFloat()
                }
            }
        }

        /** Accessor [index] as unsigned integers: triangle indices. */
        private fun ints(index: Int): IntArray {
            val a = accessors.getJSONObject(index)
            require(!a.has("sparse")) { "sparse accessors are not supported" }
            val type = a.getInt("componentType")
            val size = componentSize(type)
            val (base, stride) = layout(a, size)
            return IntArray(a.getInt("count")) { k ->
                val at = base + k * stride
                when (type) {
                    5121 -> buf.get(at).toInt() and 0xFF
                    5123 -> buf.getShort(at).toInt() and 0xFFFF
                    else -> buf.getInt(at)
                }
            }
        }

        /** Where accessor [a]'s data starts in the file, and the step from one element to the next. */
        private fun layout(a: JSONObject, packed: Int): Pair<Int, Int> {
            val view = views.getJSONObject(a.getInt("bufferView"))
            require(view.optInt("buffer", 0) == 0) { "only the GLB's own buffer is supported" }
            val base = binStart + view.optInt("byteOffset", 0) + a.optInt("byteOffset", 0)
            return base to view.optInt("byteStride", packed)
        }

        private fun components(type: String) = when (type) {
            "SCALAR" -> 1
            "VEC2" -> 2
            "VEC3" -> 3
            "VEC4" -> 4
            else -> throw IllegalArgumentException("unsupported accessor type $type")
        }

        private fun componentSize(type: Int) = when (type) {
            5120, 5121 -> 1
            5122, 5123 -> 2
            5125, 5126 -> 4
            else -> throw IllegalArgumentException("unsupported component type $type")
        }
    }
}
