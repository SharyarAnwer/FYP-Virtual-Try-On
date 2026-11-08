package com.shahryar.virtualtryon

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * A pair of glasses as a 3D shape, worn on the face Face Mesh finds and drawn with the view's own
 * Canvas (M5, 3D trial).
 *
 * The shape is built here from simple parts - two rims, a bridge, two arms - in millimetres, as
 * square tubes cut into triangles. Each frame the head's position and turn come from four Face
 * Mesh points: the outer eye corners give the left-right axis and the scale, forehead to chin the
 * up-down axis, and the two together which way the face points. The glasses are turned and
 * scaled into that frame and drawn as flat-shaded triangles, farthest first, so nearer parts
 * cover farther ones. Seen straight on, the camera is far enough away that perspective is left
 * out: depth decides only what is drawn over what.
 *
 * Parts behind the head are hidden, as on a person. The rims and bridge are always in front of
 * the face. The arms run back along the sides of the head and are hidden behind it, taken as an
 * ellipsoid behind the eyes: a point with that ellipsoid between it and the camera is not drawn.
 * The ellipsoid alone was not enough: pointed at the front where a face is broad and flat, it let
 * the far arm show through the far lens of a slightly turned head. So the far arm, and the ends
 * curling behind both ears, are also hidden wherever they fall inside the outline of the face
 * Face Mesh finds. The near arm is not: on a turned head it lies over the temple turned to the
 * camera, inside that outline but in plain view.
 *
 * No 3D engine is involved: drawing a few thousand triangles back to front with
 * Canvas.drawVertices keeps the glasses in the same view, and the same order, as the garments.
 */
class Glasses3D {

    companion object {
        // Face Mesh points.
        private const val RIGHT_EYE_OUTER = 33
        private const val LEFT_EYE_OUTER = 263
        private const val FOREHEAD = 10
        private const val CHIN = 152

        /** Distance between an adult's outer eye corners, mm: sets the glasses' scale. */
        const val EYE_CORNERS_MM = 92f

        /** How far in front of the outer eye corners the lenses sit, mm. */
        const val LENS_FORWARD_MM = 16f

        /**
         * The head ellipsoid: its centre behind and below the eye corners, and its half-axes, mm.
         * Narrower than the arms are apart, so an arm lies just outside it: visible on the side
         * turned to the camera, hidden on the far side.
         */
        const val HEAD_BACK_MM = 82f
        const val HEAD_DOWN_MM = 18f
        const val HEAD_HALF_WIDTH_MM = 62f
        const val HEAD_HALF_HEIGHT_MM = 110f
        const val HEAD_HALF_DEPTH_MM = 96f

        /** Exponential smoothing of the four face points per frame: lower is steadier but lags. */
        const val SMOOTHING = 0.5f

        /**
         * How far the head must be turned - the sine of about 3 degrees - before the arm on the
         * side turned to the camera counts as the near arm and stops hiding inside the face's
         * outline. Facing the camera both arms hide there, so jitter in a straight-on face cannot
         * flick an arm in and out.
         */
        const val NEAR_TURN = 0.05f

        // What each triangle is part of, for hiding inside the face's outline (see draw).
        private const val PART_FRAME: Byte = 0
        private const val PART_ARM: Byte = 1
        private const val PART_EAR: Byte = 2

        // The frame's shape, mm: lens width and height, bridge, how much the front wraps back.
        private const val LENS_W = 52f
        private const val LENS_H = 40f
        private const val LENS_RADIUS = 11f
        private const val BRIDGE = 18f
        private const val LENS_DROP = 3f
        private const val WRAP = 0.0022f

        /** Face Mesh's outline of the face, in order round it. */
        private val FACE_OVAL = intArrayOf(10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109)

        private val FRAME_COLOUR = Color.rgb(30, 30, 34)
        private val ARM_COLOUR = Color.rgb(42, 42, 48)
    }

    // The model: triangles as three x, y, z vertices in mm, and a colour per triangle.
    private val model: FloatArray
    private val colours: IntArray
    private val triangles: Int

    /**
     * Per triangle, which part it belongs to: the frame never hides inside the face's outline, an
     * arm only on the far side, the ends curling behind the ears on both sides.
     */
    private val parts: ByteArray

    // Per frame scratch, allocated once.
    private val placed: FloatArray
    private val hidden: BooleanArray
    private val keys: LongArray
    private val verts: FloatArray
    private val vertColours: IntArray
    private val paint = Paint()

    private val oval = FloatArray(FACE_OVAL.size * 2)
    private val smooth = FloatArray(12)
    private var smoothed = false

    init {
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

        model = tris.toFloatArray()
        colours = cols.toIntArray()
        parts = tags.toByteArray()
        triangles = colours.size
        placed = FloatArray(model.size)
        hidden = BooleanArray(model.size / 3)
        keys = LongArray(triangles)
        verts = FloatArray(triangles * 6)
        vertColours = IntArray(triangles * 3)
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
     * direction is the depth axis for a part in the frame's front ([refZ]), the side-to-side axis
     * for an arm. Each triangle is tagged [part] in [tags], or [PART_EAR] from segment [earFrom]
     * on: the arm's end curling behind the ear.
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

    /** Whether the view point ([x], [y]) is inside the face's outline, by ray casting. */
    private fun inFace(x: Float, y: Float): Boolean {
        var inside = false
        val n = FACE_OVAL.size
        var j = n - 1
        for (i in 0 until n) {
            val xi = oval[i * 2]; val yi = oval[i * 2 + 1]
            val xj = oval[j * 2]; val yj = oval[j * 2 + 1]
            if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside
            j = i
        }
        return inside
    }

    /** Forgets the smoothed face; call when the face is lost, so a new one starts fresh. */
    fun reset() {
        smoothed = false
    }

    /**
     * Draws the glasses on the face whose Face Mesh points are [face] (x, y, z per point,
     * normalised as FaceTracker returns them), mapped to view px by [scale], [dx] and [dy] with
     * the image [imageWidth] x [imageHeight] - the overlay's FILL_CENTER mapping.
     */
    fun draw(canvas: Canvas, face: FloatArray, imageWidth: Int, imageHeight: Int, scale: Float, dx: Float, dy: Float) {
        val sx = imageWidth * scale
        val sy = imageHeight * scale
        // x, y, z in view px for the four points; z on x's scale, as Face Mesh gives it.
        val ids = intArrayOf(RIGHT_EYE_OUTER, LEFT_EYE_OUTER, FOREHEAD, CHIN)
        for ((k, id) in ids.withIndex()) {
            val x = face[id * 3] * sx + dx
            val y = face[id * 3 + 1] * sy + dy
            val z = face[id * 3 + 2] * sx
            if (!smoothed) {
                smooth[k * 3] = x; smooth[k * 3 + 1] = y; smooth[k * 3 + 2] = z
            } else {
                smooth[k * 3] += SMOOTHING * (x - smooth[k * 3])
                smooth[k * 3 + 1] += SMOOTHING * (y - smooth[k * 3 + 1])
                smooth[k * 3 + 2] += SMOOTHING * (z - smooth[k * 3 + 2])
            }
        }
        smoothed = true

        // The head's frame: X along the eye corners, Y down the face square to X, Z away from
        // the camera. Mirroring may make it a reflection; the glasses are symmetric, so that
        // changes nothing.
        var xx = smooth[3] - smooth[0]
        var xy = smooth[4] - smooth[1]
        var xz = smooth[5] - smooth[2]
        val eyes = sqrt(xx * xx + xy * xy + xz * xz)
        if (eyes < 4f) return
        xx /= eyes; xy /= eyes; xz /= eyes
        var yx = smooth[9] - smooth[6]
        var yy = smooth[10] - smooth[7]
        var yz = smooth[11] - smooth[8]
        val along = yx * xx + yy * xy + yz * xz
        yx -= along * xx; yy -= along * xy; yz -= along * xz
        val yl = sqrt(yx * yx + yy * yy + yz * yz)
        if (yl < 1f) return
        yx /= yl; yy /= yl; yz /= yl
        var zx = xy * yz - xz * yy
        var zy = xz * yx - xx * yz
        var zz = xx * yy - xy * yx
        if (zz < 0f) {
            zx = -zx; zy = -zy; zz = -zz
        }
        val s = eyes / EYE_CORNERS_MM
        val mx = (smooth[0] + smooth[3]) / 2f
        val my = (smooth[1] + smooth[4]) / 2f
        val mz = (smooth[2] + smooth[5]) / 2f
        val ox = mx - zx * LENS_FORWARD_MM * s
        val oy = my - zy * LENS_FORWARD_MM * s
        val oz = mz - zz * LENS_FORWARD_MM * s
        val cx = mx + (zx * HEAD_BACK_MM + yx * HEAD_DOWN_MM) * s
        val cy = my + (zy * HEAD_BACK_MM + yy * HEAD_DOWN_MM) * s
        val cz = mz + (zz * HEAD_BACK_MM + yz * HEAD_DOWN_MM) * s
        // Towards the camera, (0, 0, -1), in the head's frame and scaled to the unit ellipsoid.
        val ux = -xz / HEAD_HALF_WIDTH_MM
        val uy = -yz / HEAD_HALF_HEIGHT_MM
        val uz = -zz / HEAD_HALF_DEPTH_MM
        val a = ux * ux + uy * uy + uz * uz

        for ((k, id) in FACE_OVAL.withIndex()) {
            oval[k * 2] = face[id * 3] * sx + dx
            oval[k * 2 + 1] = face[id * 3 + 1] * sy + dy
        }
        // The side of the model, +x or -x, turned towards the camera; 0 facing it. X's depth
        // grows away from the camera, so the side it points to is the far one.
        val nearSide = if (xz < -NEAR_TURN) 1f else if (xz > NEAR_TURN) -1f else 0f

        val points = model.size / 3
        for (v in 0 until points) {
            val px = model[v * 3]
            val py = model[v * 3 + 1]
            val pz = model[v * 3 + 2]
            val vx = ox + s * (px * xx + py * yx + pz * zx)
            val vy = oy + s * (px * xy + py * yy + pz * zy)
            val vz = oz + s * (px * xz + py * yz + pz * zz)
            placed[v * 3] = vx
            placed[v * 3 + 1] = vy
            placed[v * 3 + 2] = vz
            // Hidden if the ray from the point towards the camera passes through the head.
            val rx = vx - cx
            val ry = vy - cy
            val rz = vz - cz
            val qx = (rx * xx + ry * xy + rz * xz) / s / HEAD_HALF_WIDTH_MM
            val qy = (rx * yx + ry * yy + rz * yz) / s / HEAD_HALF_HEIGHT_MM
            val qz = (rx * zx + ry * zy + rz * zz) / s / HEAD_HALF_DEPTH_MM
            val c = qx * qx + qy * qy + qz * qz - 1f
            val b = 2f * (qx * ux + qy * uy + qz * uz)
            hidden[v] = c < 0f || (b < 0f && b * b - 4f * a * c >= 0f)
        }

        var n = 0
        for (t in 0 until triangles) {
            val v = t * 3
            if (hidden[v] || hidden[v + 1] || hidden[v + 2]) continue
            val part = parts[t]
            if (part != PART_FRAME) {
                // The near arm lies over the temple turned to the camera: inside the face's
                // outline, but in view. The far arm and both ear ends hide there.
                val nearArm = part == PART_ARM && model[t * 9] * nearSide > 0f
                if (!nearArm && (inFace(placed[v * 3], placed[v * 3 + 1]) || inFace(placed[v * 3 + 3], placed[v * 3 + 4]) ||
                        inFace(placed[v * 3 + 6], placed[v * 3 + 7]))) continue
            }
            val depth = placed[v * 3 + 2] + placed[v * 3 + 5] + placed[v * 3 + 8]
            var bits = java.lang.Float.floatToIntBits(-depth)
            if (bits < 0) bits = bits xor 0x7fffffff
            keys[n++] = (bits.toLong() shl 32) or t.toLong()
        }
        java.util.Arrays.sort(keys, 0, n)

        for (i in 0 until n) {
            val t = (keys[i] and 0xffffffffL).toInt()
            val v = t * 9
            val ax = placed[v]; val ay = placed[v + 1]; val az = placed[v + 2]
            val bx = placed[v + 3]; val by = placed[v + 4]; val bz = placed[v + 5]
            val qx = placed[v + 6]; val qy = placed[v + 7]; val qz = placed[v + 8]
            // Flat shading: lit from the camera, by how squarely the triangle faces it.
            val nx = (by - ay) * (qz - az) - (bz - az) * (qy - ay)
            val ny = (bz - az) * (qx - ax) - (bx - ax) * (qz - az)
            val nz = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
            val nl = sqrt(nx * nx + ny * ny + nz * nz)
            val light = if (nl > 0f) 0.45f + 0.55f * kotlin.math.abs(nz) / nl else 0.45f
            val base = colours[t]
            val colour = Color.rgb(min(255, (Color.red(base) * light * 1.6f).toInt()),
                min(255, (Color.green(base) * light * 1.6f).toInt()),
                min(255, (Color.blue(base) * light * 1.6f).toInt()))
            verts[i * 6] = ax; verts[i * 6 + 1] = ay
            verts[i * 6 + 2] = bx; verts[i * 6 + 3] = by
            verts[i * 6 + 4] = qx; verts[i * 6 + 5] = qy
            vertColours[i * 3] = colour
            vertColours[i * 3 + 1] = colour
            vertColours[i * 3 + 2] = colour
        }
        if (n > 0) {
            canvas.drawVertices(Canvas.VertexMode.TRIANGLES, n * 6, verts, 0, null, 0,
                vertColours, 0, null, 0, 0, paint)
        }
    }
}
