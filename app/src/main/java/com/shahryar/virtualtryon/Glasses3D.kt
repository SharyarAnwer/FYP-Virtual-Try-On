package com.shahryar.virtualtryon

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * A pair of glasses as a 3D shape, worn on the face Face Mesh finds and drawn with the view's own
 * Canvas (M5).
 *
 * The shape is a [GlassesShape]: a real 3D model read from a file, or the plain black pair built
 * in code if it cannot be read. Each frame the head's position and turn come from four Face Mesh
 * points: the outer eye corners give the left-right axis and the scale, forehead to chin the
 * up-down axis, and the two together which way the face points. The glasses are turned and
 * scaled into that frame and drawn as triangles, farthest first, so nearer parts cover farther
 * ones. Each corner is shaded as the model's material describes it: a matt part by how squarely
 * it faces the camera, its own colour where it faces it; a metal part by what it reflects, a
 * plain room bright above and dark below; a polished one with a highlight from a light above
 * and to the left. The colour is blended across each triangle, so a curved model looks smooth.
 * Seen straight on, the camera is far enough away that perspective is left out: depth decides
 * only what is drawn over what.
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
 * No 3D engine is involved: drawing the triangles back to front with Canvas.drawVertices keeps
 * the glasses in the same view, and the same order, as the garments.
 */
class Glasses3D(private val shape: GlassesShape) {

    companion object {
        // Face Mesh points.
        private const val RIGHT_EYE_OUTER = 33
        private const val LEFT_EYE_OUTER = 263
        private const val FOREHEAD = 10
        private const val CHIN = 152

        /** Distance between an adult's outer eye corners, mm: sets the glasses' scale. */
        const val EYE_CORNERS_MM = 92f

        /** How far in front of the outer eye corners the front of the lenses sits, mm. */
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

        // Matt surfaces are lit from the camera: edge-on AMBIENT, facing it AMBIENT + DIFFUSE = 1,
        // its own colour exactly.
        private const val AMBIENT = 0.45f
        private const val DIFFUSE = 0.55f

        // Metal shows what it reflects: a plain room around the head, bright above, dimmer at eye
        // level, dark below.
        private const val SKY = 1f
        private const val HORIZON = 0.7f
        private const val FLOOR = 0.2f

        // One light, up and to the left in front of the face, as unit vector towards it (view
        // axes: x right, y down, z away from the camera). Its highlight is HIGHLIGHT strong on
        // polished metal, NON_METAL_SHINE of that on plastic and glass, none on matt surfaces;
        // its sharpness runs from HIGHLIGHT_ROUGH to HIGHLIGHT_SMOOTH with polish.
        private const val LIGHT_X = -0.398f
        private const val LIGHT_Y = -0.6965f
        private const val LIGHT_Z = -0.597f
        private const val HIGHLIGHT = 0.6f
        private const val NON_METAL_SHINE = 0.04f
        private const val HIGHLIGHT_ROUGH = 4f
        private const val HIGHLIGHT_SMOOTH = 24f

        /** Face Mesh's outline of the face, in order round it. */
        private val FACE_OVAL = intArrayOf(10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109)
    }

    private val triangles = shape.triangles

    // Per frame scratch, allocated once.
    private val placed = FloatArray(triangles * 9)
    private val hidden = BooleanArray(triangles * 3)
    private val keys = LongArray(triangles)
    private val verts = FloatArray(triangles * 6)
    private val vertColours = IntArray(triangles * 3)

    /**
     * White: drawVertices multiplies each corner's colour by the paint's, so the default black
     * paint would turn every part black and leave only its alpha.
     */
    private val paint = Paint().apply { color = Color.WHITE }

    private val oval = FloatArray(FACE_OVAL.size * 2)
    private var ovalLeft = 0f
    private var ovalTop = 0f
    private var ovalRight = 0f
    private var ovalBottom = 0f
    private val smooth = FloatArray(12)
    private var smoothed = false

    /** Whether the view point ([x], [y]) is inside the face's outline, by ray casting. */
    private fun inFace(x: Float, y: Float): Boolean {
        if (x < ovalLeft || x > ovalRight || y < ovalTop || y > ovalBottom) return false
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
        // the camera. Mirroring may make it a reflection, which swaps the glasses' left and
        // right; a pair is near enough symmetric for that not to show.
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

        ovalLeft = Float.MAX_VALUE; ovalTop = Float.MAX_VALUE
        ovalRight = -Float.MAX_VALUE; ovalBottom = -Float.MAX_VALUE
        for ((k, id) in FACE_OVAL.withIndex()) {
            val x = face[id * 3] * sx + dx
            val y = face[id * 3 + 1] * sy + dy
            oval[k * 2] = x
            oval[k * 2 + 1] = y
            ovalLeft = minOf(ovalLeft, x); ovalRight = maxOf(ovalRight, x)
            ovalTop = minOf(ovalTop, y); ovalBottom = maxOf(ovalBottom, y)
        }
        // The side of the model, +x or -x, turned towards the camera; 0 facing it. X's depth
        // grows away from the camera, so the side it points to is the far one.
        val nearSide = if (xz < -NEAR_TURN) 1f else if (xz > NEAR_TURN) -1f else 0f

        val model = shape.positions
        for (v in 0 until triangles * 3) {
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

        val parts = shape.parts
        var n = 0
        for (t in 0 until triangles) {
            val v = t * 3
            if (hidden[v] || hidden[v + 1] || hidden[v + 2]) continue
            val part = parts[t]
            if (part != GlassesShape.PART_FRAME) {
                // The near arm lies over the temple turned to the camera: inside the face's
                // outline, but in view. The far arm and both ear ends hide there.
                val nearArm = part == GlassesShape.PART_ARM && model[t * 9] * nearSide > 0f
                if (!nearArm && (inFace(placed[v * 3], placed[v * 3 + 1]) || inFace(placed[v * 3 + 3], placed[v * 3 + 4]) ||
                        inFace(placed[v * 3 + 6], placed[v * 3 + 7]))) continue
            }
            val depth = placed[v * 3 + 2] + placed[v * 3 + 5] + placed[v * 3 + 8]
            var bits = java.lang.Float.floatToIntBits(-depth)
            if (bits < 0) bits = bits xor 0x7fffffff
            keys[n++] = (bits.toLong() shl 32) or t.toLong()
        }
        java.util.Arrays.sort(keys, 0, n)

        // Shading, for the triangles drawn only: each corner by its material, as the model gives it.
        val colours = shape.colours
        val normals = shape.normals
        for (i in 0 until n) {
            val t = (keys[i] and 0xffffffffL).toInt()
            val metal = shape.metallic[t]
            val rough = shape.roughness[t]
            val shine = (NON_METAL_SHINE + (1f - NON_METAL_SHINE) * metal) * (1f - rough) * HIGHLIGHT
            val sharpness = (HIGHLIGHT_ROUGH + (HIGHLIGHT_SMOOTH - HIGHLIGHT_ROUGH) * (1f - rough)).toDouble()
            for (c in 0 until 3) {
                val v = t * 3 + c
                verts[i * 6 + c * 2] = placed[v * 3]
                verts[i * 6 + c * 2 + 1] = placed[v * 3 + 1]
                // The corner's normal in view, turned to face the camera: an inside face showing
                // through is lit like an outside one.
                val nx = normals[v * 3]
                val ny = normals[v * 3 + 1]
                val nz = normals[v * 3 + 2]
                var wx = nx * xx + ny * yx + nz * zx
                var wy = nx * xy + ny * yy + nz * zy
                var wz = nx * xz + ny * yz + nz * zz
                if (wz > 0f) {
                    wx = -wx; wy = -wy; wz = -wz
                }
                // Matt: by how squarely the surface faces the camera.
                val diffuse = AMBIENT - DIFFUSE * wz
                // Metal: the room in the direction the line of sight bounces off the surface.
                val rx = -2f * wz * wx
                val ry = -2f * wz * wy
                val rz = 1f - 2f * wz * wz
                val up = -ry
                val room = if (up >= 0f) HORIZON + (SKY - HORIZON) * up else HORIZON + (HORIZON - FLOOR) * up
                val k = ((1f - metal) * diffuse + metal * room) * shape.brighten
                var add = 0f
                if (shine > 0f) {
                    val towards = rx * LIGHT_X + ry * LIGHT_Y + rz * LIGHT_Z
                    if (towards > 0f) add = shine * towards.toDouble().pow(sharpness).toFloat() * 255f
                }
                val base = colours[v]
                vertColours[i * 3 + c] = Color.argb(Color.alpha(base),
                    min(255, (Color.red(base) * k + add).toInt()),
                    min(255, (Color.green(base) * k + add).toInt()),
                    min(255, (Color.blue(base) * k + add).toInt()))
            }
        }
        if (n > 0) {
            canvas.drawVertices(Canvas.VertexMode.TRIANGLES, n * 6, verts, 0, null, 0,
                vertColours, 0, null, 0, 0, paint)
        }
    }
}
