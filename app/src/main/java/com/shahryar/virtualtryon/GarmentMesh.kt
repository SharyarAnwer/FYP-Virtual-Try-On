package com.shahryar.virtualtryon

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Where a garment's sleeve meets its body on one side, found in the photo by
 * tools/garment_points.py: the shoulder tip, the armpit, and a point below the armpit that
 * steers the boundary between the hanging cuff and the body side. The cuff hem's two ends give
 * the sleeve its own centre line and width; garments catalogued before they were recorded lack
 * them, and their sleeves fall back to following the shoulder-to-elbow anchors.
 */
/** The angle in degrees between two directions. */
internal fun degreesBetween(ax: Float, ay: Float, bx: Float, by: Float): Float {
    val n = hypot(ax, ay) * hypot(bx, by)
    if (n < 1e-6f) return 0f
    return Math.toDegrees(acos(((ax * bx + ay * by) / n).coerceIn(-1f, 1f)).toDouble()).toFloat()
}

data class Armhole(
    val shoulder: Anchor,
    val armpit: Anchor,
    val below: Anchor,
    val cuffOuter: Anchor? = null,
    val cuffInner: Anchor? = null
)

/**
 * One side's sleeve/torso boundary: shoulder tip -> armpit -> below, extended as rays past both
 * ends, with a signed distance that is positive on the sleeve side.
 *
 * Each piece's outward normal comes from its top-to-bottom direction. The upward ray *points* up,
 * and taking its normal from that direction flipped its sides: in the offline preview the neck
 * counted as sleeve on both sides at once.
 *
 * Mirrors side_boundary() and signed_distance() in tools/preview_mesh.py. Change them together.
 */
class ArmholeBoundary(armhole: Armhole, private val outward: Float) {

    private val originX = FloatArray(3)
    private val originY = FloatArray(3)
    private val dirX = FloatArray(3)
    private val dirY = FloatArray(3)
    private val length = FloatArray(3)
    private val normalX = FloatArray(3)
    private val normalY = FloatArray(3)

    init {
        val s = armhole.shoulder
        val a = armhole.armpit
        val b = armhole.below
        val segLen = hypot(a.x - s.x, a.y - s.y)
        val segX = (a.x - s.x) / segLen
        val segY = (a.y - s.y) / segLen
        val lowLen = hypot(b.x - a.x, b.y - a.y)
        val lowX = (b.x - a.x) / lowLen
        val lowY = (b.y - a.y) / lowLen

        // origin, direction, length (infinite for rays), top-to-bottom flow for the normal
        piece(0, s, -segX, -segY, Float.POSITIVE_INFINITY, segX, segY)
        piece(1, s, segX, segY, segLen, segX, segY)
        piece(2, a, lowX, lowY, Float.POSITIVE_INFINITY, lowX, lowY)
    }

    private fun piece(i: Int, o: Anchor, dx: Float, dy: Float, len: Float, flowX: Float, flowY: Float) {
        originX[i] = o.x
        originY[i] = o.y
        dirX[i] = dx
        dirY[i] = dy
        length[i] = len
        normalX[i] = outward * flowY
        normalY[i] = outward * -flowX
    }

    fun signedDistance(x: Float, y: Float): Float {
        var best = Float.POSITIVE_INFINITY
        var signed = 0f
        for (i in 0 until 3) {
            val rx = x - originX[i]
            val ry = y - originY[i]
            val t = (rx * dirX[i] + ry * dirY[i]).coerceIn(0f, length[i])
            val dist = hypot(x - (originX[i] + t * dirX[i]), y - (originY[i] + t * dirY[i]))
            if (dist < best) {
                best = dist
                val side = rx * normalX[i] + ry * normalY[i]
                signed = if (side >= 0f) dist else -dist
            }
        }
        return signed
    }
}

/**
 * The deformable grid a real garment photo is drawn through (M3.6).
 *
 * A regular grid covers the photo. Every vertex carries three weights - torso, left sleeve,
 * right sleeve - from which side of each armhole boundary it lies on, blended smoothly across a
 * band either side of it. Each frame the overlay moves every vertex to the weighted blend of where
 * each bone's transform would put it, and draws the photo bent over the grid. The photo stays
 * whole, so there is no cut to hide at the shoulder: M3.5's separate pieces worked for generated
 * shapes but would have left raw edges on a photograph.
 *
 * Mirrors grid() and vertex_weights() in tools/preview_mesh.py, which is how the warp was
 * checked before reaching a phone. Change them together.
 */
class GarmentMesh(garment: Garment) {

    companion object {
        /** Grid cell size in artwork px. Fine enough for several vertices across the blend band. */
        const val CELL = 16f

        /**
         * Most grid vertices a garment may use; a larger photo gets larger cells instead.
         *
         * The test phone's GPU driver crashed the app (SIGSEGV in a memcpy inside the Mali
         * driver, called from drawBitmapMesh) on the first real shirt photo: 1178 px wide, so
         * 4,875 vertices at 16 px cells, 78 KB of vertex data. The crash registers showed it
         * copying 4,876 vertices - one past the end. The polo, 3,185 vertices and 51 KB, never
         * crashed: an over-read is harmless inside a small allocation, and fatal at the end of a
         * large one, which Android places against an unmapped guard page. 3,600 keeps every
         * garment below 64 KB, the size proven on device.
         */
        const val MAX_VERTICES = 3600

        /** The cell size for an image: [CELL], grown until the grid fits [MAX_VERTICES]. */
        fun cellFor(width: Int, height: Int): Float {
            var cell = CELL
            while ((cells(width, cell) + 1) * (cells(height, cell) + 1) > MAX_VERTICES) cell += 1f
            return cell
        }

        internal fun cells(length: Int, cell: Float) = max(1, (length / cell).roundToInt())

        /** See [torsoWidth]: the narrowing from one row to the next that marks the hem. */
        const val HEM_DROP = 0.1f

        /** Blend half-width either side of the armhole, as a fraction of shoulder-joint width. */
        const val BLEND = 0.06f

        /**
         * How far above the armpit the torso widening starts, as a fraction of shoulder-to-hip
         * length; it is at full strength from the armpit down.
         */
        const val EASE_ABOVE = 0.1f

        /**
         * How far into each sleeve the torso widening fades out, in blend half-widths. With the
         * body widened in full at the armpit, the sleeve's underside next to it must move out too,
         * or the body's vertices overtake the sleeve's and the mesh folds into a sawtooth.
         */
        const val EASE_FADE = 3.5f

        /**
         * The chest band, for fitting the garment's body to one width (OverlayView.FitMode.CHEST):
         * from this far below the armpit, for this much further, both as fractions of
         * shoulder-to-hip length. Just below the armpit, where the wearer's own clothes lie
         * closest to the body.
         */
        const val CHEST_START = 0.05f
        const val CHEST_SPAN = 0.25f

        const val BONES = 3

        /**
         * How far the four-corner stretch of a sleeve may reach past its corners, in the corner
         * coordinates of [sleeveU] and [sleeveV]. A little, so the cuff hem and the fabric just
         * beyond the corners come along; not far, or a stray vertex is thrown across the body.
         */
        const val U_MIN = -0.1f
        const val U_MAX = 1.05f
        const val V_MIN = -0.1f
        const val V_MAX = 1f

        /** The torso near an armpit moves with it, fading over this x shoulder-joint width. */
        const val LIFT_RADIUS = 0.6f

        /**
         * Points on the body side of an armhole boundary follow the torso rather than the
         * sleeve's corners: half and half on the boundary, fully this many blend half-widths in.
         * The side seam just below the armpit sits in the blend band, so part of each point there
         * went with the sleeve; with the sleeve's underside now taken up under a raised arm, those
         * points were dragged up after it and stuck out of the shirt's side as a ragged flap.
         */
        const val BODY_SIDE = 0.3f

        internal fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
            val t = ((x - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
            return t * t * (3f - 2f * t)
        }
    }

    /** Cell size actually used, in artwork px. */
    val cell: Float = cellFor(garment.imageWidth, garment.imageHeight)
    val cols: Int = cells(garment.imageWidth, cell)
    val rows: Int = cells(garment.imageHeight, cell)
    val vertexCount = (cols + 1) * (rows + 1)

    /** Rest-pose vertex positions, x and y interleaved, spread evenly over the whole bitmap as
     *  Canvas.drawBitmapMesh requires. */
    val source = FloatArray(vertexCount * 2)

    /** Per vertex: torso, left sleeve, right sleeve. Always sums to 1. */
    val weights = FloatArray(vertexCount * BONES)

    private val left = ArmholeBoundary(garment.armholes.getValue("left"), outward = -1f)
    private val right = ArmholeBoundary(garment.armholes.getValue("right"), outward = 1f)

    /** The garment's centre line, which widening pushes away from. */
    private val centreX =
        (garment.anchor("left_shoulder")!!.x + garment.anchor("right_shoulder")!!.x) / 2f

    /**
     * How much of the widening each grid row receives: 0 down to [EASE_ABOVE] of the torso length
     * above the armpit, rising smoothly to 1 at the armpit, and 1 from there to the hem.
     *
     * Full at the armpit, so the shirt's side runs straight down from under the arm at one width.
     * Two earlier shapes failed on device:
     * - From the shoulder line down - a leftover from the fixed 18% widening - pushed each armhole
     *   out with the measured width, about 40% on the tester: the torso panel spread over his
     *   shoulders and upper arms and the underarm was stretched.
     * - From the armpit down left the shirt at the photo's width under the arm, so its side curved
     *   out below the armpit to the measured width, and with arms down pinched in under the cuffs.
     * Starting just above the armpit keeps the shoulders as the shoulders put them.
     *
     * This is the garment's *shape* of widening and belongs to the garment. The *amount* is the
     * caller's input - see [widen].
     */
    val easeRamp = FloatArray(rows + 1)

    /**
     * Per vertex, how much of the widening it receives: 1 over the torso and up to the armhole,
     * fading to 0 [EASE_FADE] blend half-widths into each sleeve. Sleeves take their width from the
     * arm; only the part of each next to the armhole moves out with the widened body.
     */
    val easeWeights = FloatArray(vertexCount)

    /** Each grid row's height on the torso: 0 on the shoulder anchors' line, 1 on the hips'. */
    val rowT = FloatArray(rows + 1)

    /** The armpits' height on the same scale as [rowT]. */
    val armpitT: Float

    /**
     * The garment body's width across the chest band, in artwork px: the median of [torsoWidth]
     * over the rows from [CHEST_START] to [CHEST_START] + [CHEST_SPAN] below the armpit. Set by
     * [measureTorso]; 0 until then.
     */
    var chestWidth = 0f
        private set

    /**
     * Width of the garment's body at each grid row, in artwork px: between its outermost body
     * pixels - opaque and on the torso side of both armholes - or 0 if the row has none. Filled
     * by [measureTorso].
     *
     * Outermost pixels rather than a walk out from the centre line: a shirt's front panels can
     * part at the hem below the last button, and a walk stopped at that gap, measuring the hem
     * row at half its width.
     *
     * A row more than [HEM_DROP] narrower than the row above takes that row's width. Going down a
     * garment, a sharp narrowing is the hem rounding off or the empty rows past it, not its cut,
     * and fitting those rows to the wearer pulled the hem's bottom edge out of line: rows past
     * the hem got no fitting, so the strip drawn between them and the hem stayed photo-width, and
     * a rounded hem's last row was widened extra to make up its missing width.
     */
    val torsoWidth = FloatArray(rows + 1)

    private val imageHeight = garment.imageHeight

    /**
     * Each sleeve's corners in the photo - shoulder tip, cuff outer end, cuff inner end, armpit -
     * as x, y pairs; index 0 is the left sleeve. Null when the catalog has no cuff ends for it.
     */
    val sleeveCorners: Array<FloatArray?> = arrayOf(corners(garment.armholes["left"]),
        corners(garment.armholes["right"]))

    /**
     * Per sleeve, every vertex's place between those corners: u from the shoulder towards the
     * cuff, v from the sleeve's top edge across to its underside, so that the vertex is the
     * bilinear blend of the four corners at (u, v). Clamped to [U_MIN]..[U_MAX], [V_MIN]..[V_MAX].
     */
    val sleeveU = Array(2) { FloatArray(vertexCount) }
    val sleeveV = Array(2) { FloatArray(vertexCount) }

    /** Per sleeve, how much each vertex moves with the armpit: 1 at it, 0 [LIFT_RADIUS] away. */
    val armpitLift = Array(2) { FloatArray(vertexCount) }

    /** Per sleeve, how far each vertex follows the torso instead of the sleeve; see [BODY_SIDE]. */
    val bodySide = Array(2) { FloatArray(vertexCount) }

    /** Per sleeve, the angle in degrees between its centre line and straight down the garment. */
    val sleeveAngle = FloatArray(2)

    /** Per sleeve, the winding of its corners in the photo: +1 or -1. */
    val sleeveWinding = FloatArray(2)

    init {
        val jointWidth = garment.anchor("right_shoulder")!!.x - garment.anchor("left_shoulder")!!.x
        val band = BLEND * jointWidth
        val fade = EASE_FADE * band

        val shoulderY = (garment.anchor("left_shoulder")!!.y + garment.anchor("right_shoulder")!!.y) / 2f
        val armpitY = (garment.armholes.getValue("left").armpit.y + garment.armholes.getValue("right").armpit.y) / 2f
        val hipY = (garment.anchor("left_hip")!!.y + garment.anchor("right_hip")!!.y) / 2f
        armpitT = (armpitY - shoulderY) / (hipY - shoulderY)
        for (r in 0..rows) {
            val y = r * garment.imageHeight.toFloat() / rows
            easeRamp[r] = smoothstep(armpitY - EASE_ABOVE * (hipY - shoulderY), armpitY, y)
            rowT[r] = (y - shoulderY) / (hipY - shoulderY)
        }
        var v = 0
        for (r in 0..rows) {
            val y = r * garment.imageHeight.toFloat() / rows
            for (c in 0..cols) {
                val x = c * garment.imageWidth.toFloat() / cols
                source[v * 2] = x
                source[v * 2 + 1] = y

                val sl = left.signedDistance(x, y)
                val sr = right.signedDistance(x, y)
                val wl = smoothstep(-band, band, sl)
                val wr = smoothstep(-band, band, sr)
                val wt = (1f - wl - wr).coerceAtLeast(0f)
                // A vertex's position is a weighted *average*; weights summing past 1 would throw
                // it beyond every bone's placement.
                val total = wt + wl + wr
                weights[v * BONES] = wt / total
                weights[v * BONES + 1] = wl / total
                weights[v * BONES + 2] = wr / total
                easeWeights[v] = 1f - max(smoothstep(0f, fade, sl), smoothstep(0f, fade, sr))
                v++
            }
        }

        val downX = (garment.anchor("left_hip")!!.x + garment.anchor("right_hip")!!.x -
            garment.anchor("left_shoulder")!!.x - garment.anchor("right_shoulder")!!.x) / 2f
        val downY = hipY - shoulderY
        val uv = FloatArray(2)
        for (s in 0..1) {
            val k = sleeveCorners[s] ?: continue
            for (i in 0 until vertexCount) {
                inverseBilinear(source[i * 2], source[i * 2 + 1], k, uv)
                sleeveU[s][i] = uv[0].coerceIn(U_MIN, U_MAX)
                sleeveV[s][i] = uv[1].coerceIn(V_MIN, V_MAX)
                val d = hypot(source[i * 2] - k[6], source[i * 2 + 1] - k[7])
                armpitLift[s][i] = smoothstep(LIFT_RADIUS * jointWidth, 0f, d)
                val depth = -(if (s == 0) left else right).signedDistance(source[i * 2], source[i * 2 + 1])
                bodySide[s][i] = smoothstep(-BODY_SIDE * band, BODY_SIDE * band, depth)
            }
            val ax = (k[2] + k[4] - k[0] - k[6]) / 2f
            val ay = (k[3] + k[5] - k[1] - k[7]) / 2f
            sleeveAngle[s] = degreesBetween(ax, ay, downX, downY)
            val cross = (k[2] - k[0]) * (k[7] - k[1]) - (k[3] - k[1]) * (k[6] - k[0])
            sleeveWinding[s] = if (cross >= 0f) 1f else -1f
        }
    }

    private fun corners(hole: Armhole?): FloatArray? {
        val outer = hole?.cuffOuter ?: return null
        val inner = hole.cuffInner ?: return null
        return floatArrayOf(hole.shoulder.x, hole.shoulder.y, outer.x, outer.y,
            inner.x, inner.y, hole.armpit.x, hole.armpit.y)
    }

    /**
     * Writes into [out] the (u, v) at which the bilinear blend of corners [k] (S, O, I, A) is the
     * point ([px], [py]). The blend is quadratic in v; of its two roots, the one that reproduces
     * the point, preferring inside the quad, wins. Points outside the quad extrapolate.
     *
     * Mirrors inverse_bilinear() in tools/preview_mesh.py. Change them together.
     */
    private fun inverseBilinear(px: Float, py: Float, k: FloatArray, out: FloatArray) {
        val ex = k[2] - k[0]; val ey = k[3] - k[1]
        val fx = k[6] - k[0]; val fy = k[7] - k[1]
        val gx = k[0] - k[2] + k[4] - k[6]; val gy = k[1] - k[3] + k[5] - k[7]
        val hx = px - k[0]; val hy = py - k[1]
        val k2 = gx * fy - gy * fx
        val k1 = ex * fy - ey * fx + hx * gy - hy * gx
        val k0 = hx * ey - hy * ex
        var bestErr = Float.POSITIVE_INFINITY
        val roots = if (abs(k2) < 1e-6f) floatArrayOf(-k0 / k1) else {
            val disc = sqrt(max(k1 * k1 - 4f * k0 * k2, 0f))
            floatArrayOf((-k1 - disc) / (2f * k2), (-k1 + disc) / (2f * k2))
        }
        for (v in roots) {
            val dx = ex + v * gx
            val dy = ey + v * gy
            val u = if (abs(dx) > abs(dy)) (hx - fx * v) / dx else (hy - fy * v) / dy
            val bx = (1 - u) * (1 - v) * k[0] + u * (1 - v) * k[2] + u * v * k[4] + (1 - u) * v * k[6]
            val by = (1 - u) * (1 - v) * k[1] + u * (1 - v) * k[3] + u * v * k[5] + (1 - u) * v * k[7]
            val err = hypot(bx - px, by - py) + 50f * (max(0f, -u) + max(0f, u - 1) + max(0f, -v) + max(0f, v - 1))
            if (err < bestErr) {
                bestErr = err
                out[0] = u
                out[1] = v
            }
        }
    }

    /**
     * Where [widen] moves a point on an armhole boundary, as an x: its ease weight there is 1.
     * Row amounts are interpolated, as for the grid itself. An armpit moves out with the body's
     * full width; a shoulder tip, well above the armpit, stays put.
     */
    fun widenOnBoundaryX(x: Float, y: Float, amount: FloatArray): Float {
        val f = (y / imageHeight * rows).coerceIn(0f, rows.toFloat())
        val r0 = f.toInt().coerceAtMost(rows - 1)
        val w = f - r0
        val e = (1 - w) * amount[r0] * easeRamp[r0] + w * amount[r0 + 1] * easeRamp[r0 + 1]
        return x + (x - centreX) * e
    }

    /**
     * Writes into [out] the grid positions to skin from: [source] with the torso widened.
     *
     * [amount] holds one widening fraction per grid row (0.18 = 18% wider, -0.2 = 20% narrower),
     * scaled by [easeRamp]. Each vertex moves away from the centre line in proportion to its
     * [easeWeights] entry, so the torso below the armpits widens or narrows while the armholes and
     * sleeves are left to the shoulders and arms.
     *
     * The amount is an input on purpose. The garment's width otherwise comes only from the
     * distance between the shoulder joints, which says nothing about how broad the wearer is
     * below them, nor how a photo was taken: a flat-laid shirt shows its whole front spread out,
     * far wider than it looks worn. The overlay passes, per row, whatever makes the garment's body
     * as wide as the wearer's torso, measured live by [BodyWidthMeter].
     *
     * Mirrors widen() in tools/preview_mesh.py. Change them together.
     */
    fun widen(amount: FloatArray, out: FloatArray) {
        val stride = cols + 1
        for (r in 0..rows) {
            val e = amount[r] * easeRamp[r]
            for (c in 0..cols) {
                val v = r * stride + c
                val x = source[v * 2]
                out[v * 2] = x + easeWeights[v] * (x - centreX) * e
                out[v * 2 + 1] = source[v * 2 + 1]
            }
        }
    }

    /** Fills [torsoWidth] from the garment photo. Call once, when the garment is loaded. */
    fun measureTorso(photo: Bitmap) {
        val w = photo.width
        val pixels = IntArray(w)
        for (r in 0..rows) {
            val y = (r * photo.height / rows).coerceAtMost(photo.height - 1)
            photo.getPixels(pixels, 0, w, 0, y, w, 1)
            val py = y + 0.5f
            var lo = 0
            while (lo < w && !isBody(pixels, lo, py)) lo++
            var hi = w - 1
            while (hi > lo && !isBody(pixels, hi, py)) hi--
            val width = if (lo < w) (hi - lo + 1).toFloat() else 0f
            val above = if (r > 0) torsoWidth[r - 1] else 0f
            torsoWidth[r] = if (width < (1f - HEM_DROP) * above) above else width
        }
        val band = (0..rows)
            .filter { rowT[it] in (armpitT + CHEST_START)..(armpitT + CHEST_START + CHEST_SPAN) && torsoWidth[it] >= 1f }
            .map { torsoWidth[it] }
            .sorted()
        chestWidth = if (band.isEmpty()) 0f else band[band.size / 2]
    }

    /** Whether pixel [x] of a row at height [py] is the garment's body: opaque, not sleeve. */
    private fun isBody(row: IntArray, x: Int, py: Float): Boolean {
        if (row[x] ushr 24 < 128) return false
        val px = x + 0.5f
        return left.signedDistance(px, py) < 0f && right.signedDistance(px, py) < 0f
    }

    /**
     * A copy of the photo keeping only sleeve pixels, for the overlay's second pass.
     *
     * A single bent mesh draws its triangles in grid order, so where a warped sleeve overlaps the
     * body - an arm hanging close in, say - the body could land on top of it. Drawing the sleeves
     * again over the whole garment guarantees they stay in front.
     */
    fun sleevesOnly(front: Bitmap): Bitmap {
        val w = front.width
        val h = front.height
        val pixels = IntArray(w * h)
        front.getPixels(pixels, 0, w, 0, 0, w, h)
        for (y in 0 until h) {
            val py = y + 0.5f
            for (x in 0 until w) {
                val i = y * w + x
                if (pixels[i] ushr 24 == 0) continue
                val px = x + 0.5f
                val sleeve = left.signedDistance(px, py) >= 0f || right.signedDistance(px, py) >= 0f
                if (!sleeve) pixels[i] = 0
            }
        }
        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
    }
}
