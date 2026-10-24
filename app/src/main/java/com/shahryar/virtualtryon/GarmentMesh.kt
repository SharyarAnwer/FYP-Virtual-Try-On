package com.shahryar.virtualtryon

import android.graphics.Bitmap
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Where a garment's sleeve meets its body on one side, found in the photo by
 * tools/garment_points.py: the shoulder tip, the armpit, and a point below the armpit that
 * steers the boundary between the hanging cuff and the body side. The cuff hem's two ends give
 * the sleeve its own centre line and width; garments catalogued before they were recorded lack
 * them, and their sleeves fall back to following the shoulder-to-elbow anchors.
 */
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

        private fun cells(length: Int, cell: Float) = max(1, (length / cell).roundToInt())

        /** See [torsoWidth]: the narrowing from one row to the next that marks the hem. */
        const val HEM_DROP = 0.1f

        /** Blend half-width either side of the armhole, as a fraction of shoulder-joint width. */
        const val BLEND = 0.06f

        /**
         * How far into each sleeve the torso widening fades out, in blend half-widths.
         *
         * Widening is full right up to the armhole and fades to nothing this far into the sleeve.
         * Fading it across the blend band instead, in step with the torso weight, folded the mesh:
         * an 18% push moved body vertices further than the band is wide, past the sleeve vertices
         * beside them, and on device the folds showed as a sawtooth of dark triangles at the
         * armpits. Spread over 3.5 half-widths, no vertex overtakes its neighbour.
         */
        const val EASE_FADE = 3.5f

        const val BONES = 3

        private fun smoothstep(edge0: Float, edge1: Float, x: Float): Float {
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
     * How much of the widening each grid row receives: 0 at shoulder-joint height, rising
     * smoothly to 1 at armpit height and staying 1 down to the hem. The shoulders therefore stay
     * exactly where the shoulder landmarks put them.
     *
     * Full by the armpit, because the body side can show from the armpit down. A ramp that only
     * began at the armpit was still part-way up where the body side came out from under the cuff,
     * and the shirt pinched in below each sleeve. How far a cuff hangs below the armpit depends on
     * the wearer's torso length - on the tester it barely does - so no ramp below the armpit is
     * safe.
     *
     * This is the garment's *shape* of widening and belongs to the garment. The *amount* is the
     * caller's input - see [widen].
     */
    val easeRamp = FloatArray(rows + 1)

    /**
     * Per vertex, how much of the widening it receives: 1 over the torso and up to the armhole,
     * fading to 0 [EASE_FADE] blend half-widths into each sleeve. The inner part of each sleeve
     * moves out with the widened body, as the armhole of a shirt does on a broader wearer.
     */
    val easeWeights = FloatArray(vertexCount)

    /** Each grid row's height on the torso: 0 on the shoulder anchors' line, 1 on the hips'. */
    val rowT = FloatArray(rows + 1)

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

    init {
        val jointWidth = garment.anchor("right_shoulder")!!.x - garment.anchor("left_shoulder")!!.x
        val band = BLEND * jointWidth
        val fade = EASE_FADE * band

        val shoulderY = (garment.anchor("left_shoulder")!!.y + garment.anchor("right_shoulder")!!.y) / 2f
        val armpitY = (garment.armholes.getValue("left").armpit.y + garment.armholes.getValue("right").armpit.y) / 2f
        val hipY = (garment.anchor("left_hip")!!.y + garment.anchor("right_hip")!!.y) / 2f
        for (r in 0..rows) {
            val y = r * garment.imageHeight.toFloat() / rows
            easeRamp[r] = smoothstep(shoulderY, armpitY, y)
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
    }

    /**
     * Writes into [out] the grid positions to skin from: [source] with the torso widened.
     *
     * [amount] holds one widening fraction per grid row (0.18 = 18% wider, -0.2 = 20% narrower),
     * scaled by [easeRamp]. Each vertex moves away from the centre line in proportion to its
     * [easeWeights] entry, so the torso widens or narrows and carries the inner part of each
     * sleeve with it, while the rest of each sleeve is left to its own bone.
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
