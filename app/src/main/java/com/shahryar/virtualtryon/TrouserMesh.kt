package com.shahryar.virtualtryon

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Where a pair of trousers' waistband, crotch and hems lie in the photo, found by
 * tools/garment_points.py. Left and right are image sides, as for every anchor.
 */
data class Legs(
    /** The waistband's top outer corners. */
    val leftWaist: Anchor,
    val rightWaist: Anchor,
    /** The top of the gap between the legs. */
    val crotch: Anchor,
    /** Each leg's hem: its outer and inner corners. */
    val leftHemOuter: Anchor,
    val leftHemInner: Anchor,
    val rightHemOuter: Anchor,
    val rightHemInner: Anchor
)

/**
 * The deformable grid a pair of trousers is drawn through (M4).
 *
 * The same grid as [GarmentMesh], with three bones: the pelvis, which carries the waistband, and
 * one per leg, from hip to ankle. Every vertex's weights come from its height and its side:
 * - Above the hip anchors it follows the pelvis, and from [LEGS_BELOW_CROTCH] below the crotch
 *   down, its leg. In between it blends, so the seat and the top of each thigh turn partly with a
 *   leg that swings out, as cloth there does.
 * - Left of the crotch it belongs to the left leg, right of it to the right leg, blended across
 *   [SPLIT] either side, so the fabric where the legs meet is not torn between them.
 *
 * Mirrors leg_weights() in tools/preview_trousers.py. Change them together.
 */
class TrouserMesh(garment: Garment) {

    companion object {
        /** Pelvis, left leg, right leg. */
        const val BONES = 3

        /**
         * How far below the crotch the legs take over completely, as a fraction of the rise
         * (waistband to crotch). The pelvis keeps a little of the crotch itself, so the point
         * where the legs part is not pulled two ways when they spread.
         */
        const val LEGS_BELOW_CROTCH = 0.1f

        /** Half-width of the blend between the legs, as a fraction of the hip anchors' spacing. */
        const val SPLIT = 0.05f
    }

    /** Cell size actually used, in artwork px. */
    val cell: Float = GarmentMesh.cellFor(garment.imageWidth, garment.imageHeight)
    val cols: Int = GarmentMesh.cells(garment.imageWidth, cell)
    val rows: Int = GarmentMesh.cells(garment.imageHeight, cell)
    val vertexCount = (cols + 1) * (rows + 1)

    /** Rest-pose vertex positions, x and y interleaved, spread evenly over the whole bitmap as
     *  Canvas.drawBitmapMesh requires. */
    val source = FloatArray(vertexCount * 2)

    /** Per vertex: pelvis, left leg, right leg. Always sums to 1. */
    val weights = FloatArray(vertexCount * BONES)

    /** The garment's centre line, x: midway between the hip anchors. */
    val centreX: Float

    /** The grid row nearest the hip anchors. */
    val hipRow: Int

    /** Per leg (0 = left), per grid row: the leg's centre line, hip anchor to knee to ankle. */
    val legAxis = Array(2) { FloatArray(rows + 1) }

    /**
     * Per leg, per grid row: how far [widen] works about the leg's own centre line rather than
     * the garment's: 0 down to the hip anchors, rising smoothly to 1 at the knee.
     *
     * Widened about the garment's centre line throughout, the legs would move apart as well as
     * thicken, and no longer sit on the wearer's knees and ankles. Widened about their own centre
     * lines throughout, the two legs' inner edges would cross at the crotch, where they meet. So
     * the seat widens about the middle, the legs below the knee about themselves, and the thighs
     * splay a little at the top between the two, as a broader body's do.
     */
    val legBlend = Array(2) { FloatArray(rows + 1) }

    /**
     * Per leg, per grid row: the row's place along the leg - 0 at the hip anchor, 1 at the knee,
     * 2 at the ankle - the scale of [BodyWidthMeter.LEG_SAMPLES].
     */
    val legT = Array(2) { FloatArray(rows + 1) }

    /**
     * Per side, per grid row: from the centre line out to the garment's outermost pixel, in
     * artwork px; 0 where the row is empty. Filled by [measure].
     */
    val halfWidth = Array(2) { FloatArray(rows + 1) }

    /**
     * Per leg, per grid row below the crotch: from the leg's centre line to its outer and its
     * inner edge, in artwork px; 0 where the leg has no fabric. Filled by [measure].
     */
    val legOuter = Array(2) { FloatArray(rows + 1) }
    val legInner = Array(2) { FloatArray(rows + 1) }

    /** Per leg, the first grid row below the crotch with fabric on it; set by [measure]. */
    val firstLegRow = IntArray(2)

    private val crotchY: Float

    init {
        val legs = garment.legs!!
        val hips = arrayOf(garment.anchor("left_hip")!!, garment.anchor("right_hip")!!)
        val knees = arrayOf(garment.anchor("left_knee")!!, garment.anchor("right_knee")!!)
        val ankles = arrayOf(garment.anchor("left_ankle")!!, garment.anchor("right_ankle")!!)
        val hipY = (hips[0].y + hips[1].y) / 2f
        val waistY = (legs.leftWaist.y + legs.rightWaist.y) / 2f
        val legsFull = legs.crotch.y + LEGS_BELOW_CROTCH * (legs.crotch.y - waistY)
        val band = SPLIT * (hips[1].x - hips[0].x)
        centreX = (hips[0].x + hips[1].x) / 2f
        crotchY = legs.crotch.y
        hipRow = (hipY / garment.imageHeight * rows).roundToInt().coerceIn(0, rows)

        var v = 0
        for (r in 0..rows) {
            val y = r * garment.imageHeight.toFloat() / rows
            val leg = GarmentMesh.smoothstep(hipY, legsFull, y)
            for (c in 0..cols) {
                val x = c * garment.imageWidth.toFloat() / cols
                source[v * 2] = x
                source[v * 2 + 1] = y
                val left = GarmentMesh.smoothstep(legs.crotch.x + band, legs.crotch.x - band, x)
                weights[v * BONES] = 1f - leg
                weights[v * BONES + 1] = leg * left
                weights[v * BONES + 2] = leg * (1f - left)
                v++
            }
            for (s in 0..1) {
                val hip = hips[s]
                val knee = knees[s]
                val ankle = ankles[s]
                legAxis[s][r] = when {
                    y <= hip.y -> hip.x
                    y <= knee.y -> hip.x + (y - hip.y) / (knee.y - hip.y) * (knee.x - hip.x)
                    y <= ankle.y -> knee.x + (y - knee.y) / (ankle.y - knee.y) * (ankle.x - knee.x)
                    else -> ankle.x
                }
                legT[s][r] = if (y < knee.y) (y - hip.y) / (knee.y - hip.y)
                    else 1f + (y - knee.y) / (ankle.y - knee.y)
                legBlend[s][r] = GarmentMesh.smoothstep(hip.y, knee.y, y)
            }
        }
    }

    /**
     * Fills [halfWidth], [legOuter], [legInner] and [firstLegRow] from the garment photo. Call
     * once, when the garment is loaded.
     */
    fun measure(photo: Bitmap) {
        val w = photo.width
        val pixels = IntArray(w)
        fun opaque(x: Int) = pixels[x] ushr 24 >= 128
        firstLegRow.fill(-1)
        for (r in 0..rows) {
            val y = (r * photo.height / rows).coerceAtMost(photo.height - 1)
            photo.getPixels(pixels, 0, w, 0, y, w, 1)
            var lo = 0
            while (lo < w && !opaque(lo)) lo++
            if (lo >= w) continue
            var hi = w - 1
            while (!opaque(hi)) hi--
            halfWidth[0][r] = centreX - lo
            halfWidth[1][r] = hi + 1 - centreX
            if (y <= crotchY) continue

            // Each leg's run, from its outer edge inward to the gap - or to the centre line.
            var x = lo
            while (x + 1 < w && opaque(x + 1) && x + 1 < centreX) x++
            legOuter[0][r] = max(0f, legAxis[0][r] - lo)
            legInner[0][r] = max(0f, x + 1 - legAxis[0][r])
            x = hi
            while (x - 1 >= 0 && opaque(x - 1) && x - 1 > centreX) x--
            legOuter[1][r] = max(0f, hi + 1 - legAxis[1][r])
            legInner[1][r] = max(0f, legAxis[1][r] - x)
            for (s in 0..1) if (firstLegRow[s] < 0 && legOuter[s][r] > 0f) firstLegRow[s] = r
        }
        for (s in 0..1) if (firstLegRow[s] < 0) firstLegRow[s] = rows

        // Going down, a row more than HEM_DROP narrower than the one above takes its width, as in
        // GarmentMesh.torsoWidth: it is a hem's rounded corner or the rows past it, not the cut.
        // Fitted as they were, the hems' last rows were widened extra and flared out.
        for (widths in arrayOf(halfWidth[0], halfWidth[1], legOuter[0], legOuter[1], legInner[0], legInner[1])) {
            for (r in 1..rows) {
                if (widths[r] < (1f - GarmentMesh.HEM_DROP) * widths[r - 1]) widths[r] = widths[r - 1]
            }
        }
    }

    /**
     * Writes into [out] the grid positions to skin from: [source] widened, row by row and side by
     * side. [outer] and [inner] hold one widening fraction per leg per grid row (0.18 = 18% wider)
     * for the fabric outward of the row's pivot and for the fabric inward of it, towards the other
     * leg. The pivot is the garment's centre line at the waist and each leg's own centre line from
     * the knee down, blended by [legBlend].
     *
     * A point is never moved across the centre line, so legs widened hard cannot cross.
     *
     * Mirrors widen() in tools/preview_trousers.py. Change them together.
     */
    fun widen(outer: Array<FloatArray>, inner: Array<FloatArray>, out: FloatArray) {
        val stride = cols + 1
        for (r in 0..rows) {
            for (c in 0..cols) {
                val v = r * stride + c
                val x = source[v * 2]
                val side = if (x < centreX) 0 else 1
                val pivot = centreX + legBlend[side][r] * (legAxis[side][r] - centreX)
                val d = x - pivot
                val outward = if (side == 0) d <= 0f else d >= 0f
                val moved = x + (if (outward) outer[side][r] else inner[side][r]) * d
                out[v * 2] = if (side == 0) min(moved, centreX) else max(moved, centreX)
                out[v * 2 + 1] = source[v * 2 + 1]
            }
        }
    }
}
