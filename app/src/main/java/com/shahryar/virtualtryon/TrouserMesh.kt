package com.shahryar.virtualtryon

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

    init {
        val legs = garment.legs!!
        val leftHip = garment.anchor("left_hip")!!
        val rightHip = garment.anchor("right_hip")!!
        val hipY = (leftHip.y + rightHip.y) / 2f
        val waistY = (legs.leftWaist.y + legs.rightWaist.y) / 2f
        val legsFull = legs.crotch.y + LEGS_BELOW_CROTCH * (legs.crotch.y - waistY)
        val band = SPLIT * (rightHip.x - leftHip.x)

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
        }
    }
}
