package com.shahryar.virtualtryon.phototryon

import kotlin.math.atan2
import kotlin.math.cbrt
import kotlin.math.pow

/** How light or deep a skin tone is, from the Individual Typology Angle's usual six bands. */
enum class SkinDepth { VERY_LIGHT, LIGHT, INTERMEDIATE, TAN, BROWN, DEEP }

/** Whether a skin tone leans yellow-golden, pink-red, or neither. */
enum class Undertone { WARM, NEUTRAL, COOL }

/** A measured skin tone: its band, its lean, the angle it was banded by, and its colour for showing. */
class SkinReading(val depth: SkinDepth, val undertone: Undertone, val itaDeg: Float, val hueDeg: Float, val rgb: Int)

/**
 * Reads skin tone from pixels of the face (Photo Try-On, stage 2), for the colour rules in
 * [PhotoRecommender].
 *
 * The pixels - from both cheeks and the forehead - are turned into CIE Lab, a colour space built
 * so that its lightness L and its two colour axes match what the eye sees. The darkest and
 * lightest of them are dropped (beard shadow, shine) and the rest are averaged.
 *
 * - **Depth** is the Individual Typology Angle, the measure dermatology uses to band skin colour
 *   from a photograph: the angle of (L - 50) against b. Its six usual bands are [SkinDepth].
 * - **Undertone** is the hue angle of the two colour axes: skin sits between red and yellow, and
 *   leans to one or the other. The two limits are starting values, to be checked on real faces.
 *
 * Light changes all of this: a warm bulb reads as a warm undertone and a dim room as deeper skin.
 * The capture screen refuses a face that is too dark, but cannot tell a bulb's colour.
 */
object SkinTone {

    /** Above this hue, degrees, skin leans yellow (warm); below [COOL_BELOW_DEG], pink (cool). */
    const val WARM_ABOVE_DEG = 60f
    const val COOL_BELOW_DEG = 48f

    /** The share of pixels dropped at each end of the lightness order. */
    const val TRIM = 0.2f

    /** Fewer pixels than this is not a reading. */
    const val MIN_PIXELS = 20

    /** An sRGB colour, each part 0 to 255, as CIE Lab under daylight (D65). */
    fun toLab(r: Int, g: Int, b: Int): FloatArray {
        fun linear(c: Int): Double {
            val v = c / 255.0
            return if (v <= 0.04045) v / 12.92 else ((v + 0.055) / 1.055).pow(2.4)
        }
        val lr = linear(r)
        val lg = linear(g)
        val lb = linear(b)
        val x = (0.4124 * lr + 0.3576 * lg + 0.1805 * lb) / 0.95047
        val y = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb
        val z = (0.0193 * lr + 0.1192 * lg + 0.9505 * lb) / 1.08883
        fun f(t: Double) = if (t > 0.008856) cbrt(t) else 7.787 * t + 16.0 / 116.0
        return floatArrayOf((116 * f(y) - 16).toFloat(), (500 * (f(x) - f(y))).toFloat(), (200 * (f(y) - f(z))).toFloat())
    }

    /** The band for an Individual Typology Angle, degrees. */
    fun depthFor(itaDeg: Float) = when {
        itaDeg > 55f -> SkinDepth.VERY_LIGHT
        itaDeg > 41f -> SkinDepth.LIGHT
        itaDeg > 28f -> SkinDepth.INTERMEDIATE
        itaDeg > 10f -> SkinDepth.TAN
        itaDeg > -30f -> SkinDepth.BROWN
        else -> SkinDepth.DEEP
    }

    fun undertoneFor(hueDeg: Float) = when {
        hueDeg > WARM_ABOVE_DEG -> Undertone.WARM
        hueDeg < COOL_BELOW_DEG -> Undertone.COOL
        else -> Undertone.NEUTRAL
    }

    /** The skin tone of [pixels], each 0xAARRGGBB, or null if there are too few. */
    fun read(pixels: IntArray): SkinReading? {
        if (pixels.size < MIN_PIXELS) return null
        val labs = pixels.map { toLab((it shr 16) and 0xFF, (it shr 8) and 0xFF, it and 0xFF) to it }.sortedBy { it.first[0] }
        val drop = (labs.size * TRIM).toInt()
        val kept = labs.subList(drop, labs.size - drop)
        val l = kept.map { it.first[0] }.average().toFloat()
        val a = kept.map { it.first[1] }.average().toFloat()
        val b = kept.map { it.first[2] }.average().toFloat()
        val ita = Math.toDegrees(atan2((l - 50f).toDouble(), b.toDouble())).toFloat()
        val hue = Math.toDegrees(atan2(b.toDouble(), a.toDouble())).toFloat()
        val red = kept.map { (it.second shr 16) and 0xFF }.average().toInt()
        val green = kept.map { (it.second shr 8) and 0xFF }.average().toInt()
        val blue = kept.map { it.second and 0xFF }.average().toInt()
        return SkinReading(depthFor(ita), undertoneFor(hue), ita, hue, (0xFF shl 24) or (red shl 16) or (green shl 8) or blue)
    }
}
