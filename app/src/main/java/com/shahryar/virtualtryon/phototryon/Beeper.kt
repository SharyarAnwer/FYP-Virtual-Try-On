package com.shahryar.virtualtryon.phototryon

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

/**
 * Short tones for the guided capture, played at media volume so they carry across a room. Each view of
 * the turn gets a beep a little higher than the last, so a wearer facing away from the phone can hear
 * how far round they are; the end gets a rising chord of its own.
 */
class Beeper {

    companion object {
        private const val RATE = 44_100
        private const val BASE_HZ = 523.25f          // C5
        private const val STEP_SEMITONES = 2         // each view a whole tone higher
    }

    private val player = Executors.newSingleThreadExecutor()
    private var track: AudioTrack? = null

    /** The beep for having [count] views of all [total]: higher as the count rises. */
    fun progress(count: Int) = play(listOf(BASE_HZ * 2f.pow(STEP_SEMITONES * (count - 1) / 12f) to 150))

    /** Three rising notes: all views are taken. */
    fun done() = play(listOf(784f to 140, 988f to 140, 1319f to 320))

    fun release() {
        player.execute {
            track?.release()
            track = null
        }
        player.shutdown()
    }

    private fun play(notes: List<Pair<Float, Int>>) {
        if (player.isShutdown) return
        player.execute {
            val t = track ?: AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build())
                .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(RATE).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes(AudioTrack.getMinBufferSize(RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT) * 4)
                .build().also { track = it; it.play() }
            for ((hz, ms) in notes) t.write(tone(hz, ms), 0, RATE * ms / 1000)
        }
    }

    /** A sine tone with soft ends, so it does not click. */
    private fun tone(hz: Float, ms: Int): ShortArray {
        val n = RATE * ms / 1000
        val fade = min(n / 4, RATE / 100)
        return ShortArray(n) { i ->
            val edge = min(1f, min(i, n - 1 - i).toFloat() / fade)
            (sin(2 * PI * hz * i / RATE) * edge * 0.6 * Short.MAX_VALUE).toInt().toShort()
        }
    }
}
