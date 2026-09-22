package com.shahryar.virtualtryon

/**
 * Tracks throughput and per-frame processing cost for the camera analysis stream.
 *
 * Two deliberate choices here:
 *
 * 1. FPS is measured on the *analysis* stream, not the preview. The preview sits near 30 FPS
 *    almost regardless of load, so measuring it would hide exactly the problem this class
 *    exists to catch. The analysis stream is where pose inference runs from M1 onward.
 *
 * 2. Latency is measured around the whole analyzer callback, so when inference is added in M1
 *    it is included automatically with no changes here.
 *
 * Values are read from the main thread while being written from the analysis thread, hence
 * the volatile fields. A stale read is harmless — these drive a display, not a decision.
 */
class PerfMonitor(private val windowSize: Int = 30) {

    private val frameTimesNs = ArrayDeque<Long>()
    private val latenciesMs = ArrayDeque<Double>()
    private var frameStartNs = 0L

    /** Frames per second over the recent window. */
    @Volatile
    var fps: Double = 0.0
        private set

    /** Mean processing time per frame, in milliseconds, over the recent window. */
    @Volatile
    var latencyMs: Double = 0.0
        private set

    /** Total frames processed since the last reset. */
    @Volatile
    var frameCount: Long = 0L
        private set

    /** Call as the first statement in the analyzer callback. */
    fun frameStarted() {
        frameStartNs = System.nanoTime()
    }

    /** Call once all per-frame work is done, before releasing the frame. */
    fun frameFinished() {
        val nowNs = System.nanoTime()
        frameCount++

        latenciesMs.addLast((nowNs - frameStartNs) / 1_000_000.0)
        while (latenciesMs.size > windowSize) latenciesMs.removeFirst()
        latencyMs = latenciesMs.average()

        frameTimesNs.addLast(nowNs)
        while (frameTimesNs.size > windowSize) frameTimesNs.removeFirst()
        if (frameTimesNs.size >= 2) {
            val spanNs = frameTimesNs.last() - frameTimesNs.first()
            if (spanNs > 0L) {
                fps = (frameTimesNs.size - 1) * 1_000_000_000.0 / spanNs
            }
        }
    }

    /** Clears the window. Call when the camera is rebound, so old timings don't skew the average. */
    fun reset() {
        frameTimesNs.clear()
        latenciesMs.clear()
        frameCount = 0L
        fps = 0.0
        latencyMs = 0.0
    }
}
