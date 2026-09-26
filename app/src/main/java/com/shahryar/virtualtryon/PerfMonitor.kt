package com.shahryar.virtualtryon

/**
 * Tracks throughput and per-frame processing cost for the camera analysis stream.
 *
 * Three deliberate choices here:
 *
 * 1. FPS is measured on the *analysis* stream, not the preview. The preview sits near 30 FPS
 *    almost regardless of load, so measuring it would hide exactly the problem this class
 *    exists to catch.
 *
 * 2. Latency is measured around the whole analyzer callback, so it always reflects the true
 *    end-to-end cost of a frame no matter what is added inside it.
 *
 * 3. That total is also broken into named stages. A single number tells you a frame is too
 *    slow but not which part to fix, and the two candidates here — preparing the bitmap versus
 *    running the model — have completely different remedies.
 *
 * Values are read from the main thread while being written from the analysis thread, hence the
 * volatile fields. A stale read is harmless: these drive a display, not a decision.
 */
class PerfMonitor(private val windowSize: Int = 30) {

    private val frameTimesNs = ArrayDeque<Long>()
    private val latenciesMs = ArrayDeque<Double>()
    private val prepMsWindow = ArrayDeque<Double>()
    private val inferMsWindow = ArrayDeque<Double>()
    private var frameStartNs = 0L

    /** Frames per second over the recent window. */
    @Volatile
    var fps: Double = 0.0
        private set

    /** Mean total time per frame, in milliseconds, over the recent window. */
    @Volatile
    var latencyMs: Double = 0.0
        private set

    /** Mean time spent converting the camera frame into model input. */
    @Volatile
    var prepMs: Double = 0.0
        private set

    /** Mean time spent inside the model. */
    @Volatile
    var inferenceMs: Double = 0.0
        private set

    /** Total frames processed since the last reset. */
    @Volatile
    var frameCount: Long = 0L
        private set

    /** Call as the first statement in the analyzer callback. */
    fun frameStarted() {
        frameStartNs = System.nanoTime()
    }

    /** Time spent turning the camera frame into an upright, mirrored bitmap. */
    fun recordPrep(ms: Double) {
        prepMs = push(prepMsWindow, ms)
    }

    /** Time spent inside the pose model. */
    fun recordInference(ms: Double) {
        inferenceMs = push(inferMsWindow, ms)
    }

    /** Call once all per-frame work is done, before releasing the frame. */
    fun frameFinished() {
        val nowNs = System.nanoTime()
        frameCount++

        latencyMs = push(latenciesMs, (nowNs - frameStartNs) / 1_000_000.0)

        frameTimesNs.addLast(nowNs)
        while (frameTimesNs.size > windowSize) frameTimesNs.removeFirst()
        if (frameTimesNs.size >= 2) {
            val spanNs = frameTimesNs.last() - frameTimesNs.first()
            if (spanNs > 0L) {
                fps = (frameTimesNs.size - 1) * 1_000_000_000.0 / spanNs
            }
        }
    }

    private fun push(window: ArrayDeque<Double>, value: Double): Double {
        window.addLast(value)
        while (window.size > windowSize) window.removeFirst()
        return window.average()
    }

    /** Clears every window. Call when the camera or model is rebound, so old timings don't skew. */
    fun reset() {
        frameTimesNs.clear()
        latenciesMs.clear()
        prepMsWindow.clear()
        inferMsWindow.clear()
        frameCount = 0L
        fps = 0.0
        latencyMs = 0.0
        prepMs = 0.0
        inferenceMs = 0.0
    }
}
