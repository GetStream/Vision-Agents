package io.getstream.visionagents.android.core

import kotlin.time.Clock
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import kotlin.time.Instant

/**
 * What this device's camera showed lately, kept so the agent can look when it asks.
 *
 * The router asks with `get_video_frames` when its vision skill needs to see, naming the moment it
 * took the question and how many frames it wants. The frames are sampled off the call's own
 * camera twice a second, through a video filter that leaves every frame as it was, and the newest
 * eight are kept: about four seconds.
 */
internal class CameraFrames(private val clock: () -> Instant = { Clock.System.now() }) {
    class Frame(val id: Int, val capturedAt: Instant, val jpeg: ByteArray)

    /** A frame [due] said to keep, which [keep] records unless the camera was cleared since. */
    class Due(val at: Instant, val generation: Int)

    /** The camera has shown nothing recent enough. */
    class NoFrames : Exception(
        "The camera has shown nothing in the last few seconds; ask them to point it at what they mean again.",
    )

    private val lock = Any()
    private val kept = ArrayDeque<Frame>()
    private var lastKept: Instant? = null
    private var counter = 0
    private var generation = 0

    /**
     * Whether a frame is due now, and if so the moment it counts as taken. A due frame is the
     * caller's to encode and hand to [keep]; the next is due [EVERY] later whether or not it was.
     */
    fun due(): Due? = synchronized(lock) {
        val now = clock()
        val last = lastKept
        if (last != null && now - last < EVERY) return null
        lastKept = now
        Due(now, generation)
    }

    /** Keeps the frame [due] asked for, unless [clear] was called in between. */
    fun keep(due: Due, jpeg: ByteArray) {
        synchronized(lock) {
            if (due.generation != generation) return
            append(jpeg, due.at)
        }
    }

    fun record(jpeg: ByteArray, capturedAt: Instant) {
        synchronized(lock) { append(jpeg, capturedAt) }
    }

    /** Forgets every frame, for a camera turned off: what it showed before is not what it shows. */
    fun clear() {
        synchronized(lock) {
            kept.clear()
            lastKept = null
            generation++
        }
    }

    /**
     * The newest [limit] frames taken at or before [at], oldest first, as the router's own workers
     * pick them: never a newer frame in place of one the question was asked about.
     *
     * @throws NoFrames when the newest of them is older than [FRESH_FOR] at [at].
     */
    fun select(at: Instant, limit: Int): List<Frame> = synchronized(lock) {
        val before = kept.filter { it.capturedAt <= at + SKEW }
        val newest = before.lastOrNull()
        if (newest == null || at - newest.capturedAt > FRESH_FOR) throw NoFrames()
        before.takeLast(limit.coerceIn(1, KEEP))
    }

    private fun append(jpeg: ByteArray, capturedAt: Instant) {
        counter++
        kept.addLast(Frame(counter, capturedAt, jpeg))
        while (kept.size > KEEP) kept.removeFirst()
    }

    companion object {
        /** The tool the router asks for frames with, answered here rather than offered to the model. */
        const val TOOL_NAME = "get_video_frames"

        /** How often a frame is kept. */
        val EVERY = 500.milliseconds

        /** How many are kept. */
        const val KEEP = 8

        /** How old the newest may be at the moment asked about, as the router's own workers allow. */
        val FRESH_FOR = 5.seconds

        /**
         * How far this device's clock may run ahead of the router's: a frame taken just after the
         * question by this clock may be just before it by the router's.
         */
        val SKEW = 2.seconds

        /** The longest side a frame is kept at, in pixels. */
        const val LONGEST_SIDE = 1024

        /** The JPEG quality a frame is kept at. */
        const val QUALITY = 70
    }
}
