package io.getstream.visionagents.android.demo

import android.os.SystemClock
import io.getstream.video.android.core.call.audio.InputAudioFilter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.sqrt
import org.webrtc.AudioTrackSink

/**
 * How loud one side of the call is, measured from its audio as it flows.
 *
 * The call's own audio levels arrive a few times a second, too coarse to move with the words.
 * This hears every 10 ms of audio instead, so the glow swells on each syllable and settles in
 * each pause. The agent is heard through [sink], added to its track; this device's microphone
 * through [filter], set as the call's audio filter, which reads each recorded buffer before
 * WebRTC's processing and leaves it as it was. So unlike the agent's track, the microphone also
 * hears the agent coming out of the speaker: it is good for how loud the person is, not for
 * whether they are talking. Both are called on WebRTC's audio threads; [read] is called on the
 * main thread.
 *
 * @param audible The level, from 0 to 1, below which the voice counts as silent.
 */
class VoiceMeter(private val audible: Float) {
    private val lock = Any()
    private var sumOfSquares = 0f
    private var samples = 0
    private var lastAudible = Long.MIN_VALUE / 2

    /** Hears a track as it plays, for the agent. Its audio is 16-bit PCM. */
    val sink = AudioTrackSink { audio, bitsPerSample, _, _, _, _ ->
        if (bitsPerSample == 16) measure(audio)
    }

    /** Hears this device's microphone, for the person. Its audio is 16-bit PCM. */
    val filter = InputAudioFilter { _, _, _, audio -> measure(audio) }

    /** The voice's level since the last reading, from 0 to 1, or null when it is not talking. */
    fun read(): Float? {
        val now = SystemClock.elapsedRealtime()
        synchronized(lock) {
            try {
                if (now - lastAudible >= HOLD_MILLIS) return null
                if (samples == 0) return 0f
                return level(sumOfSquares, samples)
            } finally {
                sumOfSquares = 0f
                samples = 0
            }
        }
    }

    private fun measure(audio: ByteBuffer) {
        // A view of its own, so the buffer WebRTC goes on to use is left exactly as it was.
        val pcm = audio.duplicate().order(ByteOrder.nativeOrder()).asShortBuffer()
        val count = pcm.remaining()
        if (count == 0) return
        var sum = 0f
        for (index in 0 until count) {
            val value = pcm.get(index) / 32768f
            sum += value * value
        }
        val now = SystemClock.elapsedRealtime()
        synchronized(lock) {
            sumOfSquares += sum
            samples += count
            if (level(sum, count) > audible) lastAudible = now
        }
    }

    private companion object {
        /**
         * How long a voice counts as talking after it was last heard, so the gaps between words
         * and sentences read as pauses rather than the end of what is being said.
         */
        const val HOLD_MILLIS = 600L

        /** RMS scaled the way the web demo scales its microphone, so normal speech peaks near 1. */
        fun level(sumOfSquares: Float, samples: Int): Float = min(1f, sqrt(sumOfSquares / samples) * 4.2f)
    }
}
