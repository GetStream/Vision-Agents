package io.getstream.visionagents.android.core

import android.graphics.Bitmap
import io.getstream.video.android.core.call.video.BitmapVideoFilter
import io.getstream.video.android.core.call.video.RawVideoFilter
import io.getstream.video.android.core.call.video.VideoFilter
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.reflect.KMutableProperty0
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.webrtc.SurfaceTextureHelper
import org.webrtc.VideoFrame

/**
 * Samples a call's outgoing camera into [frames], on top of whatever video filter the call had.
 *
 * A call has one filter, `call.videoFilter`, and Stream Video's own moderation blur uses it too. So
 * the filter already there is kept underneath and run first, and what is sampled is what it
 * published: a picture the call blurred is blurred for the agent as well. Every frame goes on as
 * that filter left it; one now and then is copied and encoded off the frame thread, and a frame
 * due while the last is still encoding is skipped rather than waited for.
 *
 * Something that sets the filter later replaces this one, as moderation does when it blurs and
 * again, with null, when it stops. [heal] puts this one back on top of whatever is there, and
 * [start] runs it every second while the camera is on, so sampling pauses for a second at most.
 * [stop] hands the slot back to what is underneath, if it is still this filter's.
 *
 * [slot] is where the filter goes: `call::videoFilter`.
 */
internal class CameraFilter(
    private val slot: KMutableProperty0<VideoFilter?>,
    private val frames: CameraFrames,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val encoding = AtomicBoolean(false)
    private val lock = Any()
    private var stopped = false

    /** The filter this one sits on and runs first, or null. */
    @Volatile internal var below: VideoFilter? = null
        private set

    /** This filter as it was last put in the slot, or null once stopped. */
    private var installed: VideoFilter? = null

    fun start() {
        heal()
        scope.launch {
            while (isActive) {
                delay(CHECK_EVERY)
                heal()
            }
        }
    }

    /** Puts this filter back on top of the slot, if something replaced it. */
    fun heal() {
        synchronized(lock) {
            if (stopped) return
            val current = slot.get()
            if (current != null && current === installed) return
            val wrapped = over(current) ?: return
            below = current
            installed = wrapped
            slot.set(wrapped)
        }
    }

    fun stop() {
        scope.cancel()
        synchronized(lock) {
            stopped = true
            if (installed != null && slot.get() === installed) slot.set(below)
            installed = null
            below = null
        }
    }

    /**
     * This filter over [under], in the shape [under] needs: a bitmap filter can only be run by the
     * call itself, so one over it is a bitmap filter too.
     */
    private fun over(under: VideoFilter?): VideoFilter? = when (under) {
        null -> Raw(null)
        is RawVideoFilter -> Raw(under)
        is BitmapVideoFilter -> Bitmaps(under)
        else -> null
    }

    private inner class Raw(private val under: RawVideoFilter?) : RawVideoFilter() {
        override fun applyFilter(videoFrame: VideoFrame, surfaceTextureHelper: SurfaceTextureHelper): VideoFrame {
            val out = under?.applyFilter(videoFrame, surfaceTextureHelper) ?: videoFrame
            sample { due ->
                // A camera frame is usually a texture, read back here on the thread that owns it.
                // The copy is this filter's to release once it is encoded.
                val i420 = out.buffer.toI420() ?: return@sample null
                val rotation = out.rotation
                Encoding(due) { i420.use { it.jpeg(rotation) } }
            }
            return out
        }
    }

    private inner class Bitmaps(private val under: BitmapVideoFilter) : BitmapVideoFilter() {
        override fun applyFilter(videoFrameBitmap: Bitmap) {
            under.applyFilter(videoFrameBitmap)
            // The call recycles the bitmap once this returns, so the copy is taken now.
            sample { due ->
                val copy = videoFrameBitmap.scaledCopy()
                Encoding(due) {
                    try {
                        copy.jpeg()
                    } finally {
                        copy.recycle()
                    }
                }
            }
        }
    }

    private class Encoding(val due: CameraFrames.Due, val encode: () -> ByteArray?)

    /** Takes a frame if one is due and none is still encoding, and encodes it off this thread. */
    private fun sample(take: (CameraFrames.Due) -> Encoding?) {
        if (!encoding.compareAndSet(false, true)) return
        val encoding = try {
            frames.due()?.let(take)
        } catch (e: RuntimeException) {
            // A frame that cannot be read is a frame not kept; the call's own frame is untouched.
            null
        }
        if (encoding == null) {
            this.encoding.set(false)
            return
        }
        Dispatchers.Default.asExecutor().execute {
            try {
                encoding.encode()?.let { frames.keep(encoding.due, it) }
            } catch (e: RuntimeException) {
                // As above: nothing is kept, and the next frame due is tried as usual.
            } finally {
                this.encoding.set(false)
            }
        }
    }

    private companion object {
        val CHECK_EVERY = 1.seconds
    }
}

private inline fun <T> VideoFrame.I420Buffer.use(block: (VideoFrame.I420Buffer) -> T): T =
    try {
        block(this)
    } finally {
        release()
    }
