package io.getstream.visionagents.android.core

import android.graphics.Bitmap
import io.getstream.video.android.core.call.video.BitmapVideoFilter
import io.getstream.video.android.core.call.video.RawVideoFilter
import io.getstream.video.android.core.call.video.VideoFilter
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame
import org.webrtc.SurfaceTextureHelper
import org.webrtc.VideoFrame

/** The one filter a call has, as `call.videoFilter` is. */
private class Slot {
    var filter: VideoFilter? = null
}

private class AppFilter : RawVideoFilter() {
    override fun applyFilter(videoFrame: VideoFrame, surfaceTextureHelper: SurfaceTextureHelper) = videoFrame
}

private class Blur : BitmapVideoFilter() {
    override fun applyFilter(videoFrameBitmap: Bitmap) = Unit
}

class CameraFilterTest {
    private val slot = Slot()
    private val filter = CameraFilter(slot::filter, CameraFrames())

    @AfterTest
    fun stop() = filter.stop()

    @Test
    fun `a filter the app set stays underneath and comes back when the camera stops`() {
        val app = AppFilter()
        slot.filter = app

        filter.heal()

        assertIs<RawVideoFilter>(slot.filter)
        assertSame(app, filter.below)
        filter.stop()
        assertSame(app, slot.filter)
    }

    @Test
    fun `a moderation blur set over it is kept underneath until moderation clears it`() {
        filter.heal()
        val blur = Blur()

        slot.filter = blur
        filter.heal()

        // The blur can only be run by the call itself, so what sits over it is a bitmap filter,
        // and what the agent is shown is blurred too.
        assertIs<BitmapVideoFilter>(slot.filter)
        assertSame(blur, filter.below)

        slot.filter = null
        filter.heal()
        assertIs<RawVideoFilter>(slot.filter)
        assertNull(filter.below)
    }

    @Test
    fun `stopping leaves a filter somebody else set since`() {
        filter.heal()
        val blur = Blur()
        slot.filter = blur

        filter.stop()

        assertSame(blur, slot.filter)
    }

    @Test
    fun `a stopped filter is not put back`() {
        filter.heal()
        assertNotNull(slot.filter)
        filter.stop()

        filter.heal()

        assertNull(slot.filter)
    }
}
