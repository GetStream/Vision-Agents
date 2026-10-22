package io.getstream.visionagents.android.core

import java.nio.ByteBuffer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import kotlin.time.Instant

class CameraFramesTest {
    private val start = Instant.fromEpochSeconds(1_800_000_000)
    private var now = start
    private val frames = CameraFrames { now }

    private fun at(offset: Duration) = start + offset

    private fun offer(offset: Duration) {
        now = at(offset)
        frames.due()?.let { frames.keep(it, byteArrayOf(1)) }
    }

    @Test
    fun `a frame is kept twice a second at most`() {
        offer(Duration.ZERO)
        offer(200.milliseconds)
        offer(600.milliseconds)

        assertEquals(2, frames.select(at(1.seconds), limit = 8).size)
    }

    @Test
    fun `only the newest eight are kept`() {
        for (i in 0 until 12) frames.record(byteArrayOf(i.toByte()), at(250.milliseconds * i))

        val kept = frames.select(at(3.seconds), limit = 8)

        assertEquals((4 until 12).map { listOf(it.toByte()) }, kept.map { it.jpeg.toList() })
    }

    @Test
    fun `the newest frames at the moment are given oldest first`() {
        for (i in 0 until 6) frames.record(byteArrayOf(i.toByte()), at(i.seconds))

        val picked = frames.select(at(2500.milliseconds), limit = 2)

        // Frames up to the moment, and the clock skew past it a phone may run ahead by.
        assertEquals(listOf(listOf<Byte>(3), listOf<Byte>(4)), picked.map { it.jpeg.toList() })
    }

    @Test
    fun `nothing recent enough is said rather than an old frame given`() {
        frames.record(byteArrayOf(1), start)

        assertFailsWith<CameraFrames.NoFrames> {
            frames.select(start + CameraFrames.FRESH_FOR + 1.seconds, limit = 1)
        }
    }

    @Test
    fun `a camera turned off forgets what it showed`() {
        frames.record(byteArrayOf(1), start)

        frames.clear()

        assertFailsWith<CameraFrames.NoFrames> { frames.select(start, limit = 1) }
    }

    @Test
    fun `a frame still encoding when the camera is turned off is not kept`() {
        val due = assertNotNull(frames.due())

        frames.clear()
        frames.keep(due, byteArrayOf(1))

        assertFailsWith<CameraFrames.NoFrames> { frames.select(start, limit = 1) }
        assertNotNull(frames.due(), "the next frame is due at once")
    }

    @Test
    fun `however many are asked for, at least one and at most eight come back`() {
        for (i in 0 until 10) frames.record(byteArrayOf(i.toByte()), at(250.milliseconds * i))

        assertEquals(1, frames.select(at(3.seconds), limit = 0).size)
        assertEquals(8, frames.select(at(3.seconds), limit = 50).size)
    }
}

class Nv21Test {
    /** A 4x2 picture whose luma counts 0..7 along its rows, with 2x1 chroma planes. */
    private val y = plane(byteArrayOf(0, 1, 2, 3, 4, 5, 6, 7), stride = 4)
    private val u = plane(byteArrayOf(10, 11), stride = 2)
    private val v = plane(byteArrayOf(20, 21), stride = 2)

    private fun plane(bytes: ByteArray, stride: Int) = Plane(ByteBuffer.wrap(bytes), stride)

    private fun luma(picture: Nv21) = picture.bytes.take(picture.width * picture.height).map { it.toInt() }

    private fun chroma(picture: Nv21) = picture.bytes.drop(picture.width * picture.height).map { it.toInt() }

    @Test
    fun `an upright frame is copied as it is, with V before U`() {
        val picture = assertNotNull(nv21(4, 2, y, u, v, rotation = 0))

        assertEquals(4 to 2, picture.width to picture.height)
        assertEquals((0..7).toList(), luma(picture))
        assertEquals(listOf(20, 10, 21, 11), chroma(picture))
    }

    @Test
    fun `a frame the camera took sideways is turned upright`() {
        // Turned a quarter clockwise, the bottom row becomes the left column.
        val clockwise = assertNotNull(nv21(4, 2, y, u, v, rotation = 90))
        assertEquals(2 to 4, clockwise.width to clockwise.height)
        assertEquals(listOf(4, 0, 5, 1, 6, 2, 7, 3), luma(clockwise))

        val upsideDown = assertNotNull(nv21(4, 2, y, u, v, rotation = 180))
        assertEquals(listOf(7, 6, 5, 4, 3, 2, 1, 0), luma(upsideDown))

        val counterClockwise = assertNotNull(nv21(4, 2, y, u, v, rotation = 270))
        assertEquals(listOf(3, 7, 2, 6, 1, 5, 0, 4), luma(counterClockwise))
        assertEquals(listOf(21, 11, 20, 10), chroma(counterClockwise))
    }

    @Test
    fun `a large frame is scaled so its longest side fits, keeping both sides even`() {
        val width = 1920
        val height = 1080
        val luma = plane(ByteArray(width * height), stride = width)
        val half = plane(ByteArray(width / 2 * height / 2), stride = width / 2)

        val landscape = assertNotNull(nv21(width, height, luma, half, half, rotation = 0))
        assertEquals(1024 to 576, landscape.width to landscape.height)

        val portrait = assertNotNull(nv21(width, height, luma, half, half, rotation = 90))
        assertEquals(576 to 1024, portrait.width to portrait.height)
        assertEquals(576 * 1024 * 3 / 2, portrait.bytes.size)
    }

    @Test
    fun `a picture too small to encode is none`() {
        assertNull(nv21(1, 1, plane(byteArrayOf(0), 1), plane(byteArrayOf(0), 1), plane(byteArrayOf(0), 1), rotation = 0))
    }
}
