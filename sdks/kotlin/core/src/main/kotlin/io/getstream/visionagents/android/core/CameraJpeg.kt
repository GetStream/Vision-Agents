package io.getstream.visionagents.android.core

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min
import org.webrtc.VideoFrame

/** One picture in the NV21 layout `YuvImage` encodes: the luma plane, then V and U interleaved. */
internal class Nv21(val bytes: ByteArray, val width: Int, val height: Int)

/** One plane of an I420 picture, read from index 0 of [data]. */
internal class Plane(val data: ByteBuffer, val stride: Int)

/**
 * An I420 picture turned upright by [rotation] (clockwise degrees, as a `VideoFrame` says), scaled
 * down so its longest side is at most [longest], and laid out as NV21. Both sides are even, which
 * NV21 needs; null for a picture too small to have any.
 *
 * Each output pixel is the nearest source pixel, which is enough for a model to look at and costs
 * one read per pixel.
 */
internal fun nv21(
    width: Int,
    height: Int,
    y: Plane,
    u: Plane,
    v: Plane,
    rotation: Int,
    longest: Int = CameraFrames.LONGEST_SIDE,
): Nv21? {
    val turned = rotation.mod(360)
    val sideways = turned == 90 || turned == 270
    val uprightWidth = if (sideways) height else width
    val uprightHeight = if (sideways) width else height
    val side = max(uprightWidth, uprightHeight)
    fun scaled(length: Int) = if (side <= longest) length else (length.toLong() * longest / side).toInt()
    val outWidth = scaled(uprightWidth) and 1.inv()
    val outHeight = scaled(uprightHeight) and 1.inv()
    if (outWidth < 2 || outHeight < 2) return null

    // Where each output column and row falls in the upright picture, before turning it back into
    // the source's own coordinates.
    val columns = IntArray(outWidth) { (it.toLong() * uprightWidth / outWidth).toInt() }
    val rows = IntArray(outHeight) { (it.toLong() * uprightHeight / outHeight).toInt() }
    fun sourceX(x: Int, y: Int) = when (turned) {
        90 -> y
        180 -> width - 1 - x
        270 -> width - 1 - y
        else -> x
    }
    fun sourceY(x: Int, y: Int) = when (turned) {
        90 -> height - 1 - x
        180 -> height - 1 - y
        270 -> x
        else -> y
    }

    val out = ByteArray(outWidth * outHeight * 3 / 2)
    var i = 0
    for (row in 0 until outHeight) {
        val uy = rows[row]
        for (column in 0 until outWidth) {
            val ux = columns[column]
            out[i++] = y.data.get(sourceY(ux, uy) * y.stride + sourceX(ux, uy))
        }
    }
    for (row in 0 until outHeight step 2) {
        val uy = rows[row]
        for (column in 0 until outWidth step 2) {
            val ux = columns[column]
            val sx = sourceX(ux, uy) / 2
            val sy = sourceY(ux, uy) / 2
            out[i++] = v.data.get(sy * v.stride + sx)
            out[i++] = u.data.get(sy * u.stride + sx)
        }
    }
    return Nv21(out, outWidth, outHeight)
}

internal fun Nv21.jpeg(): ByteArray {
    val out = ByteArrayOutputStream()
    YuvImage(bytes, ImageFormat.NV21, width, height, null)
        .compressToJpeg(Rect(0, 0, width, height), CameraFrames.QUALITY, out)
    return out.toByteArray()
}

/** The picture upright and at most [CameraFrames.LONGEST_SIDE] on its longest side, as a JPEG. */
internal fun VideoFrame.I420Buffer.jpeg(rotation: Int): ByteArray? =
    nv21(width, height, Plane(dataY, strideY), Plane(dataU, strideU), Plane(dataV, strideV), rotation)?.jpeg()

/**
 * A copy at most [CameraFrames.LONGEST_SIDE] on its longest side, taken while the frame's bitmap
 * is still the filter's to read.
 */
internal fun Bitmap.scaledCopy(): Bitmap {
    val scale = min(1f, CameraFrames.LONGEST_SIDE.toFloat() / max(width, height))
    return if (scale < 1f) {
        Bitmap.createScaledBitmap(this, max(1, (width * scale).toInt()), max(1, (height * scale).toInt()), true)
    } else {
        copy(config ?: Bitmap.Config.ARGB_8888, false)
    }
}

internal fun Bitmap.jpeg(): ByteArray {
    val out = ByteArrayOutputStream()
    compress(Bitmap.CompressFormat.JPEG, CameraFrames.QUALITY, out)
    return out.toByteArray()
}
