package io.getstream.visionagents.androidassistant

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.util.Base64
import io.getstream.visionagents.android.core.ImageSource
import java.io.ByteArrayOutputStream
import java.io.IOException
import kotlin.math.max
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Photos from the composer, as the router takes them.
 *
 * The composer hands over photos on the phone; a question takes an image by URL, and a phone has
 * nowhere public to put one. So each is shrunk and sent inline, as a data URI: a photo straight
 * off the camera is several megabytes, which is more than a model needs to read it.
 */
object Photos {
    /** The longest side a photo is sent at, in pixels. */
    private const val LONGEST_SIDE = 1280

    private const val JPEG_QUALITY = 70

    /** Each photo that reads as an image, ready to ask with. One that does not is left out. */
    suspend fun sources(context: Context, photos: Collection<Uri>): List<ImageSource> = withContext(Dispatchers.IO) {
        photos.mapNotNull { photo ->
            val image = decode(context, photo) ?: return@mapNotNull null
            val jpeg = ByteArrayOutputStream().use {
                image.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, it)
                it.toByteArray()
            }
            image.recycle()
            ImageSource("data:image/jpeg;base64," + Base64.encodeToString(jpeg, Base64.NO_WRAP))
        }
    }

    /**
     * The photo at most [LONGEST_SIDE] across. From Android 9 the decoder turns it the way the
     * camera held it; before, it is decoded as stored.
     */
    private fun decode(context: Context, photo: Uri): Bitmap? = try {
        if (Build.VERSION.SDK_INT >= 28) {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, photo)) { decoder, info, _ ->
                val scale = scale(info.size.width, info.size.height)
                decoder.setTargetSize((info.size.width * scale).roundToInt(), (info.size.height * scale).roundToInt())
                // A hardware bitmap cannot be compressed.
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        } else {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.contentResolver.openInputStream(photo)?.use { BitmapFactory.decodeStream(it, null, bounds) }
            // A power of two near the size wanted, then the rest exactly.
            val sample = Integer.highestOneBit(max(1, (1 / scale(bounds.outWidth, bounds.outHeight)).toInt()))
            val decoded = context.contentResolver.openInputStream(photo)?.use {
                BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
            }
            decoded?.let {
                val scale = scale(it.width, it.height)
                if (scale >= 1f) {
                    it
                } else {
                    Bitmap.createScaledBitmap(it, (it.width * scale).roundToInt(), (it.height * scale).roundToInt(), true)
                        .also { _ -> it.recycle() }
                }
            }
        }
    } catch (_: IOException) {
        null
    }

    private fun scale(width: Int, height: Int): Float =
        if (width <= 0 || height <= 0) 1f else minOf(1f, LONGEST_SIDE.toFloat() / max(width, height))
}
