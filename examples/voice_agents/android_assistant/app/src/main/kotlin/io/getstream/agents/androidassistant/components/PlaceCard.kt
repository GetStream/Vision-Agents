@file:Suppress("MagicNumber") // The card's layout.

package io.getstream.agents.androidassistant.components

import android.graphics.BitmapFactory
import android.util.LruCache
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponent
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponentDefinition
import io.getstream.chat.android.ai.openui.rendering.OpenUITheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response

/** A place's card, which asks about the place when tapped. */
internal fun placeCard(definition: AIUIComponentDefinition) = AIUIComponent(definition) { arguments, context ->
    val title = arguments.string("title") ?: return@AIUIComponent
    PlaceCard(title, arguments.string("note")) {
        context.emit("ask_about", label = "Tell me more about $title", payload = mapOf("title" to title))
    }
}

/** One place of a plan, with its photo and summary from Wikipedia. Tapping it asks about it. */
@Composable
private fun PlaceCard(title: String, note: String?, onAsk: () -> Unit) {
    val theme = OpenUITheme.current
    var lookup by remember(title) { mutableStateOf(Wikipedia.cached(title)) }
    LaunchedEffect(title) {
        if (lookup == null) lookup = Wikipedia.summary(title)
    }
    val summary = lookup?.summary
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(theme.radius.large))
            .background(theme.colors.subtleFill)
            .clickable(onClickLabel = "Ask Juno about it", onClick = onAsk)
            .padding(10.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Crossfade(
            targetState = summary?.thumbnail,
            animationSpec = tween(300),
            modifier = Modifier
                .size(88.dp)
                .clip(RoundedCornerShape(theme.radius.medium)),
            label = "photo",
        ) { photo ->
            if (photo != null) {
                Image(photo, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
            } else {
                Box(Modifier.fillMaxSize().background(theme.colors.sunkenSurface))
            }
        }
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(summary?.title ?: title, style = theme.typography.heading, color = theme.colors.textPrimary)
            if (note != null) Text(note, style = theme.typography.bodySmall, color = theme.colors.accent)
            summary?.extract?.let { extract ->
                Text(
                    extract,
                    style = theme.typography.caption,
                    color = theme.colors.textSecondary,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

/** A place's summary on Wikipedia, the last few dozen kept while the app runs. */
private object Wikipedia {
    class Summary(val title: String, val extract: String?, val thumbnail: ImageBitmap?)

    /** What Wikipedia answered for a title: its summary, or null when it has no such article. */
    class Lookup(val summary: Summary?)

    private const val SUMMARIES = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    // Wikipedia asks every client to say who it is.
    private const val USER_AGENT = "AndroidAssistant/1.0 (https://github.com/GetStream/Vision-Agents)"

    private val http = OkHttpClient()
    private val json = Json { ignoreUnknownKeys = true }
    private val summaries = LruCache<String, Lookup>(48)

    fun cached(title: String): Lookup? = summaries.get(title)

    /**
     * What Wikipedia has for [title], with its photo, or null when it can't be asked right now,
     * which is not remembered, so the card asks again next time.
     */
    suspend fun summary(title: String): Lookup? = summaries.get(title) ?: withContext(Dispatchers.IO) {
        runCatching { load(title) }.getOrNull()?.also { summaries.put(title, it) }
    }

    private fun load(title: String): Lookup? {
        val url = SUMMARIES.toHttpUrl().newBuilder().addPathSegment(title.replace(' ', '_')).build()
        val page = call(url.toString()).use { response ->
            when {
                response.code == 404 -> return Lookup(null)
                !response.isSuccessful -> return null
                else -> json.parseToJsonElement(response.body.string()).jsonObject
            }
        }
        val photo = page["thumbnail"]?.jsonObject?.get("source")?.jsonPrimitive?.contentOrNull
            ?.let(::get)
            ?.let { BitmapFactory.decodeByteArray(it, 0, it.size)?.asImageBitmap() }
        return Lookup(
            Summary(
                title = page["title"]?.jsonPrimitive?.contentOrNull ?: title,
                extract = page["extract"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotEmpty() },
                thumbnail = photo,
            ),
        )
    }

    private fun get(url: String): ByteArray? =
        call(url).use { response -> if (response.isSuccessful) response.body.bytes() else null }

    private fun call(url: String): Response =
        http.newCall(Request.Builder().url(url).header("User-Agent", USER_AGENT).build()).execute()
}
