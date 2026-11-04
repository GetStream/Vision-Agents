package io.getstream.visionagents.androidassistant

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * How the agent writes its replies: plain Markdown, or a generated UI in A2UI or OpenUI.
 *
 * The agent's instructions choose it, on the backend, so a change applies to the next new
 * conversation. This phone keeps the choice too, and shows generated UI unless it is Markdown.
 */
enum class ReplyFormat(val wire: String, val title: String) {
    Markdown("markdown", "Markdown"),
    A2UI("a2ui", "A2UI"),
    OpenUI("openui", "OpenUI");

    companion object {
        private const val PREFERENCES = "juno"
        private const val KEY = "reply_format"

        fun of(wire: String?): ReplyFormat? = entries.firstOrNull { it.wire == wire }

        /** The format chosen on this phone, Markdown until one is. */
        fun stored(context: Context): ReplyFormat =
            of(context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).getString(KEY, null)) ?: Markdown

        fun store(context: Context, format: ReplyFormat) {
            context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).edit().putString(KEY, format.wire).apply()
        }

        private val http = OkHttpClient()
        private val json = Json { ignoreUnknownKeys = true }

        private fun url() = "${Assistant.backendUrl.trimEnd('/')}/reply-format"

        /** The format the backend's agent writes in now, or null when it cannot say. */
        suspend fun fetch(): ReplyFormat? = withContext(Dispatchers.IO) {
            runCatching {
                http.newCall(Request.Builder().url(url()).get().build()).execute().use { response ->
                    if (!response.isSuccessful) return@use null
                    of(json.parseToJsonElement(response.body.string()).jsonObject["format"]?.jsonPrimitive?.contentOrNull)
                }
            }.getOrNull()
        }

        /** Tells the backend to have the agent write in [format]; false when it would not. */
        suspend fun save(format: ReplyFormat): Boolean = withContext(Dispatchers.IO) {
            runCatching {
                val body = """{"format":"${format.wire}"}""".toRequestBody("application/json".toMediaType())
                http.newCall(Request.Builder().url(url()).put(body).build()).execute().use { it.isSuccessful }
            }.getOrDefault(false)
        }
    }
}
