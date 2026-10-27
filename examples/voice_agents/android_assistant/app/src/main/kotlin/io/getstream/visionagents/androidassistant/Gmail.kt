package io.getstream.visionagents.androidassistant

import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * Sending from the person's Gmail.
 *
 * The person connects Gmail in the chat, the first time the agent reaches for it, and the router
 * keeps the Google credential, which the app never sees. Google's Gmail MCP server writes drafts
 * but cannot send, so a send goes through the app's backend, which posts to Gmail's REST API
 * through the router's proxy for the same connection.
 */
object Gmail {
    private val http = OkHttpClient()
    private val json = Json { ignoreUnknownKeys = true }

    /** Sends a plain-text email from the person's Gmail, through the backend, and returns its id. */
    suspend fun send(to: String, subject: String, body: String): String = withContext(Dispatchers.IO) {
        val asked = buildJsonObject {
            put("user_id", Assistant.user.id)
            put("to", to)
            put("subject", subject)
            put("body", body)
        }
        val request = Request.Builder()
            .url("${Assistant.backendUrl.trimEnd('/')}/gmail/send")
            .post(asked.toString().toRequestBody("application/json".toMediaType()))
            .build()
        val response = try {
            http.newCall(request).execute()
        } catch (_: IOException) {
            throw ToolFailure("The backend could not reach Gmail.")
        }
        response.use {
            val said = it.body.string().trim()
            // The backend says why in words, such as Gmail not being connected yet, which the
            // model passes on or acts on.
            if (!it.isSuccessful) throw ToolFailure(said.ifEmpty { "The backend could not reach Gmail." })
            json.parseToJsonElement(said).jsonObject["id"]?.jsonPrimitive?.contentOrNull.orEmpty()
        }
    }
}
