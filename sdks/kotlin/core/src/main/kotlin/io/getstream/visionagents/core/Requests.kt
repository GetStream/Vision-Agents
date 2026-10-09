package io.getstream.visionagents.core

import io.ktor.client.request.header
import io.ktor.client.request.request
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpMethod
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import java.io.IOException
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * The one JSON configuration, for requests and frames alike.
 *
 * `explicitNulls` off is what makes a field left unset stay off the wire, so the config or the
 * router decides it. The generated models carry no schema defaults (`generate.py` strips them),
 * so every field starts null and anything a caller set is sent, `false` included.
 */
internal val wire = Json {
    ignoreUnknownKeys = true
    explicitNulls = false
    encodeDefaults = true
}

/** How much of a body that is not the router's error is kept for the error it becomes. */
private const val MAX_ERROR_BODY = 512

/**
 * Sends one request and reads its answer.
 *
 * A request refused with a 401 is sent once more with a fresh token, because the likeliest
 * reason is that the one held expired. A failure to reach the router is [AgentsException.Transport];
 * an answer is [AgentsException.Http], whatever it says.
 */
internal suspend fun <T> Backend.send(
    method: HttpMethod,
    path: List<String>,
    answer: KSerializer<T>?,
    query: List<Pair<String, String>> = emptyList(),
    body: String? = null,
): T? {
    var response = attempt(method, path, query, body)
    if (response.status.value == 401 && expireToken()) {
        response = attempt(method, path, query, body)
    }

    val text = try {
        response.bodyAsText()
    } catch (e: IOException) {
        throw AgentsException.Transport(e)
    }
    if (!response.status.isSuccess()) {
        throw refusal(response.status.value, text) { response.headers[it] }
    }
    if (answer == null) return null
    return try {
        wire.decodeFromString(answer, text)
    } catch (e: SerializationException) {
        throw AgentsException.Unreadable("${path.joinToString("/")}: ${e.message}", e)
    } catch (e: IllegalArgumentException) {
        throw AgentsException.Unreadable("${path.joinToString("/")}: ${e.message}", e)
    }
}

internal suspend fun <T> Backend.get(
    path: List<String>,
    answer: KSerializer<T>,
    query: List<Pair<String, String>> = emptyList(),
): T = send(HttpMethod.Get, path, answer, query)!!

internal suspend fun <B, T> Backend.post(
    path: List<String>,
    request: KSerializer<B>,
    body: B,
    answer: KSerializer<T>,
): T = send(HttpMethod.Post, path, answer, body = wire.encodeToString(request, body))!!

private suspend fun Backend.attempt(
    method: HttpMethod,
    path: List<String>,
    query: List<Pair<String, String>>,
    body: String?,
): HttpResponse {
    val headers = headers()
    return try {
        http.request(url(*path.toTypedArray(), query = query)) {
            this.method = method
            headers.forEach { (name, value) -> header(name, value) }
            if (body != null) {
                contentType(ContentType.Application.Json)
                setBody(body)
            }
        }
    } catch (e: IOException) {
        throw AgentsException.Transport(e)
    }
}

/**
 * The error a refusal is, whether a request or a socket's handshake was refused.
 *
 * Every failure the router answers is `{"error": {"message", "type", "code", "doc_url"}}`. A
 * body that is not, such as a proxy's page, an empty one or an older router's `{"error": "..."}`,
 * keeps its text as the reason with no type or code, since a parse error in its place would hide
 * what went wrong. The request id is read either way, since a proxy may pass it on.
 */
internal fun refusal(status: Int, body: String, header: (String) -> String?): AgentsException.Http {
    val error = envelope(body)
    return AgentsException.Http(
        status = status,
        reason = error?.text("message")
            ?: body.trim().take(MAX_ERROR_BODY).ifEmpty { HttpStatusCode.fromValue(status).description },
        retryAfterSeconds = header("Retry-After")?.toLongOrNull(),
        type = error?.text("type"),
        code = error?.text("code"),
        docUrl = error?.text("doc_url"),
        requestId = header("X-Request-Id")?.ifEmpty { null },
    )
}

/**
 * The envelope's `error`, or null when the body is anything else.
 *
 * Read as JSON rather than through the generated `ErrorResponse`, so that a type this SDK never
 * heard of is kept as it came rather than folded into unknown.
 */
private fun envelope(body: String): JsonObject? {
    val parsed = try {
        wire.parseToJsonElement(body)
    } catch (_: SerializationException) {
        return null
    }
    val error = (parsed as? JsonObject)?.get("error") as? JsonObject ?: return null
    return error.takeIf { it.text("message") != null }
}

private fun JsonObject.text(key: String): String? =
    (get(key) as? JsonPrimitive)?.takeIf { it.isString }?.content?.ifEmpty { null }

/** The optional query parameters that were set, as the wire spells them. */
internal fun queryOf(vararg pairs: Pair<String, Any?>): List<Pair<String, String>> =
    pairs.mapNotNull { (name, value) ->
        when (value) {
            null -> null
            is String -> if (value.isEmpty()) null else name to value
            else -> name to value.toString()
        }
    }
