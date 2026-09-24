package io.getstream.visionagents.core

import io.getstream.visionagents.core.generated.Error
import io.ktor.client.request.header
import io.ktor.client.request.request
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpMethod
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import java.io.IOException
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json

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

/** How much of a body that is not JSON is kept for the error it becomes. */
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
        throw AgentsException.Http(
            status = response.status.value,
            reason = reason(text),
            retryAfterSeconds = response.headers["Retry-After"]?.toLongOrNull(),
        )
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

/** What the router said went wrong, which is `{"error": "..."}` from every handler. */
private fun reason(text: String): String {
    val decoded = try {
        wire.decodeFromString(Error.serializer(), text).error
    } catch (_: SerializationException) {
        null
    } catch (_: IllegalArgumentException) {
        null
    }
    return decoded ?: text.take(MAX_ERROR_BODY)
}

/** The optional query parameters that were set, as the wire spells them. */
internal fun queryOf(vararg pairs: Pair<String, Any?>): List<Pair<String, String>> =
    pairs.mapNotNull { (name, value) ->
        when (value) {
            null -> null
            is String -> if (value.isEmpty()) null else name to value
            else -> name to value.toString()
        }
    }
