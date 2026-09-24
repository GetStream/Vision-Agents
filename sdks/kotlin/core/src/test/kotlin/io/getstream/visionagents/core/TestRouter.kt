package io.getstream.visionagents.core

import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.install
import io.ktor.server.cio.CIO
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import io.ktor.server.request.httpMethod
import io.ktor.server.request.path
import io.ktor.server.request.receiveText
import io.ktor.server.response.header
import io.ktor.server.response.respondText
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import io.ktor.server.websocket.WebSockets
import io.ktor.server.websocket.webSocket
import io.ktor.websocket.CloseReason
import io.ktor.websocket.Frame
import io.ktor.websocket.close
import io.ktor.websocket.readText
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout

/** A request as it reached the far end. */
data class Arrived(
    val method: String,
    val path: String,
    val query: Map<String, List<String>>,
    val headers: Map<String, String>,
    val body: String,
) {
    fun header(name: String): String? = headers.entries.firstOrNull { it.key.equals(name, ignoreCase = true) }?.value
}

/** What the router answers one request with. */
data class Reply(val status: Int = 200, val body: String = "", val headers: Map<String, String> = emptyMap())

/** Something the test tells the socket to do. */
sealed interface Script {
    data class Send(val frame: String) : Script

    data class Close(val code: Short, val reason: String) : Script
}

/**
 * A real HTTP server with a real WebSocket upgrade on 127.0.0.1, standing in for the router.
 *
 * It records what arrived and answers with what the test scripted, so what a test checks is the
 * request this SDK actually put on the wire: headers, encoded paths and query parameters.
 */
class TestRouter : AutoCloseable {
    val requests = CopyOnWriteArrayList<Arrived>()
    val handshakes = CopyOnWriteArrayList<Arrived>()

    /** Frames the SDK sent over the socket, in order. */
    val sent = Channel<String>(Channel.UNLIMITED)

    /** What the socket does next, in order. */
    val script = Channel<Script>(Channel.UNLIMITED)

    @Volatile var answer: (Arrived) -> Reply = { Reply(404, """{"error":"nothing scripted"}""") }

    private val server: EmbeddedServer<*, *> = embeddedServer(CIO, port = 0, host = "127.0.0.1") {
        install(WebSockets)
        routing {
            webSocket("/v1/agents/sessions/{id}/events") {
                handshakes += Arrived(
                    "GET",
                    call.request.path(),
                    call.request.queryParameters.entries().associate { it.key to it.value },
                    call.request.headers.entries().associate { it.key to it.value.joinToString(",") },
                    "",
                )
                val reading = launch {
                    for (frame in incoming) {
                        if (frame is Frame.Text) sent.send(frame.readText())
                    }
                }
                for (step in script) {
                    when (step) {
                        is Script.Send -> outgoing.send(Frame.Text(step.frame))
                        is Script.Close -> {
                            close(CloseReason(step.code, step.reason))
                            break
                        }
                    }
                }
                reading.join()
            }
            route("{...}") {
                handle {
                    val arrived = Arrived(
                        call.request.httpMethod.value,
                        call.request.path(),
                        call.request.queryParameters.entries().associate { it.key to it.value },
                        call.request.headers.entries().associate { it.key to it.value.joinToString(",") },
                        call.receiveText(),
                    )
                    requests += arrived
                    val reply = answer(arrived)
                    reply.headers.forEach { (name, value) -> call.response.header(name, value) }
                    call.respondText(reply.body, ContentType.Application.Json, HttpStatusCode.fromValue(reply.status))
                }
            }
        }
    }

    val url: String

    init {
        server.start(wait = false)
        url = runBlocking { "http://127.0.0.1:${server.engine.resolvedConnectors().first().port}" }
    }

    /** The next frame the SDK sent, waiting for it rather than for a guessed length of time. */
    suspend fun nextSent(): String = withTimeout(5.seconds) { sent.receive() }

    override fun close() {
        script.close()
        server.stop(0, 0)
    }
}

/** Waits until the condition holds, so a test waits for what happens rather than for a fixed time. */
suspend fun until(seconds: Int = 5, done: suspend () -> Boolean) {
    withTimeout(seconds.seconds) {
        while (!done()) kotlinx.coroutines.delay(10)
    }
}
