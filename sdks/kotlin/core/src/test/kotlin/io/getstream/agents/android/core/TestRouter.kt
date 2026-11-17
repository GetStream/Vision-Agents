package io.getstream.agents.android.core

import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCallPipeline
import io.ktor.server.application.call
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
import java.io.IOException
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread
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

/**
 * What the router answers one request with. [hangUp] hangs up once the request is handled and
 * before any of its answer is out, as a network that drops the answer does: the router acted on
 * it and the SDK never hears so. OkHttp sends a request again on a connection that drops that
 * early unless it was told the request may go only once.
 */
data class Reply(
    val status: Int = 200,
    val body: String = "",
    val headers: Map<String, String> = emptyMap(),
    val hangUp: Boolean = false,
)

/** A failure as the router answers every one. */
fun envelope(type: String, code: String, message: String) =
    """{"error":{"message":"$message","type":"$type","code":"$code",""" +
        """"doc_url":"https://getstream.io/agents/docs/api/errors/#$code"}}"""

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

    @Volatile var answer: (Arrived) -> Reply = { Reply(404, envelope("not_found", "not_found", "nothing scripted")) }

    /** What the socket's handshake is answered with in place of the upgrade, or null to upgrade. */
    @Volatile var refuseHandshake: Reply? = null

    /**
     * Where the SDK connects. Each connection is passed on to the server as it is, so the server
     * can hang up on one before writing anything to it, which it cannot do itself.
     */
    private val front = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))

    /** The SDK's end of each connection passed on, by the port the server sees it come from. */
    private val passed = ConcurrentHashMap<Int, Socket>()

    private val server: EmbeddedServer<*, *> = embeddedServer(CIO, port = 0, host = "127.0.0.1") {
        install(WebSockets)
        intercept(ApplicationCallPipeline.Plugins) {
            if (!call.request.path().endsWith("/events")) return@intercept
            // Recorded before the upgrade is answered: the SDK counts as connected once it is.
            handshakes += Arrived(
                "GET",
                call.request.path(),
                call.request.queryParameters.entries().associate { it.key to it.value },
                call.request.headers.entries().associate { it.key to it.value.joinToString(",") },
                "",
            )
            val refusal = refuseHandshake
            if (refusal != null) {
                refusal.headers.forEach { (name, value) -> call.response.header(name, value) }
                call.respondText(refusal.body, ContentType.Application.Json, HttpStatusCode.fromValue(refusal.status))
                finish()
            }
        }
        routing {
            webSocket("/v1/agents/sessions/{id}/events") {
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
                    if (reply.hangUp) {
                        passed.remove(call.request.local.remotePort)?.close()
                        return@handle
                    }
                    reply.headers.forEach { (name, value) -> call.response.header(name, value) }
                    call.respondText(reply.body, ContentType.Application.Json, HttpStatusCode.fromValue(reply.status))
                }
            }
        }
    }

    val url: String

    init {
        server.start(wait = false)
        val port = runBlocking { server.engine.resolvedConnectors().first().port }
        thread(isDaemon = true) {
            while (true) {
                val outer = try {
                    front.accept()
                } catch (_: IOException) {
                    break
                }
                val inner = Socket("127.0.0.1", port)
                passed[inner.localPort] = outer
                pipe(outer, inner)
                pipe(inner, outer)
            }
        }
        url = "http://127.0.0.1:${front.localPort}"
    }

    /** The next frame the SDK sent, waiting for it rather than for a guessed length of time. */
    suspend fun nextSent(): String = withTimeout(5.seconds) { sent.receive() }

    override fun close() {
        script.close()
        front.close()
        passed.values.forEach { it.close() }
        server.stop(0, 0)
    }

    /** Copies one way of a connection until either end goes, and then closes both. */
    private fun pipe(from: Socket, to: Socket) = thread(isDaemon = true) {
        try {
            from.getInputStream().copyTo(to.getOutputStream())
        } catch (_: IOException) {
            // One end went, which ends the connection.
        } finally {
            from.close()
            to.close()
        }
    }
}

/** Waits until the condition holds, so a test waits for what happens rather than for a fixed time. */
suspend fun until(seconds: Int = 5, done: suspend () -> Boolean) {
    withTimeout(seconds.seconds) {
        while (!done()) kotlinx.coroutines.delay(10)
    }
}
