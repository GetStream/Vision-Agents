package io.getstream.visionagents.core

import io.ktor.client.plugins.websocket.DefaultClientWebSocketSession
import io.ktor.client.plugins.websocket.webSocketSession
import io.ktor.client.request.header
import io.ktor.websocket.CloseReason
import io.ktor.websocket.Frame
import io.ktor.websocket.close
import io.ktor.websocket.readBytes
import io.ktor.websocket.readText
import java.io.IOException
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.channels.ClosedSendChannelException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * The socket carrying one conversation.
 *
 * Hand-written because OpenAPI stops at the upgrade. OkHttp answers the router's pings itself,
 * so there is no keepalive here, and no JSON `"ping"` either.
 *
 * **No automatic reconnection, on purpose.** `respond` and `tool_result` are not idempotent,
 * and the protocol has no sequence number to resume from, so replaying after a reconnect would
 * duplicate turns and tool results. A socket that drops ends the flow with
 * [AgentsException.SocketClosed] and the caller decides.
 */
public class SessionSocket internal constructor(
    private val backend: Backend,
    private val sessionId: String,
    private val query: List<Pair<String, String>>,
) {
    private val opened = AtomicBoolean(false)
    private val closing = AtomicBoolean(false)
    @Volatile private var connection: DefaultClientWebSocketSession? = null

    /**
     * Connects, and returns the session's events.
     *
     * Returned rather than exposed as a property so that it cannot be read twice: one reader
     * per connection, because two readers take half the frames each. Nothing is missed between
     * connecting and collecting, since the router starts watching before the upgrade and the
     * frames wait in the connection until they are read.
     *
     * The flow completes when the session ends or [close] is called, and throws
     * [AgentsException.SocketClosed] when the socket goes away for any other reason.
     */
    public suspend fun open(): Flow<AgentEvent> {
        if (!opened.compareAndSet(false, true)) {
            throw AgentsException.Configuration("a session socket is opened once")
        }
        val headers = backend.headers()
        val url = backend.socketUrl("v1", "agents", "sessions", sessionId, "events", query = query)
        val ws = try {
            backend.http.webSocketSession(url) {
                headers.forEach { (name, value) -> header(name, value) }
            }
        } catch (e: IOException) {
            throw AgentsException.Transport(e)
        }
        connection = ws

        return flow {
            try {
                for (frame in ws.incoming) {
                    val text = when (frame) {
                        is Frame.Text -> frame.readText()
                        is Frame.Binary -> frame.readBytes().decodeToString()
                        else -> continue
                    }
                    AgentEvent.decode(text)?.let { emit(it) }
                }
            } catch (e: IOException) {
                if (closing.get()) return@flow
                // 1006 is what RFC 6455 reserves for a connection that ended with no close frame.
                throw AgentsException.SocketClosed(1006, e.message.orEmpty())
            }
            val reason = ws.closeReason.await()
            if (!closing.get() && reason != null && reason.code != CloseReason.Codes.NORMAL.code) {
                throw AgentsException.SocketClosed(reason.code.toInt(), reason.message)
            }
        }
    }

    /** Sends one command. Fails if the socket is not open. */
    public suspend fun send(command: Command) {
        val ws = connection ?: throw AgentsException.SocketClosed(0, "the socket is not open")
        try {
            ws.send(Frame.Text(command.encode()))
        } catch (_: ClosedSendChannelException) {
            throw AgentsException.SocketClosed(0, "the socket is closed")
        }
    }

    /** Closes the socket with a normal closure. Safe to call more than once. */
    public suspend fun close() {
        if (!closing.compareAndSet(false, true)) return
        connection?.close(CloseReason(CloseReason.Codes.NORMAL, ""))
    }
}
