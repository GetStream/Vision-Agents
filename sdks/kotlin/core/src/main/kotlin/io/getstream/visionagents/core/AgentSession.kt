package io.getstream.visionagents.core

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.CoroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** How many events a slow [AgentSession.events] collector may fall behind before the oldest go. */
private const val EVENT_BUFFER = 256

/**
 * A live conversation, as `StateFlow`s a screen can collect.
 *
 * The state is one immutable [Conversation], folded from the socket's events by a single read
 * loop, so a view never sees a transcript and a state that disagree. Collect [conversation]
 * with `collectAsStateWithLifecycle()` in Compose, or build your own view of it: everything a
 * screen needs is in it.
 *
 * The session owns a scope, and [close] ends it. It is not tied to a screen: an `AndroidX`
 * `ViewModel` that holds one and closes it in `onCleared` is the usual home.
 */
public class AgentSession internal constructor(
    private val backend: Backend,
    /** The session the router opened. */
    public val session: Session,
    tools: List<AgentTool>,
    context: CoroutineContext = Dispatchers.Default,
) {
    private val scope = CoroutineScope(SupervisorJob() + context)
    private val socket = SessionSocket(
        backend,
        session.id,
        // Decisions arrive several times a second and are for somebody watching a call, not
        // for an app holding one.
        listOf("decisions" to "false"),
    )
    private val tools = tools.associateBy { it.name }
    private val running = ConcurrentHashMap<String, Job>()
    private val started = AtomicBoolean(false)

    private val state = MutableStateFlow(Conversation())
    private val connected = MutableStateFlow(false)
    private val stoppedBy = MutableStateFlow<AgentsException?>(null)
    private val broadcast = MutableSharedFlow<AgentEvent>(
        extraBufferCapacity = EVENT_BUFFER,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    /** What the router holds this session by, which is what addresses it and its socket. */
    public val id: String get() = session.id

    /** The transcript and what the agent is doing. */
    public val conversation: StateFlow<Conversation> = state.asStateFlow()

    /** Whether the socket is still carrying the conversation. */
    public val isConnected: StateFlow<Boolean> = connected.asStateFlow()

    /** Why the socket stopped, or null. A conversation that ended normally has none. */
    public val failure: StateFlow<AgentsException?> = stoppedBy.asStateFlow()

    /** The session's turns as the router wrote them down, for reading back and rewinding. */
    public val responses: Responses = Responses(backend, session.id)

    /**
     * Every event as it arrives, for a caller building on more than [conversation] holds.
     *
     * Each collector gets its own copy from the moment it starts collecting; nothing is
     * replayed. A collector more than 256 events behind loses the oldest rather than stalling
     * the socket. Losing one never loses a tool call, since those are answered here whether
     * or not anybody is collecting.
     */
    public fun events(): Flow<AgentEvent> = broadcast.asSharedFlow()

    /**
     * Opens the socket and starts following the conversation. Doing this twice does nothing.
     *
     * The shorthands that return an [AgentSession] have already called it.
     */
    public suspend fun start() {
        if (!started.compareAndSet(false, true)) return
        val events = try {
            socket.open()
        } catch (e: AgentsException) {
            started.set(false)
            throw e
        }
        connected.value = true
        scope.launch {
            try {
                events.collect(::apply)
                stopped(null)
            } catch (e: AgentsException) {
                stopped(e)
            }
        }
    }

    /** Says this to the agent, as though it had been heard. */
    public suspend fun send(text: String, images: List<ImageSource> = emptyList()) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return
        state.update { it.said(trimmed) }
        socket.send(Command.Respond(trimmed, images))
    }

    /** Speaks this without going through the model. */
    public suspend fun say(text: String) {
        socket.send(Command.Say(text))
    }

    /** Abandons the reply in flight, or stops the durable command named. */
    public suspend fun interrupt(commandId: String = "") {
        socket.send(Command.Interrupt(commandId))
    }

    /** Replaces the system prompt, from the next turn on. */
    public suspend fun setInstructions(instructions: String) {
        socket.send(Command.Instructions(instructions))
    }

    /**
     * Continues this conversation as a new session, leaving this one as it was.
     *
     * Follow the fork with [VisionAgents.attach].
     */
    public suspend fun fork(options: ForkOptions = ForkOptions()): Session =
        Sessions(backend).fork(session.id, options)

    /** Ends the session and closes the socket. Safe to call more than once. */
    public suspend fun close() {
        try {
            socket.send(Command.Close)
        } catch (_: AgentsException) {
            // Already gone, which is what closing wanted.
        }
        socket.close()
        scope.cancel()
        connected.value = false
        state.update { it.copy(state = Conversation.State.Ended) }
    }

    private fun apply(event: AgentEvent) {
        state.update { it.reduce(event) }
        when (event.kind) {
            AgentEvent.Kind.ToolCall -> event.toolCall?.let(::answer)
            AgentEvent.Kind.ToolCancel -> running.remove(event.string("id"))?.cancel()
            else -> Unit
        }
        broadcast.tryEmit(event)
    }

    private fun stopped(error: AgentsException?) {
        stoppedBy.value = error
        connected.value = false
        state.update { it.copy(state = Conversation.State.Ended) }
        running.values.forEach { it.cancel() }
    }

    /**
     * Runs a tool the model asked for and sends back what it returned.
     *
     * A coroutine of its own, so a slow tool holds up neither the transcript nor the read loop,
     * which is also what delivers `tool_cancel`.
     */
    private fun answer(call: AgentEvent.ToolCall) {
        // Session events go to every watcher, and another one may own this call. Answering a
        // tool this device never declared would resolve it before its owner could.
        val tool = tools[call.name] ?: return
        val job = scope.launch(start = CoroutineStart.LAZY) {
            val result = try {
                Command.ToolResult(call.id, output = tool.run(call.argumentValues), commandId = call.commandId, turnId = call.turnId)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                // The model can only say something useful about a tool that did not work if
                // it is told that it did not work.
                Command.ToolResult(call.id, error = e.message ?: e.toString(), commandId = call.commandId, turnId = call.turnId)
            }
            try {
                socket.send(result)
            } catch (_: AgentsException) {
                // The socket went while the tool ran; there is nobody left to tell.
            }
        }
        running[call.id] = job
        job.invokeOnCompletion { running.remove(call.id, job) }
        job.start()
    }
}
