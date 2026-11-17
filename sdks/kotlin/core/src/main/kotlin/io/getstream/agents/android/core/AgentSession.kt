package io.getstream.agents.android.core

import io.getstream.agents.android.core.generated.AddSessionToolsRequest
import io.ktor.http.HttpMethod
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.CoroutineContext
import kotlin.io.encoding.Base64
import kotlin.time.Clock
import kotlin.time.Instant
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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.addJsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray

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
    /** Where this conversation is held, and who is holding it. */
    public val backend: Backend,
    session: Session,
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
    /** The tools whose calls are answered here, by name. */
    private val tools = ConcurrentHashMap(tools.associateBy { it.name })
    /**
     * The tools whose adding was never answered, so the router may or may not offer them. Read
     * and written holding [changing].
     */
    private val unsettled = mutableSetOf<String>()
    /** Held through a change of tools, so that one runs at a time, in the order they were asked for. */
    private val changing = Mutex()
    private val running = ConcurrentHashMap<String, Job>()
    private val started = AtomicBoolean(false)

    private val state = MutableStateFlow(Conversation())
    private val connected = MutableStateFlow(false)
    private val stoppedBy = MutableStateFlow<AgentsException?>(null)
    private val held = MutableStateFlow<List<ToolApprovalRequest>>(emptyList())
    private val broadcast = MutableSharedFlow<AgentEvent>(
        extraBufferCapacity = EVENT_BUFFER,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    /** What the call's camera showed lately while it is on, which answers the router's request for frames. */
    @Volatile internal var cameraFrames: CameraFrames? = null

    /** The session the router opened, as [update] last left it. */
    @Volatile public var session: Session = session
        private set

    /** What the router holds this session by, which is what addresses it and its socket. */
    public val id: String get() = session.id

    /** The transcript and what the agent is doing. */
    public val conversation: StateFlow<Conversation> = state.asStateFlow()

    /** Whether the socket is still carrying the conversation. */
    public val isConnected: StateFlow<Boolean> = connected.asStateFlow()

    /** Why the socket stopped, or null. A conversation that ended normally has none. */
    public val failure: StateFlow<AgentsException?> = stoppedBy.asStateFlow()

    /** Calls of tools that ask first, waiting for the person to answer with [decide], oldest first. */
    public val approvals: StateFlow<List<ToolApprovalRequest>> = held.asStateFlow()

    /**
     * The session's turns: asking, reading back and rewinding. What is asked here shows in
     * [conversation] at once, and the reply streams into it.
     */
    public val responses: Responses = Responses(
        backend,
        session.id,
        asked = { text -> state.update { it.said(text) } },
    )

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

    /**
     * Speaks this without going through the model. [interrupt] stops what the agent is saying
     * first, rather than waiting for it to finish.
     */
    public suspend fun say(text: String, interrupt: Boolean = false) {
        if (interrupt) socket.send(Command.Interrupt())
        socket.send(Command.Say(text))
    }

    /** Abandons the reply in flight, or stops the request named. */
    public suspend fun interrupt(requestId: String = "") {
        socket.send(Command.Interrupt(requestId))
    }

    /**
     * Answers a call waiting in [approvals].
     *
     * Allowed, the tool runs and its result goes back as it would have. Declined, it never runs:
     * the model is told the person declined, and the call's step shows [summary]. A call that
     * is not waiting is left alone, so answering twice does nothing. If the answer cannot be
     * sent, the call waits again.
     */
    public suspend fun decide(toolCallId: String, allowed: Boolean, summary: String = "") {
        var index = -1
        var request: ToolApprovalRequest? = null
        held.update { waiting ->
            index = waiting.indexOfFirst { it.id == toolCallId }
            request = waiting.getOrNull(index)
            if (index < 0) waiting else waiting.filterIndexed { i, _ -> i != index }
        }
        val decided = request ?: return
        val call = decided.call
        try {
            socket.send(Command.ToolApproval(call.id, allowed, summary, call.requestId, call.turnId))
        } catch (e: AgentsException) {
            held.update { it.toMutableList().apply { add(minOf(index, size), decided) } }
            throw e
        }
        if (allowed) {
            run(call, decided.tool)
        } else {
            socket.send(
                Command.ToolResult(call.id, error = "The person declined this call.", requestId = call.requestId, turnId = call.turnId),
            )
        }
    }

    /**
     * Declines the sign-in that a call of one of the agent's connectors waits for, whose step in
     * the conversation is `awaiting_authorization`. The router answers the call itself, so it
     * runs nothing and the agent is told the person did not connect. A call that is not waiting
     * is left alone.
     */
    public suspend fun declineLogin(toolCallId: String) {
        socket.send(Command.ToolApproval(toolCallId, allowed = false))
    }

    /** Puts the agent on the session's call, `agent:<session id>`. See [Sessions.startVoice]. */
    public suspend fun startVoice(): Session = Sessions(backend).startVoice(session.id).also { session = it }

    /** Takes the agent off the call and carries on in writing. See [Sessions.stopVoice]. */
    public suspend fun stopVoice(): Session = Sessions(backend).stopVoice(session.id).also { session = it }

    /** Renames this conversation or relabels it, and returns it as it now is. See [Sessions.update]. */
    public suspend fun update(
        title: String? = null,
        description: String? = null,
        custom: JsonObject? = null,
        pinned: Boolean? = null,
    ): Session = Sessions(backend).update(session.id, title, description, custom, pinned).also { session = it }

    /** Pins this conversation, and returns it with its [Session.pinnedAt]. See [Sessions.pin]. */
    public suspend fun pin(): Session = update(pinned = true)

    /** Unpins this conversation. See [Sessions.unpin]. */
    public suspend fun unpin(): Session = update(pinned = false)

    /**
     * Continues this conversation as a new session, leaving this one as it was.
     *
     * Follow the fork with [StreamAgents.attach].
     */
    public suspend fun fork(options: ForkOptions = ForkOptions()): Session =
        Sessions(backend).fork(session.id, options)

    /**
     * Gives the agent more of your tools while the conversation runs, such as the ones a card
     * shown in it brings. Their calls are answered here, as those of the tools the session was
     * opened with are, and they are this session's alone.
     *
     * The router waits for the reply in progress, and every call it made, to finish, and the
     * model is offered the tools from the next turn. So never call this from inside one of this
     * session's tools: it waits for that call. A tool named as one this session already has is
     * refused with a 409: to change one, [removeTool] it first, so that no call of it runs half
     * under each. Only the kind of caller that opened the session may change its tools (403
     * otherwise), a name the agent has of its own, such as a connector's tool, is a 409, and a
     * speech-to-speech model whose tools are fixed when it opens is a 400.
     *
     * A request that gets no answer may or may not have added them, so their calls are still
     * answered here until adding or removing the name again settles it.
     */
    public suspend fun addTools(tools: List<AgentTool>) {
        if (tools.isEmpty()) return
        changing.withLock {
            // A name whose adding was never answered may or may not be the router's: it is
            // withdrawn first, so the router and this session agree on which tool it is.
            for (tool in tools) {
                if (tool.name in unsettled) withdraw(tool.name)
            }
            // Answered from before the router says they are added: the next turn can call one
            // while that answer is still on its way here. A name already answered here keeps
            // its handler, since the router refuses to add it again.
            val added = tools.filter { this.tools.putIfAbsent(it.name, it) == null }
            try {
                backend.send<Unit>(
                    HttpMethod.Post,
                    listOf("v1", "agents", "sessions", id, "tools"),
                    answer = null,
                    body = wire.encodeToString(AddSessionToolsRequest.serializer(), AddSessionToolsRequest(tools.map { it.schema })),
                )
            } catch (e: AgentsException.Http) {
                // Refused, so the router offers none of them.
                added.forEach { this.tools.remove(it.name) }
                throw e
            } catch (e: Throwable) {
                // No answer, cancelled waiting for one included: whether the router offers them
                // is unknown, so they stay answered until a later add or remove of the name
                // settles it.
                unsettled += added.map { it.name }
                throw e
            }
        }
    }

    /**
     * Withdraws one of your tools, one the session was opened with or one [addTools] gave it.
     * The router waits for the reply in progress, and every call it made, to finish, as
     * [addTools] does, so once this returns no call of the tool can still arrive and its handler
     * goes; never call it from inside one of this session's tools. A name this session has no
     * tool by is a 404. A name withdrawn can be added again, with another handler.
     */
    public suspend fun removeTool(name: String) {
        changing.withLock { withdraw(name) }
    }

    /**
     * Stops the session and closes the socket. Safe to call more than once. What it recorded and
     * remembered is kept; [delete] takes it away.
     */
    public suspend fun close() {
        try {
            socket.send(Command.Close)
        } catch (_: AgentsException) {
            // Already gone, which is what closing wanted.
        }
        end()
    }

    /** Deletes this conversation and closes the socket. See [Sessions.delete]. */
    public suspend fun delete() {
        Sessions(backend).delete(session.id)
        end()
    }

    /**
     * Asks the router to withdraw a tool, and lets its handler go once it has. For a name whose
     * adding was never answered, the router having no tool by it settles it as well.
     */
    private suspend fun withdraw(name: String) {
        try {
            backend.send<Unit>(HttpMethod.Delete, listOf("v1", "agents", "sessions", id, "tools", name), answer = null)
        } catch (e: AgentsException.Http) {
            if (e.status != 404 || name !in unsettled) throw e
        }
        tools.remove(name)
        unsettled.remove(name)
    }

    private suspend fun end() {
        socket.close()
        scope.cancel()
        connected.value = false
        held.value = emptyList()
        state.update { it.copy(state = Conversation.State.Ended) }
    }

    private fun apply(event: AgentEvent) {
        state.update { it.reduce(event) }
        when (event.kind) {
            AgentEvent.Kind.ToolCall -> event.toolCall?.let(::answer)
            AgentEvent.Kind.ToolCancel -> {
                val id = event.string("id")
                running.remove(id)?.cancel()
                held.update { waiting -> waiting.filterNot { it.id == id } }
            }
            else -> Unit
        }
        broadcast.tryEmit(event)
    }

    private fun stopped(error: AgentsException?) {
        stoppedBy.value = error
        connected.value = false
        held.value = emptyList()
        state.update { it.copy(state = Conversation.State.Ended) }
        running.values.forEach { it.cancel() }
    }

    /** Runs a tool the model asked for, or holds the call for the person when the tool asks first. */
    private fun answer(call: AgentEvent.ToolCall) {
        // Session events go to every watcher, and another one may own this call. Answering a
        // tool this device never declared would resolve it before its owner could. The request
        // for frames is this device's only while its camera is on; otherwise a video worker in
        // the backend may own it.
        if (call.name == CameraFrames.TOOL_NAME) {
            cameraFrames?.let { return show(it, call) }
        }
        val tool = tools[call.name] ?: return
        val approval = tool.approval
        if (approval != null) {
            held.update { it + ToolApprovalRequest(call, approval, tool) }
        } else {
            run(call, tool)
        }
    }

    /**
     * Answers the router's request for frames with the ones the camera showed at the moment it
     * names: what the frames are, then the frames, oldest first. Nothing recent enough is said as
     * the answer rather than as a failure, as the router's own video workers say it, so the agent
     * can ask the person to show it again.
     */
    private fun show(frames: CameraFrames, call: AgentEvent.ToolCall) {
        scope.launch {
            val arguments = call.argumentValues
            val at = (arguments["at_ms"] as? JsonPrimitive)?.doubleOrNull
                ?.let { Instant.fromEpochMilliseconds(it.toLong()) }
                ?: Clock.System.now()
            val limit = (arguments["limit"] as? JsonPrimitive)?.doubleOrNull?.toInt() ?: 1
            val answer = try {
                val selected = frames.select(at, limit)
                val words = buildJsonObject {
                    put("clock", "device_unix_ms")
                    putJsonArray("frames") {
                        selected.forEach { frame ->
                            addJsonObject {
                                put("captured_at_ms", frame.capturedAt.toEpochMilliseconds())
                                put("frame_id", "camera-${frame.id}")
                            }
                        }
                    }
                    put("source", "camera")
                }
                Command.ToolResult(
                    call.id,
                    output = words.toString(),
                    requestId = call.requestId,
                    turnId = call.turnId,
                    images = selected.map { ImageSource("data:image/jpeg;base64," + Base64.encode(it.jpeg)) },
                )
            } catch (e: CameraFrames.NoFrames) {
                Command.ToolResult(call.id, output = e.message.orEmpty(), requestId = call.requestId, turnId = call.turnId)
            }
            try {
                socket.send(answer)
            } catch (_: AgentsException) {
                // The socket went; there is nobody left to show.
            }
        }
    }

    /**
     * Runs a tool and sends back what it returned.
     *
     * A coroutine of its own, so a slow tool holds up neither the transcript nor the read loop,
     * which is also what delivers `tool_cancel`.
     */
    private fun run(call: AgentEvent.ToolCall, tool: AgentTool) {
        val job = scope.launch(start = CoroutineStart.LAZY) {
            val result = try {
                val output = tool.run(call.argumentValues)
                Command.ToolResult(
                    call.id,
                    output = output.text,
                    requestId = call.requestId,
                    turnId = call.turnId,
                    images = output.images,
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                // The model can only say something useful about a tool that did not work if
                // it is told that it did not work.
                Command.ToolResult(call.id, error = e.message ?: e.toString(), requestId = call.requestId, turnId = call.turnId)
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

/** A call of one of your tools, waiting for the person to allow it. */
public class ToolApprovalRequest internal constructor(
    public val call: AgentEvent.ToolCall,
    /** The question the tool asks. */
    public val approval: AgentTool.Approval,
    internal val tool: AgentTool,
) {
    public val id: String get() = call.id

    /** The model's own words for why it wants the call, from the argument the approval names, or the empty string. */
    public val reason: String
        get() = approval.reasonArgument?.let { (call.argumentValues[it] as? JsonPrimitive)?.contentOrNull }.orEmpty()
}
