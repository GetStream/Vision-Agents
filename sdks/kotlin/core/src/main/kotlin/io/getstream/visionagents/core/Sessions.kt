package io.getstream.visionagents.core

import io.getstream.visionagents.core.generated.CreateSessionRequest
import io.getstream.visionagents.core.generated.ForkSessionRequest
import io.getstream.visionagents.core.generated.Session as SessionSchema
import io.getstream.visionagents.core.generated.SessionTool
import io.ktor.http.HttpMethod
import kotlinx.serialization.builtins.ListSerializer

/**
 * Conversations: opening one, finding the ones there were, and acting on one by id.
 *
 * What comes back are the router's rows rather than live handles, because reading a
 * conversation back is not the same as holding one and most of them are over. Holding one is
 * [VisionAgents.chat], [VisionAgents.voice] or [VisionAgents.attach].
 *
 * From [Agent.sessions] every call is about that agent: opening names it, and listing is
 * narrowed to it.
 */
public class Sessions internal constructor(
    private val backend: Backend,
    private val agent: String? = null,
) {
    /**
     * Opens a session without following it, for a caller building its own state layer.
     *
     * It returns once the router is holding the conversation. Without a [callId] it is held
     * in writing: nothing is joined, transcribed or spoken.
     */
    public suspend fun create(options: SessionOptions = SessionOptions(), callId: String? = null): Session {
        val request = CreateSessionRequest(
            callId = callId,
            text = if (callId == null) true else null,
            agent = (options.agent ?: agent)?.ifEmpty { null },
            configId = options.configId?.ifEmpty { null },
            title = options.title,
            description = options.description,
            project = options.project,
            custom = options.custom,
            incognito = options.incognito,
            persistConversation = options.persistConversation,
            conversationId = options.conversationId,
            modelOverwrites = options.modelOverwrites?.schema,
            instructions = options.instructions,
            greeting = options.greeting,
            llm = options.llm,
            stt = options.stt,
            tts = options.tts,
            voice = options.voice,
            tools = options.tools.ifEmpty { null }?.map { SessionTool(it.name, it.description, it.parameters) },
            tags = options.tags.ifEmpty { null },
        )
        val created = backend.post(
            listOf("v1", "agents", "sessions"),
            CreateSessionRequest.serializer(),
            request,
            SessionSchema.serializer(),
        )
        return Session.of(created)
    }

    /**
     * This caller's conversations, newest first, the ones that ended included.
     *
     * A page shorter than the limit asked for is the last one.
     */
    public suspend fun query(query: SessionQuery = SessionQuery()): List<Session> =
        backend.get(
            listOf("v1", "agents", "sessions"),
            ListSerializer(SessionSchema.serializer()),
            scoped(query).parameters(),
        ).map(Session::of)

    /**
     * Finds conversations by what they were called: their title, description, project and
     * agent name, best match first.
     *
     * What was said is not searched. An empty [text] is the same as [query], so a search box
     * nobody has typed in yet shows a person their conversations rather than nothing.
     */
    public suspend fun search(text: String, query: SessionQuery = SessionQuery()): List<Session> =
        backend.get(
            listOf("v1", "agents", "sessions", "search"),
            ListSerializer(SessionSchema.serializer()),
            queryOf("q" to text) + scoped(query).parameters(),
        ).map(Session::of)

    /**
     * One session. Somebody else's is reported as not found, so this is not a way to find out
     * whose an id is.
     */
    public suspend fun get(id: String): Session =
        Session.of(backend.get(listOf("v1", "agents", "sessions", id), SessionSchema.serializer()))

    /** Ends a session, which is how the agent leaves. */
    public suspend fun close(id: String) {
        backend.send<Unit>(HttpMethod.Delete, listOf("v1", "agents", "sessions", id), answer = null)
    }

    /**
     * Continues a conversation as a new session, leaving the parent as it was.
     *
     * Follow the fork the way any session is followed, with [VisionAgents.attach].
     */
    public suspend fun fork(id: String, options: ForkOptions = ForkOptions()): Session {
        val request = ForkSessionRequest(
            agent = options.agent?.ifEmpty { null },
            configId = options.configId?.ifEmpty { null },
            title = options.title,
            description = options.description,
            project = options.project,
            custom = options.custom,
            modelOverwrites = options.modelOverwrites?.schema,
            instructions = options.instructions,
            incognito = options.incognito,
            messages = if (options.withoutHistory) false else null,
            responseId = options.responseId?.ifEmpty { null },
            callId = options.callId,
        )
        val forked = backend.post(
            listOf("v1", "agents", "sessions", id, "fork"),
            ForkSessionRequest.serializer(),
            request,
            SessionSchema.serializer(),
        )
        return Session.of(forked)
    }

    /** A session's turns: asking, reading back, and rewinding. */
    public fun responses(id: String): Responses = Responses(backend, id)

    private fun scoped(query: SessionQuery): SessionQuery =
        if (agent == null || query.agent != null) query else query.copy(agent = agent)
}
