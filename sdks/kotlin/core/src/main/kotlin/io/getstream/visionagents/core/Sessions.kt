package io.getstream.visionagents.core

import io.getstream.visionagents.core.generated.CreateSessionRequest
import io.getstream.visionagents.core.generated.ForkSessionRequest
import io.getstream.visionagents.core.generated.Session as SessionSchema
import io.getstream.visionagents.core.generated.SessionPage
import io.getstream.visionagents.core.generated.SessionQuery as QuerySchema
import io.getstream.visionagents.core.generated.UpdateSessionRequest
import io.ktor.http.HttpMethod
import kotlinx.serialization.json.JsonObject

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
            id = options.id?.ifEmpty { null },
            callId = callId,
            text = if (callId == null) true else null,
            agent = (options.agent ?: agent)?.ifEmpty { null },
            configId = options.configId?.ifEmpty { null },
            title = options.title,
            description = options.description,
            projectId = options.projectId,
            custom = options.custom,
            incognito = options.incognito,
            conversationId = options.conversationId,
            modelOverwrites = options.modelOverwrites?.schema,
            instructions = options.instructions,
            greeting = options.greeting,
            llm = options.llm,
            stt = options.stt,
            tts = options.tts,
            voice = options.voice,
            tools = options.tools.ifEmpty { null }?.map { it.schema },
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
     * A page of this caller's conversations, most recently active first, the ones that ended
     * included. Pass its [Page.nextCursor] as [SessionQuery.cursor] for the next one.
     */
    public suspend fun query(query: SessionQuery = SessionQuery()): Page<Session> = page(scoped(query).schema(text = ""))

    /**
     * Finds conversations by what they were called: their title, description, project and
     * agent name, best match first. It pages the way [query] does.
     *
     * What was said is not searched. An empty [text] is the same as [query], so a search box
     * nobody has typed in yet shows a person their conversations rather than nothing.
     */
    public suspend fun search(text: String, query: SessionQuery = SessionQuery()): Page<Session> =
        page(scoped(query).schema(text))

    /**
     * One session. Somebody else's is reported as not found, so this is not a way to find out
     * whose an id is.
     */
    public suspend fun get(id: String): Session =
        Session.of(backend.get(listOf("v1", "agents", "sessions", id), SessionSchema.serializer()))

    /**
     * Renames a session or relabels it, running or ended, and returns it as it now is. Null
     * leaves a field as it is; [custom] replaces the labels whole, and an empty one clears them.
     */
    public suspend fun update(
        id: String,
        title: String? = null,
        description: String? = null,
        custom: JsonObject? = null,
    ): Session {
        val request = UpdateSessionRequest(title = title, description = description, custom = custom)
        val updated = backend.send(
            HttpMethod.Patch,
            listOf("v1", "agents", "sessions", id),
            SessionSchema.serializer(),
            body = wire.encodeToString(UpdateSessionRequest.serializer(), request),
        )!!
        return Session.of(updated)
    }

    /**
     * Stops a session, which is how the agent leaves. What it recorded and remembered is kept;
     * [delete] takes it away.
     */
    public suspend fun close(id: String) {
        backend.send<Unit>(HttpMethod.Post, listOf("v1", "agents", "sessions", id, "stop"), answer = null)
    }

    /**
     * Deletes a session, running or ended: it is stopped, and its turns and what it remembered
     * are deleted with it. The user's other memories are kept.
     */
    public suspend fun delete(id: String) {
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
            projectId = options.projectId,
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

    private suspend fun page(request: QuerySchema): Page<Session> {
        val page = backend.post(
            listOf("v1", "agents", "sessions", "query"),
            QuerySchema.serializer(),
            request,
            SessionPage.serializer(),
        )
        return Page(page.items.map(Session::of), page.hasMore, page.nextCursor?.ifEmpty { null })
    }
}
