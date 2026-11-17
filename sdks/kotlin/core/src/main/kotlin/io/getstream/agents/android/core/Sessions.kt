package io.getstream.agents.android.core

import io.getstream.agents.android.core.generated.CreateSessionRequest
import io.getstream.agents.android.core.generated.Equals
import io.getstream.agents.android.core.generated.ForkSessionRequest
import io.getstream.agents.android.core.generated.Session as SessionSchema
import io.getstream.agents.android.core.generated.SessionFilter as FilterSchema
import io.getstream.agents.android.core.generated.SessionPage
import io.getstream.agents.android.core.generated.SessionQuery as QuerySchema
import io.getstream.agents.android.core.generated.TextMatch
import io.getstream.agents.android.core.generated.UpdateSessionRequest
import io.ktor.http.HttpMethod
import kotlinx.serialization.json.JsonObject

/**
 * Conversations: opening one, finding the ones there were, and acting on one by id.
 *
 * What comes back are the router's rows rather than live handles, because reading a
 * conversation back is not the same as holding one and most of them are over. Holding one is
 * [StreamAgents.chat], [StreamAgents.voice] or [StreamAgents.attach].
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
     * It returns once the router is holding the conversation. Without
     * [SessionOptions.startVoice] it is held in writing: nothing is joined, transcribed or spoken.
     */
    public suspend fun create(options: SessionOptions = SessionOptions()): Session {
        val request = CreateSessionRequest(
            id = options.id?.ifEmpty { null },
            startVoice = if (options.startVoice) true else null,
            agent = (options.agent ?: agent)?.ifEmpty { null },
            configId = options.configId?.ifEmpty { null },
            title = options.title,
            description = options.description,
            projectId = options.projectId,
            custom = options.custom,
            incognito = options.incognito,
            modelOverwrites = options.modelOverwrites?.schema,
            greeting = options.greeting?.schema,
            llm = options.llm,
            stt = options.stt,
            tts = options.tts,
            voice = options.voice,
            tools = options.tools.ifEmpty { null }?.map { it.schema },
            tags = options.tags.ifEmpty { null },
            returnUrl = options.returnUrl?.ifEmpty { null },
            memory = options.memory?.schema(backend.user?.id),
            timezone = options.timezone?.ifEmpty { null },
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
     * A page of this caller's conversations, the ones that ended included.
     *
     * [filter] narrows it and [sort] orders it: nothing is the most recent message first,
     * `listOf(SessionSort(SessionSortField.Pinned), SessionSort(SessionSortField.LastMessageAt))`
     * puts the pinned ones on top. [limit] is up to 200, and 25 when left out. Pass a page's
     * [Page.nextCursor] as [cursor], with the same filter and sort, for the next one.
     */
    public suspend fun query(
        filter: SessionFilter? = null,
        sort: List<SessionSort> = emptyList(),
        limit: Int? = null,
        cursor: String? = null,
    ): Page<Session> = page(sent(filter, text = ""), sort, limit, cursor)

    /**
     * Finds conversations by what they were called: their title, description, project and
     * agent name, best match first. It narrows and pages the way [query] does, and takes no
     * sort. A [filter] that already holds a [SessionFilter.Text] is refused.
     *
     * What was said is not searched. An empty [text] is the same as [query], so a search box
     * nobody has typed in yet shows a person their conversations rather than nothing.
     */
    public suspend fun search(
        text: String,
        filter: SessionFilter? = null,
        limit: Int? = null,
        cursor: String? = null,
    ): Page<Session> = page(sent(filter, text), emptyList(), limit, cursor)

    /**
     * One session. Somebody else's is reported as not found, so this is not a way to find out
     * whose an id is.
     */
    public suspend fun get(id: String): Session =
        Session.of(backend.get(listOf("v1", "agents", "sessions", id), SessionSchema.serializer()))

    /**
     * Renames a session or relabels it, running or ended, and returns it as it now is. Null
     * leaves a field as it is; [custom] replaces the labels whole, and an empty one clears them.
     * [pinned] pins it or unpins it, which moves it nowhere in the most-recent-message order.
     */
    public suspend fun update(
        id: String,
        title: String? = null,
        description: String? = null,
        custom: JsonObject? = null,
        pinned: Boolean? = null,
    ): Session {
        val request = UpdateSessionRequest(title = title, description = description, custom = custom, pinned = pinned)
        val updated = backend.send(
            HttpMethod.Patch,
            listOf("v1", "agents", "sessions", id),
            SessionSchema.serializer(),
            body = wire.encodeToString(UpdateSessionRequest.serializer(), request),
        )!!
        return Session.of(updated)
    }

    /**
     * Pins a session, running or ended, and returns it with its [Session.pinnedAt]. Pinning one
     * that is already pinned keeps the time it was first pinned.
     */
    public suspend fun pin(id: String): Session = update(id, pinned = true)

    /** Unpins a session and returns it. Unpinning one that is not pinned does nothing. */
    public suspend fun unpin(id: String): Session = update(id, pinned = false)

    /**
     * Has the agent join a session's call, `agent:<session id>`, and carry the conversation on
     * there, and returns the session on it. Starting voice that is already on does nothing.
     */
    public suspend fun startVoice(id: String): Session = Session.of(
        backend.send(HttpMethod.Post, listOf("v1", "agents", "sessions", id, "voice"), SessionSchema.serializer())!!,
    )

    /**
     * Takes the agent off a session's call and carries the conversation on in writing. Stopping
     * voice that is off does nothing.
     */
    public suspend fun stopVoice(id: String): Session = Session.of(
        backend.send(HttpMethod.Delete, listOf("v1", "agents", "sessions", id, "voice"), SessionSchema.serializer())!!,
    )

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
     * Continues a conversation as a new session held in writing, leaving the parent as it was.
     *
     * Follow the fork the way any session is followed, with [StreamAgents.attach].
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
            incognito = options.incognito,
            messages = if (options.withoutHistory) false else null,
            responseId = options.responseId?.ifEmpty { null },
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

    /**
     * The filter that is sent: the caller's, with the words of a search and this handle's agent
     * set beside its other top-level keys. It is not wrapped in an `$and`, because a text search
     * has to stay at the top of the filter.
     */
    private fun sent(filter: SessionFilter?, text: String): FilterSchema {
        var schema = filter?.schema() ?: FilterSchema()
        if (text.isNotEmpty()) {
            if (schema.text != null || schema.dollarAnd.orEmpty().any { it.text != null }) {
                throw AgentsException.Configuration("a search sets the filter's text, so the filter cannot hold one too")
            }
            schema = schema.copy(text = TextMatch(text))
        }
        return if (agent.isNullOrEmpty()) schema else schema.copy(agent = Equals(agent))
    }

    private suspend fun page(filter: FilterSchema, sort: List<SessionSort>, limit: Int?, cursor: String?): Page<Session> {
        val request = QuerySchema(
            filter = filter,
            sort = sort.ifEmpty { null }?.map { it.schema() },
            limit = limit?.toLong(),
            cursor = cursor?.ifEmpty { null },
        )
        val page = backend.post(
            listOf("v1", "agents", "sessions", "query"),
            QuerySchema.serializer(),
            request,
            SessionPage.serializer(),
        )
        return Page(page.items.map(Session::of), page.hasMore, page.nextCursor?.ifEmpty { null })
    }
}
