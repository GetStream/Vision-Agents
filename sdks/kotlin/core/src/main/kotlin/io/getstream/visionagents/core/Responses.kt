package io.getstream.visionagents.core

import io.getstream.visionagents.core.generated.AgentResponse as ResponseSchema
import io.getstream.visionagents.core.generated.AgentResponseItemPage
import io.getstream.visionagents.core.generated.AgentResponsePage
import io.getstream.visionagents.core.generated.CreateResponseRequest
import io.getstream.visionagents.core.generated.RewindSessionRequest
import io.ktor.http.HttpMethod
import java.util.UUID
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/** How many items are read per request while unwinding. */
private const val ITEM_PAGE = 200

/**
 * A session's turns, as the router wrote them down.
 *
 * Read rather than watched: this is the same whether the conversation is still going or ended
 * last week. Deltas are not here — a hundred fragments of one sentence are the sentence — so a
 * caller who wants to watch words arrive reads [AgentSession.conversation].
 */
public class Responses internal constructor(
    private val backend: Backend,
    public val sessionId: String,
    /** Shows what was asked in a live transcript, before the router has started on it. */
    private val asked: (String) -> Unit = {},
) {
    /**
     * Asks the agent something and names the turn it answers as.
     *
     * It returns as soon as the agent has started answering rather than when it has finished,
     * so the result is a handle on an answer in progress: [items] with its id reads what has
     * been written down so far. From [AgentSession.responses] the question shows in
     * [AgentSession.conversation] at once. A text question carries a fresh request id, so a
     * retry of the same request starts no second turn.
     */
    public suspend fun create(text: String, images: List<ImageSource> = emptyList()): AgentResponse {
        val request = CreateResponseRequest(
            text = text,
            images = images.ifEmpty { null }?.map { it.schema },
            // The router refuses a request id on a question showing the agent something.
            requestId = if (images.isEmpty()) UUID.randomUUID().toString() else null,
        )
        asked(text)
        val created = backend.post(
            path(),
            CreateResponseRequest.serializer(),
            request,
            ResponseSchema.serializer(),
        )
        return AgentResponse.of(created)
    }

    /**
     * A page of the turns so far, oldest first. Pass its [Page.nextCursor] as [cursor] for the
     * next one.
     *
     * A session that records nothing has none, and one rewound has none after the response it
     * went back to.
     */
    public suspend fun list(limit: Int? = null, cursor: String? = null): Page<AgentResponse> {
        val page = backend.get(path(), AgentResponsePage.serializer(), queryOf("limit" to limit, "cursor" to cursor))
        return Page(page.items.map(AgentResponse::of), page.hasMore, page.nextCursor?.ifEmpty { null })
    }

    /**
     * One page of what the agent did, turn by turn, in the order it happened.
     *
     * Every turn in the session, or only [responseId]'s. Nothing comes back for an incognito
     * session, which has none to return.
     */
    public suspend fun items(responseId: String? = null, limit: Int? = null, cursor: String? = null): Page<ResponseItem> {
        val page = backend.get(
            path("items"),
            AgentResponseItemPage.serializer(),
            queryOf("response_id" to responseId, "limit" to limit, "cursor" to cursor),
        )
        return Page(page.items.map(ResponseItem::of), page.hasMore, page.nextCursor?.ifEmpty { null })
    }

    /**
     * Every item, oldest first, a page at a time.
     *
     * Paging is inside rather than outside because a conversation's length is not something the
     * caller chose: a turn or a thousand are read the same way.
     */
    public fun unwind(responseId: String? = null): Flow<ResponseItem> = flow {
        var cursor: String? = null
        while (true) {
            val page = items(responseId, ITEM_PAGE, cursor)
            page.items.forEach { emit(it) }
            cursor = page.nextCursor
            if (!page.hasMore || cursor == null) return@flow
        }
    }

    /**
     * Goes back to a response and carries on from there, as though nothing after it was said.
     *
     * The model forgets the later turns and they drop out of [list] and [items]. A transcript
     * an [AgentSession] is showing still has them, so read it back after this. A conversation
     * kept in Stream Chat cannot be rewound, because the channel would still hold the later
     * turns: fork it at the response instead.
     */
    public suspend fun rewind(responseId: String) {
        if (responseId.isEmpty()) {
            throw AgentsException.Configuration("a response that was never recorded cannot be rewound to")
        }
        backend.send<Unit>(
            HttpMethod.Post,
            listOf("v1", "agents", "sessions", sessionId, "rewind"),
            answer = null,
            body = wire.encodeToString(RewindSessionRequest.serializer(), RewindSessionRequest(responseId)),
        )
    }

    /** Goes back to this response. */
    public suspend fun rewind(to: AgentResponse): Unit = rewind(to.id)

    /** Goes back to the response this item belongs to, which is what a transcript renders. */
    public suspend fun rewind(to: ResponseItem): Unit = rewind(to.responseId)

    private fun path(vararg rest: String) = listOf("v1", "agents", "sessions", sessionId, "responses") + rest
}
