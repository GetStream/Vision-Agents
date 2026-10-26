package io.getstream.visionagents.android.demo

import android.content.Context
import android.net.Uri
import io.getstream.chat.android.ai.compose.conversation.AIMessage
import io.getstream.chat.android.ai.compose.conversation.AIMessageAttachment
import io.getstream.chat.android.ai.compose.conversation.AIMessageRating
import io.getstream.chat.android.ai.compose.conversation.AIMessageRole
import io.getstream.chat.android.ai.compose.parts.AIMessagePart
import io.getstream.chat.android.ai.compose.parts.AIReasoningPart
import io.getstream.chat.android.client.api.models.Pagination
import io.getstream.chat.android.client.api.models.QueryChannelRequest
import io.getstream.chat.android.client.api.models.WatchChannelRequest
import io.getstream.chat.android.client.channel.ChannelClient
import io.getstream.chat.android.client.events.ChatEvent
import io.getstream.chat.android.client.events.MessageDeletedEvent
import io.getstream.chat.android.client.events.MessageUpdatedEvent
import io.getstream.chat.android.client.events.NewMessageEvent
import io.getstream.chat.android.client.utils.observable.Disposable
import io.getstream.chat.android.models.Message
import io.getstream.result.Result
import io.getstream.visionagents.android.core.AgentSession
import io.getstream.visionagents.android.core.AgentsException
import io.getstream.visionagents.android.core.Conversation
import io.getstream.visionagents.android.core.LiveReasoning
import io.getstream.visionagents.android.core.chat
import io.getstream.visionagents.android.core.read
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * A written conversation as Stream's AI components want it: the channel it is kept in, the
 * whole of each reply's thinking, and what the agent is doing before it writes.
 *
 * The router writes everything here. Asking goes through the session, which records the
 * question in the channel and answers it there: the reply's text, and its steps (each round of
 * reasoning and each tool call) as attachments. The session's socket is what runs this phone's
 * tools and carries the person's answers to the ones that ask first.
 */
class AgentChat private constructor(
    val session: AgentSession,
    private val channel: ChannelClient,
    private val scope: CoroutineScope,
) {
    private val list = MutableStateFlow<List<Message>>(emptyList())
    private val asking = MutableStateFlow<String?>(null)
    private val thinking = MutableStateFlow(0)
    private val older = MutableStateFlow(false)
    private val rated = MutableStateFlow<Map<String, AIMessageRating>>(emptyMap())

    // The newest message when something was last asked here, until it fails or is stopped: what
    // arrives after it is the question and then its reply.
    private val askedAfter = MutableStateFlow<String?>(null)
    private var subscription: Disposable? = null

    /** The channel's messages, oldest first. */
    val messages: StateFlow<List<Message>> = list.asStateFlow()

    /** Whether the channel holds messages older than those loaded. */
    val hasOlderMessages: StateFlow<Boolean> = older.asStateFlow()

    /**
     * How the person rated replies on this phone, by message. Kept for the session only: the
     * router takes no ratings yet.
     */
    val ratings: StateFlow<Map<String, AIMessageRating>> = rated.asStateFlow()

    /** Each reply's thinking in full, from its live updates. A stored step only keeps its opening. */
    val reasoning: LiveReasoning = LiveReasoning()

    /**
     * The messages as Stream's AI components show them, each reply with the whole of its
     * thinking. A new list each time more thinking arrives, so the conversation redraws it.
     */
    val conversation: StateFlow<List<AgentMessage>> = combine(list, thinking) { messages, version ->
        messages.map { AgentMessage(it, reasoning, version) }
    }.stateIn(scope, SharingStarted.Eagerly, emptyList())

    /** Why asking failed, or null. */
    val failure: StateFlow<String?> = asking.asStateFlow()

    /** What the agent is doing while its reply shows nothing yet. */
    sealed interface Activity {
        /** Thinking about the question. */
        data object Thinking : Activity

        /** Running skills, such as "Working on order lookup". */
        data class Working(val text: String) : Activity
    }

    /**
     * What the agent is doing while its reply has nothing to show, or null.
     *
     * From the session's socket rather than Stream's AI indicator events: Stream Chat 7.15
     * cannot read those as the router sends them, with their state under `custom`, and drops
     * every one.
     */
    val activity: StateFlow<Activity?> = combine(session.conversation, list, askedAfter) { conversation, messages, asked ->
        when (val state = conversation.state) {
            is Conversation.State.Working ->
                Activity.Working("Working on " + state.skills.joinToString(" and ") { it.replace('_', ' ') })
            Conversation.State.Responding -> if (messages.replyHasBegun) null else Activity.Thinking
            // The session can pass through idle between the question and the first of its reply,
            // which would hide the activity for a moment.
            else -> if (messages.isAwaitingReply(asked)) Activity.Thinking else null
        }
    }.stateIn(scope, SharingStarted.Eagerly, null)

    /** Whether a reply is on its way, from the first thought to the last word. */
    val isAnswering: StateFlow<Boolean> = combine(session.conversation, list) { conversation, messages ->
        when (conversation.state) {
            Conversation.State.Responding, is Conversation.State.Working -> true
            else -> messages.lastOrNull { it.isAgentReply }?.isGenerating == true
        }
    }.stateIn(scope, SharingStarted.Eagerly, false)

    fun ask(text: String) {
        asking.value = null
        askedAfter.value = list.value.lastOrNull()?.id.orEmpty()
        scope.launch {
            try {
                session.responses.create(text)
            } catch (e: AgentsException) {
                asking.value = e.message
                askedAfter.value = null
            }
        }
    }

    /** Loads the messages before the oldest one loaded. */
    suspend fun loadOlderMessages() {
        val oldest = list.value.firstOrNull()?.id ?: return
        val request = QueryChannelRequest().withMessages(Pagination.LESS_THAN, oldest, PAGE_SIZE)
        when (val page = channel.query(request).await()) {
            is Result.Success -> {
                val loaded = page.value.messages
                list.update { messages -> (loaded.filterNot { new -> messages.any { it.id == new.id } } + messages).sorted() }
                older.value = loaded.size >= PAGE_SIZE
            }
            is Result.Failure -> asking.value = page.value.message
        }
    }

    /** Rates a reply, or takes the rating back with null. */
    fun rate(reply: String, rating: AIMessageRating?) {
        rated.update { if (rating == null) it - reply else it + (reply to rating) }
    }

    /** Stops the reply in flight. What was written so far is kept. */
    fun stop() {
        askedAfter.value = null
        scope.launch {
            try {
                session.interrupt()
            } catch (_: AgentsException) {
                // The reply finished first, which is what stopping it was for.
            }
        }
    }

    suspend fun close() {
        subscription?.dispose()
        session.close()
    }

    private suspend fun follow() {
        // The client's events rather than a list kept for us: each live update carries its own
        // window of the reply's thinking, and folding them into one would lose all but the last.
        subscription = channel.subscribe(::receive)
        when (val watched = channel.watch(WatchChannelRequest().withMessages(PAGE_SIZE)).await()) {
            is Result.Success -> {
                list.value = watched.value.messages.sorted()
                older.value = watched.value.messages.size >= PAGE_SIZE
            }
            is Result.Failure -> throw AgentsException.ChatFailed(watched.value.message)
        }
    }

    private fun receive(event: ChatEvent) {
        when (event) {
            is NewMessageEvent -> list.update { it.upsert(event.message) }
            is MessageUpdatedEvent -> {
                reasoning.read(event.message)
                thinking.update { it + 1 }
                list.update { it.upsert(event.message) }
            }
            is MessageDeletedEvent -> list.update { messages -> messages.filterNot { it.id == event.message.id } }
            else -> Unit
        }
    }

    companion object {
        private const val PAGE_SIZE = 30

        /** Opens the session's channel and starts following it. */
        suspend fun open(context: Context, session: AgentSession, scope: CoroutineScope): AgentChat {
            val chat = AgentChat(session, session.chat(context), scope)
            chat.follow()
            return chat
        }
    }
}

/** A reply the agent wrote, which Stream's AI components render. */
val Message.isAgentReply: Boolean get() = extraData["ai_generated"] == true

/** Whether the agent is still writing it. */
val Message.isGenerating: Boolean get() = extraData["generating"] == true

/** The reply's steps, in the order they happened. */
val Message.parts: List<AIMessagePart>
    // Stream lifts `name` out of an attachment's fields, and a tool call's step needs it back.
    get() = AIMessagePart.parts(attachments.map { it.type.orEmpty() to it.extraData + ("name" to it.name) })

/** Whether the reply to the latest question shows anything yet: a step or some text. */
private val List<Message>.replyHasBegun: Boolean
    get() = lastOrNull()?.takeIf { it.isAgentReply }?.let { it.text.isNotEmpty() || it.parts.isNotEmpty() } == true

/** Whether the question asked here is in the channel and its reply shows nothing yet. */
private fun List<Message>.isAwaitingReply(askedAfter: String?): Boolean {
    val last = lastOrNull() ?: return false
    if (askedAfter == null || last.id == askedAfter) return false
    return !last.isAgentReply || (last.isGenerating && !replyHasBegun)
}

private fun List<Message>.sorted(): List<Message> = sortedBy { it.createdAt ?: it.createdLocallyAt }

/**
 * A message of the conversation as Stream's AI components show it.
 *
 * [version] is how much live thinking had arrived when it was made, so a reply whose thinking
 * grew is a different message and redraws.
 */
data class AgentMessage(val message: Message, private val reasoning: LiveReasoning, private val version: Int) : AIMessage {
    override val id: String get() = message.id
    override val role: AIMessageRole get() = if (message.isAgentReply) AIMessageRole.Assistant else AIMessageRole.User
    override val text: String get() = message.text
    override val isGenerating: Boolean get() = message.isGenerating
    override val parts: List<AIMessagePart> = message.parts
    override val attachments: List<AIMessageAttachment>
        get() = message.attachments.mapNotNull { attachment ->
            when (attachment.type) {
                "image" -> (attachment.imageUrl ?: attachment.assetUrl)?.let {
                    AIMessageAttachment.image(Uri.parse(it), name = attachment.title)
                }
                "file" -> attachment.assetUrl?.let {
                    AIMessageAttachment.file(
                        Uri.parse(it),
                        name = attachment.title ?: attachment.name,
                        size = attachment.fileSize.toLong().takeIf { size -> size > 0 },
                        mimeType = attachment.mimeType,
                    )
                }
                else -> null
            }
        }

    // A step keeps only the opening of its thinking; the whole of it came with the live updates,
    // while the reply was being written.
    override fun reasoningText(step: AIReasoningPart): String? = reasoning[message.id, step.id]
}

private fun List<Message>.upsert(message: Message): List<Message> {
    val at = indexOfFirst { it.id == message.id }
    return if (at < 0) this + message else toMutableList().apply { set(at, message) }
}
