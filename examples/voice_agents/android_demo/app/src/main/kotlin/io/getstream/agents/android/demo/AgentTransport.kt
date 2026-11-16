package io.getstream.agents.android.demo

import android.content.Context
import android.net.Uri
import io.getstream.chat.android.ai.compose.conversation.AIAgentActivity
import io.getstream.chat.android.ai.compose.conversation.AIConversationFeed
import io.getstream.chat.android.ai.compose.conversation.AIConversationRequest
import io.getstream.chat.android.ai.compose.conversation.AIConversationTransport
import io.getstream.chat.android.ai.compose.conversation.AIMessage
import io.getstream.chat.android.ai.compose.conversation.AIMessageAttachment
import io.getstream.chat.android.ai.compose.conversation.AIMessageRole
import io.getstream.chat.android.ai.compose.parts.AIMessagePart
import io.getstream.chat.android.ai.compose.parts.AIReasoningPart
import io.getstream.chat.android.client.ChatClient
import io.getstream.chat.android.client.api.state.loadOlderMessages
import io.getstream.chat.android.client.channel.state.ChannelState
import io.getstream.chat.android.client.events.ChatEvent
import io.getstream.chat.android.client.events.MessageUpdatedEvent
import io.getstream.chat.android.client.utils.observable.Disposable
import io.getstream.chat.android.models.Message
import io.getstream.result.Result
import io.getstream.result.call.Call
import io.getstream.agents.android.core.AgentEvent
import io.getstream.agents.android.core.AgentSession
import io.getstream.agents.android.core.Conversation
import io.getstream.agents.android.core.LiveReasoning
import io.getstream.agents.android.core.channelState
import io.getstream.agents.android.core.chat
import io.getstream.agents.android.core.read
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.scan
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Connects a written conversation's
 * [AIConversationState][io.getstream.chat.android.ai.compose.conversation.AIConversationState] to
 * its session.
 *
 * The router writes everything to the session's channel. Asking goes through the session, which
 * records the question in the channel and answers it there: the reply's text, and its steps (each
 * round of reasoning and each tool call) as attachments. The session's socket is what runs this
 * phone's tools and carries the person's answers to the ones that ask first. The messages are the
 * channel's state, which Stream Chat keeps up to date, after a reconnect too.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class AgentTransport private constructor(
    val session: AgentSession,
    private val channel: StateFlow<ChannelState?>,
    private val scope: CoroutineScope,
) : AIConversationTransport<AgentMessage> {
    private val thinking = MutableStateFlow(0)
    private var subscription: Disposable? = null

    /** Each reply's thinking in full, from its live updates. A stored step only keeps its opening. */
    private val reasoning = LiveReasoning()

    override fun connect(feed: AIConversationFeed<AgentMessage>) {
        // The channel's messages, oldest first, each reply with the whole of its thinking. A new
        // list each time more thinking arrives, so the conversation redraws it.
        scope.launch {
            val messages = channel.flatMapLatest { it?.messages ?: flowOf(emptyList()) }
                .map { messages -> messages.filter { it.deletedAt == null } }
            combine(messages, thinking) { messages, version -> messages.map { AgentMessage(it, reasoning, version) } }
                .collect { if (feed.isConnected) feed.update(messages = it) }
        }
        scope.launch {
            channel.flatMapLatest { it?.endOfOlderMessages?.map { end -> !end } ?: flowOf(false) }
                .collect { if (feed.isConnected) feed.update(hasOlderMessages = it) }
        }
        // What the agent is doing, from the session's socket rather than Stream's AI indicator
        // events, which Stream Chat cannot read as the router sends them. A reply that fails ends
        // with nothing written to the channel; only the socket says so, as an error of the model's,
        // which holds until the next reply starts.
        scope.launch {
            val failed = session.events().scan(false) { failed, event ->
                when {
                    event.isFailedReply -> true
                    event.kind == AgentEvent.Kind.Responding -> false
                    else -> failed
                }
            }
            combine(session.conversation, failed) { conversation, failed ->
                if (failed) AIAgentActivity(AIAgentActivity.State.Error) else conversation.activity
            }.distinctUntilChanged().collect { if (feed.isConnected) feed.update(activity = it) }
        }
    }

    override suspend fun send(request: AIConversationRequest<AgentMessage>) {
        session.responses.create(request.text)
    }

    override suspend fun stop() {
        session.interrupt()
    }

    override suspend fun loadOlderMessages() {
        val cid = channel.value?.cid ?: return
        val page = ChatClient.instance().loadOlderMessages(cid, PAGE_SIZE).await()
        if (page is Result.Failure && !page.isCanceled) throw IllegalStateException(page.value.message)
    }

    suspend fun close() {
        subscription?.dispose()
        session.close()
    }

    private suspend fun follow(context: Context) {
        // The client's events rather than the channel's state for the thinking: each live update
        // carries its own window of it, and the state folds them into one, losing all but the last.
        subscription = session.chat(context).subscribe(::receive)
    }

    private fun receive(event: ChatEvent) {
        if (event !is MessageUpdatedEvent) return
        reasoning.read(event.message)
        thinking.update { it + 1 }
    }

    companion object {
        private const val PAGE_SIZE = 30

        /** Opens the session's channel and starts following it. */
        suspend fun open(context: Context, session: AgentSession, scope: CoroutineScope): AgentTransport {
            val transport = AgentTransport(session, session.channelState(context, scope, PAGE_SIZE), scope)
            transport.follow(context)
            return transport
        }
    }
}

/** What the agent says it is doing while the session works on a reply. */
private val Conversation.activity: AIAgentActivity?
    get() = when (val state = state) {
        Conversation.State.Responding -> AIAgentActivity(AIAgentActivity.State.Thinking)
        is Conversation.State.Working -> AIAgentActivity(
            AIAgentActivity.State.CheckingExternalSources,
            "Working on " + state.skills.joinToString(" and ") { it.replace('_', ' ') },
        )
        else -> null
    }

/**
 * Whether the call was cancelled rather than failed: Stream Chat cancels a load of older messages
 * when a newer one for the same channel takes its place, and the newer one loads them.
 */
private val Result.Failure.isCanceled: Boolean
    get() = value == (Call.callCanceledError<Unit>() as Result.Failure).value

/** Whether this is the router reporting that a reply failed, rather than some other part. */
private val AgentEvent.isFailedReply: Boolean
    get() = kind == AgentEvent.Kind.Error && string("context") in setOf("llm", "sts")

/** A reply the agent wrote, which Stream's AI components render. */
private val Message.isAgentReply: Boolean get() = extraData["ai_generated"] == true

/**
 * A message of the conversation as Stream's AI components show it.
 *
 * [version] is how much live thinking had arrived when it was made, so a reply whose thinking
 * grew is a different message and redraws.
 */
data class AgentMessage(val message: Message, private val reasoning: LiveReasoning, private val version: Int) : AIMessage {
    override val id: String = message.id
    override val role: AIMessageRole = if (message.isAgentReply) AIMessageRole.Assistant else AIMessageRole.User
    override val text: String = message.text
    override val isGenerating: Boolean = message.extraData["generating"] == true

    // Read on every update while a reply streams, so decoded once, when the message is made.
    // Stream lifts `name` out of an attachment's fields, and a tool call's step needs it back.
    override val parts: List<AIMessagePart> =
        AIMessagePart.parts(message.attachments.map { it.type.orEmpty() to it.extraData + ("name" to it.name) })

    override val attachments: List<AIMessageAttachment> = message.attachments.mapNotNull { attachment ->
        when (attachment.type) {
            "image" -> (attachment.imageUrl ?: attachment.assetUrl)?.let {
                AIMessageAttachment.image(Uri.parse(it), name = attachment.title ?: attachment.name)
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
