package io.getstream.visionagents.android.demo

import android.content.Context
import io.getstream.chat.android.ai.compose.parts.AIMessagePart
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
    private var subscription: Disposable? = null

    /** The channel's messages, oldest first. */
    val messages: StateFlow<List<Message>> = list.asStateFlow()

    /** Each reply's thinking in full, from its live updates. A stored step only keeps its opening. */
    val reasoning: LiveReasoning = LiveReasoning()

    /** Goes up each time [reasoning] has more in it, which is what redraws a step's thinking. */
    val reasoningVersion: StateFlow<Int> = thinking.asStateFlow()

    /** Why asking failed, or null. */
    val failure: StateFlow<String?> = asking.asStateFlow()

    /**
     * What the agent is doing while its reply has nothing to show, such as "Thinking", or null.
     *
     * From the session's socket rather than Stream's AI indicator events: Stream Chat 7.15
     * cannot read those as the router sends them, with their state under `custom`, and drops
     * every one.
     */
    val indicator: StateFlow<String?> = combine(session.conversation, list) { conversation, messages ->
        when (val state = conversation.state) {
            is Conversation.State.Working ->
                "Working on " + state.skills.joinToString(" and ") { it.replace('_', ' ') }
            Conversation.State.Responding -> if (messages.replyHasBegun) null else "Thinking"
            else -> null
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
        scope.launch {
            try {
                session.responses.create(text)
            } catch (e: AgentsException) {
                asking.value = e.message
            }
        }
    }

    /** Stops the reply in flight. What was written so far is kept. */
    fun stop() {
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
        when (val watched = channel.watch().await()) {
            is Result.Success -> list.value = watched.value.messages.sortedBy { it.createdAt ?: it.createdLocallyAt }
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

private fun List<Message>.upsert(message: Message): List<Message> {
    val at = indexOfFirst { it.id == message.id }
    return if (at < 0) this + message else toMutableList().apply { set(at, message) }
}
