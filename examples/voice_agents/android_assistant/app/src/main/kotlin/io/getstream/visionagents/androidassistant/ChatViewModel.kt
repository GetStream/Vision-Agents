package io.getstream.visionagents.androidassistant

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.getstream.chat.android.ai.compose.conversation.AIMessageRating
import io.getstream.chat.android.ai.compose.conversation.AIPendingMessage
import io.getstream.visionagents.android.core.AgentsException
import io.getstream.visionagents.android.core.SessionOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * The conversation on screen, and switching to another.
 *
 * A new conversation has no session until its first message, so opening the app or tapping New
 * chat leaves nothing in the list. Opening a session takes seconds, though, so a new
 * conversation's is opened as its first message is written rather than once it is sent, and
 * deleted again if the person leaves without sending it. A conversation from the list is
 * followed again by its id, and one that ended carries on when it is next asked something.
 *
 * It lives as long as the activity, so turning the phone keeps the conversation, and picking
 * another one closes the last.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val shown = MutableStateFlow<String?>(null)
    private val screenId = MutableStateFlow(0)
    private val opened = MutableStateFlow<AgentChat?>(null)
    private val failed = MutableStateFlow<String?>(null)
    private val asked = MutableStateFlow<Asked?>(null)

    /** The conversation open, or null for a new one that has not been written to. */
    val selected: StateFlow<String?> = shown.asStateFlow()

    /**
     * Which conversation the screen was opened for. It changes only when the person picks one,
     * so a new conversation getting its session keeps the screen it was started on.
     */
    val screen: StateFlow<Int> = screenId.asStateFlow()

    /** The conversation's chat, once its session is open. */
    val chat: StateFlow<AgentChat?> = opened.asStateFlow()

    /**
     * What the person just asked, shown at once until Stream's copy of their message arrives and
     * takes its place: opening a conversation and writing the message take seconds together.
     */
    val pending: StateFlow<AIPendingMessage?> = asked.map { it?.message }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** The messages as Stream's AI components show them. */
    val messages: StateFlow<List<AgentMessage>> = opened.flatMapLatest { it?.conversation ?: flowOf(emptyList()) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    /** What the agent is doing while its reply shows nothing yet. */
    val activity: StateFlow<AgentChat.Activity?> = opened.flatMapLatest { it?.activity ?: flowOf(null) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** Whether a reply is on its way, from the first thought to the last word. */
    val isAnswering: StateFlow<Boolean> = opened.flatMapLatest { it?.isAnswering ?: flowOf(false) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    /** Whether the channel holds messages older than those loaded. */
    val hasOlderMessages: StateFlow<Boolean> = opened.flatMapLatest { it?.hasOlderMessages ?: flowOf(false) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    /** How the person rated replies on this phone, by message. */
    val ratings: StateFlow<Map<String, AIMessageRating>> = opened.flatMapLatest { it?.ratings ?: flowOf(emptyMap()) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, emptyMap())

    /** Why opening or asking failed, or why the session stopped, or null. */
    val failure: StateFlow<String?> = opened.flatMapLatest { chat ->
        if (chat == null) {
            failed
        } else {
            combine(failed, chat.failure, chat.session.failure) { opening, asking, session ->
                opening ?: asking ?: session?.message
            }
        }
    }.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** The tools the agent may call, which run on this phone. */
    private val tools = DeviceTools(application).all

    // What belongs to the conversation on screen, and goes when another is picked.
    private var screenScope = newScreenScope()
    private var openedFor: String? = null
    private var opening: Deferred<AgentChat?>? = null

    // Whether the list has been told about this new conversation.
    private var isAnnounced = false

    // A new conversation's session, opened while the person wrote their first message and not
    // used yet. Leaving without using it deletes it, so it is not left in the list.
    private var unused: String? = null

    /** Shows a conversation from the list, or a new one for null, closing the one on screen. */
    fun show(sessionId: String?) {
        leave()
        screenScope = newScreenScope()
        openedFor = sessionId
        opening = null
        isAnnounced = false
        unused = null
        opened.value = null
        failed.value = null
        asked.value = null
        shown.value = sessionId
        screenId.value++
        if (sessionId != null) screenScope.launch { open() }
    }

    /** Follows what the person is writing, to open a new conversation's session meanwhile. */
    fun draftChanged(text: String) {
        if (openedFor != null || opened.value != null || opening != null || text.isEmpty()) return
        screenScope.launch {
            val chat = open()
            if (chat != null && !isAnnounced) unused = chat.session.id
        }
    }

    fun send(text: String) {
        val asking = Asked(AIPendingMessage(text = text), after = opened.value?.messages?.value?.count { !it.isAgentReply } ?: 0)
        asked.value = asking
        screenScope.launch {
            val chat = open()
            if (chat == null) {
                asked.value = null
                return@launch
            }
            markUsed(chat)
            chat.ask(text)
            // Stream's copy of the question is the person's first message after the ones there
            // were when they asked. Once it is in, or asking failed, the question on its way goes.
            combine(chat.messages, chat.failure) { messages, failure ->
                failure != null || messages.count { !it.isAgentReply } > asking.after
            }.first { it }
            if (asked.value === asking) asked.value = null
        }
    }

    /** Stops the reply in flight. What was written so far is kept. */
    fun stop() {
        opened.value?.stop()
    }

    suspend fun loadOlderMessages() {
        opened.value?.loadOlderMessages()
    }

    /** Rates a reply, or takes the rating back with null. */
    fun rate(reply: String, rating: AIMessageRating?) {
        opened.value?.rate(reply, rating)
    }

    override fun onCleared() {
        leave()
    }

    /** Tells the list about a new conversation once it is used. */
    private fun markUsed(chat: AgentChat) {
        unused = null
        if (openedFor != null || isAnnounced) return
        isAnnounced = true
        shown.value = chat.session.id
    }

    /** The conversation's chat, starting a session for a new one. */
    private suspend fun open(): AgentChat? {
        opened.value?.let { return it }
        opening?.let { return it.await() }
        val task = screenScope.async { start() }
        opening = task
        val chat = task.await()
        opening = null
        return chat
    }

    /**
     * Carries on the conversation's session, or starts one for a new conversation, and opens its
     * channel.
     */
    private suspend fun start(): AgentChat? {
        failed.value = null
        return try {
            val session = openedFor?.let { Assistant.agents.attach(it, tools) } ?: Assistant.agents.agent(Assistant.AGENT_NAME).chat(
                SessionOptions(llm = Assistant.CHAT_MODEL, tools = tools, returnUrl = Assistant.RETURN_URL),
            )
            AgentChat.open(getApplication(), session, screenScope).also { opened.value = it }
        } catch (e: AgentsException) {
            failed.value = e.message
            null
        }
    }

    /**
     * Closes the conversation on screen, and deletes a new one's session if it was never used.
     * The screen's scope is gone by then, and closing tells the router the conversation is over.
     */
    private fun leave() {
        val closing = opened.value
        val abandoned = unused
        screenScope.cancel()
        CoroutineScope(Dispatchers.IO).launch {
            withContext(NonCancellable) {
                closing?.close()
                if (abandoned != null) {
                    try {
                        Assistant.sessions.delete(abandoned)
                    } catch (_: AgentsException) {
                        // Left in the list, untitled, which the person can delete there.
                    }
                }
            }
        }
    }

    private fun newScreenScope(): CoroutineScope =
        CoroutineScope(viewModelScope.coroutineContext + SupervisorJob(viewModelScope.coroutineContext[Job]))

    /**
     * A question on its way, as the person sees it before Stream's copy of it arrives, and how
     * many of the person's messages the conversation held when it was asked: the next one is it.
     */
    private class Asked(val message: AIPendingMessage, val after: Int)
}
