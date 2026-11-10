package io.getstream.agents.androidassistant

import android.Manifest
import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.getstream.chat.android.ai.a2ui.generativeui.A2UIRenderer
import io.getstream.chat.android.ai.compose.conversation.AIMessageAttachment
import io.getstream.chat.android.ai.compose.conversation.AIMessageRating
import io.getstream.chat.android.ai.compose.conversation.AIPendingMessage
import io.getstream.chat.android.ai.compose.generativeui.AIUIAction
import io.getstream.chat.android.ai.compose.generativeui.AIUIHost
import io.getstream.chat.android.ai.openui.generativeui.OpenUIRenderer
import io.getstream.agents.android.core.AgentsException
import io.getstream.agents.android.core.Conversation
import io.getstream.agents.android.core.ModelOverwrites
import io.getstream.agents.android.core.SessionOptions
import io.getstream.agents.android.core.VoiceSession
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

    /** Why opening or loading failed, or why the session stopped, or null. */
    val failure: StateFlow<String?> = opened.flatMapLatest { chat ->
        if (chat == null) {
            failed
        } else {
            combine(failed, chat.failure, chat.session.failure) { opening, loading, session ->
                opening ?: loading ?: session?.message
            }
        }
    }.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /**
     * Why the latest question or action went unanswered, shown under it with a way to [retry],
     * or null.
     */
    val unanswered: StateFlow<String?> = opened.flatMapLatest { it?.unanswered ?: flowOf(null) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    // The latest question or action asked here, which retrying asks again.
    private var latest: (() -> Unit)? = null

    private val calling = MutableStateFlow<VoiceSession?>(null)
    private val voiceShown = MutableStateFlow(false)
    private val voiceFailed = MutableStateFlow<String?>(null)
    private var voiceJob: Job? = null

    /** Whether voice mode is up, over the chat. */
    val isVoiceShown: StateFlow<Boolean> = voiceShown.asStateFlow()

    /** The call in voice mode, once it has started. */
    val voice: StateFlow<VoiceSession?> = calling.asStateFlow()

    /** Why voice mode could not start, or why the camera could not, or null. */
    val voiceFailure: StateFlow<String?> = voiceFailed.asStateFlow()

    /** The tools the agent may call, which run on this phone. */
    private val tools = DeviceTools(application).all

    private val format = MutableStateFlow(ReplyFormat.stored(application))

    /** How the agent writes its replies, which picks the renderer for their generated UI. */
    val replyFormat: StateFlow<ReplyFormat> = format.asStateFlow()

    /**
     * Renders the generated UI of replies, in either format, and sends back what the person does
     * in it. One for the screen's life, whatever the reply format, so earlier replies keep their UI
     * and its state when the format changes, Markdown included.
     */
    val uiHost: AIUIHost = AIUIHost(renderers = listOf(OpenUIRenderer(), A2UIRenderer())) { action -> answer(action) }

    init {
        // The backend's agent decides the format; follow it when it can say.
        viewModelScope.launch {
            ReplyFormat.fetch()?.takeIf { it != format.value }?.let(::useFormat)
        }
    }

    /** Has the agent write in [chosen] from the next new conversation. */
    fun chooseReplyFormat(chosen: ReplyFormat) {
        useFormat(chosen)
        viewModelScope.launch {
            if (!ReplyFormat.save(chosen)) failed.value = "The backend would not change the reply format."
        }
    }

    private fun useFormat(chosen: ReplyFormat) {
        if (chosen == format.value) return
        format.value = chosen
        ReplyFormat.store(getApplication(), chosen)
    }

    /** Sends what the person did in a reply's generated UI. */
    private fun answer(action: AIUIAction) {
        latest = { answer(action) }
        screenScope.launch {
            val chat = open() ?: return@launch
            markUsed(chat)
            chat.answer(action)
        }
    }

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
        latest = null
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

    /** Asks, with the photos the composer holds, which show on the question at once. */
    fun send(text: String, photos: Collection<Uri> = emptyList()) {
        latest = { send(text, photos) }
        val asking = Asked(
            AIPendingMessage(text = text, attachments = photos.map { AIMessageAttachment.image(it) }),
            after = opened.value?.messages?.value?.count { !it.isAgentReply } ?: 0,
        )
        asked.value = asking
        screenScope.launch {
            val chat = open()
            if (chat == null) {
                asked.value = null
                return@launch
            }
            markUsed(chat)
            // A question the router would not take is not in the channel, so the one on its way
            // stays, with why it went unanswered under it.
            if (!chat.ask(text) { Photos.sources(getApplication(), photos) }) return@launch
            // Stream's copy of the question is the person's first message after the ones there
            // were when they asked. Once it is in, or its reply failed, the question on its way goes.
            combine(chat.messages, chat.unanswered) { messages, unanswered ->
                unanswered != null || messages.count { !it.isAgentReply } > asking.after
            }.first { it }
            if (asked.value === asking) asked.value = null
        }
    }

    /** Asks the latest question again, or sends the latest action again, after it went unanswered. */
    fun retry() {
        latest?.invoke()
    }

    /** Switches to voice mode on this conversation, starting a session for a new one. */
    fun talk() {
        screenScope.launch {
            val chat = open() ?: return@launch
            markUsed(chat)
            voiceFailed.value = null
            voiceShown.value = true
        }
    }

    /**
     * Puts the agent on the chat session's call and joins it, with a second session of its own
     * that runs no tools: the chat's socket stays open and runs them. The microphone is asked for
     * first. The agent leaving ends voice mode.
     */
    fun startVoice() {
        val chat = opened.value ?: return
        if (voiceJob != null) return
        voiceJob = viewModelScope.launch {
            if (!Permissions.allOf(getApplication(), Manifest.permission.RECORD_AUDIO)) {
                voiceFailed.value = "Talking to Juno needs the microphone."
                return@launch
            }
            try {
                chat.session.startVoice()
                val voice = VoiceSession.attach(getApplication(), Assistant.agents, chat.session.id)
                calling.value = voice
                voice.join()
                voice.session.conversation.first { it.state == Conversation.State.Ended }
                closeVoice()
            } catch (e: AgentsException) {
                voiceFailed.value = e.message
            }
        }
    }

    /**
     * Leaves the call and takes the agent off it, which carries the conversation on in writing,
     * and goes back to the chat. Leaving an attached call keeps the session.
     */
    fun hangUp() {
        val chat = opened.value?.takeIf { voiceShown.value }
        closeVoice()
        if (chat != null) {
            // Not the screen's scope, which may be going: the agent has to be taken off either way.
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    withContext(NonCancellable) { chat.session.stopVoice() }
                } catch (_: AgentsException) {
                    // The agent had left already, which is what stopping voice was for.
                }
            }
        }
    }

    fun setMuted(muted: Boolean) {
        calling.value?.setMuted(muted)
    }

    /** Shows Juno the camera or stops, asking for the camera the first time. */
    fun setCameraEnabled(enabled: Boolean) {
        val voice = calling.value ?: return
        viewModelScope.launch {
            if (enabled && !Permissions.allOf(getApplication(), Manifest.permission.CAMERA)) {
                voiceFailed.value = "Showing Juno the camera needs the camera."
                return@launch
            }
            voiceFailed.value = null
            voice.setCameraEnabled(enabled)
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
        hangUp()
        leave()
    }

    private fun closeVoice() {
        val voice = calling.value
        voiceJob?.cancel()
        voiceJob = null
        calling.value = null
        voiceShown.value = false
        if (voice != null) CoroutineScope(Dispatchers.IO).launch { withContext(NonCancellable) { voice.leave() } }
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
                SessionOptions(modelOverwrites = ModelOverwrites(llm = Assistant.CHAT_MODEL), tools = tools, returnUrl = Assistant.RETURN_URL),
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
