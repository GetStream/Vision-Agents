package io.getstream.agents.androidassistant

import android.Manifest
import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.getstream.chat.android.ai.a2ui.generativeui.A2UIRenderer
import io.getstream.chat.android.ai.compose.conversation.AIAgentError
import io.getstream.chat.android.ai.compose.conversation.AIConversationState
import io.getstream.chat.android.ai.compose.generativeui.AIUIAction
import io.getstream.chat.android.ai.compose.generativeui.AIUIHost
import io.getstream.chat.android.ai.compose.parts.AIClientIdentity
import io.getstream.chat.android.ai.compose.ui.component.AIToolApprover
import io.getstream.chat.android.ai.compose.ui.component.AIToolAuthorizer
import io.getstream.chat.android.ai.openui.generativeui.OpenUIRenderer
import io.getstream.agents.android.core.AgentSession
import io.getstream.agents.android.core.AgentsException
import io.getstream.agents.android.core.Conversation
import io.getstream.agents.android.core.ModelOverwrites
import io.getstream.agents.android.core.SessionOptions
import io.getstream.agents.android.core.VoiceSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flatMapLatest
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
    private val failed = MutableStateFlow<String?>(null)

    /** The conversation open, or null for a new one that has not been written to. */
    val selected: StateFlow<String?> = shown.asStateFlow()

    /**
     * Which conversation the screen was opened for. It changes only when the person picks one,
     * so a new conversation getting its session keeps the screen it was started on.
     */
    val screen: StateFlow<Int> = screenId.asStateFlow()

    // What belongs to the conversation on screen, and goes when another is picked.
    private var screenScope = newScreenScope()
    private var openedFor: String? = null
    private var transport = newTransport()
    private val conversations = MutableStateFlow(newConversation())

    /**
     * The conversation on screen: its messages, the question on its way, what the agent is doing,
     * and why the latest question went unanswered. A new one each time another conversation is
     * picked, which lives as long as the screen's scope.
     */
    val conversation: StateFlow<AIConversationState<AgentMessage>> = conversations.asStateFlow()

    private val transports = MutableStateFlow(transport)

    /** The conversation's session, once it is open. */
    val session: StateFlow<AgentSession?> = transports.flatMapLatest { it.session }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** Why opening failed, or why the session stopped, or null. */
    val failure: StateFlow<String?> = session.flatMapLatest { session ->
        if (session == null) failed else combine(failed, session.failure) { opening, stopped -> opening ?: stopped?.message }
    }.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /**
     * The activity the sign-in pages of connectors open over, while the screen is up. The tool
     * approver outlives it, since it lives with the conversation.
     */
    var signIn: SignIn? = null

    // The latest question or action asked here, which asking again after it went unanswered
    // repeats: the conversation's own retry only resends what never reached the router.
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
        conversations.value.send(action)
    }

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
        isAnnounced = false
        unused = null
        failed.value = null
        latest = null
        transport = newTransport()
        transports.value = transport
        conversations.value = newConversation()
        shown.value = sessionId
        screenId.value++
        if (sessionId != null) screenScope.launch { open() }
    }

    /** Follows what the person is writing, to open a new conversation's session meanwhile. */
    fun draftChanged(text: String) {
        if (openedFor != null || session.value != null || text.isEmpty()) return
        val transport = transport
        screenScope.launch {
            val session = open()
            if (session != null && !isAnnounced && transport === this@ChatViewModel.transport) unused = session.id
        }
    }

    /**
     * Asks, with the photos the composer holds, which show on the question at once. The
     * conversation opens a new conversation's session first.
     */
    fun send(text: String, photos: Collection<Uri> = emptyList()) {
        latest = { send(text, photos) }
        conversations.value.send(text, photos.toList())
    }

    /**
     * Tries again after the latest question or action failed. One the router never had is sent
     * again as it was. One it had and could not answer is asked again, since the conversation
     * leaves that to the person.
     */
    fun retry() {
        val conversation = conversations.value
        if (conversation.error.value is AIAgentError) latest?.invoke() else conversation.retry()
    }

    /** Switches to voice mode on this conversation, starting a session for a new one. */
    fun talk() {
        screenScope.launch {
            val session = open() ?: return@launch
            markUsed(session)
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
        val session = this.session.value ?: return
        if (voiceJob != null) return
        voiceJob = viewModelScope.launch {
            if (!Permissions.allOf(getApplication(), Manifest.permission.RECORD_AUDIO)) {
                voiceFailed.value = "Talking to Juno needs the microphone."
                return@launch
            }
            try {
                session.startVoice()
                val voice = VoiceSession.attach(getApplication(), Assistant.agents, session.id)
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
        val session = this.session.value?.takeIf { voiceShown.value }
        closeVoice()
        if (session != null) {
            // Not the screen's scope, which may be going: the agent has to be taken off either way.
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    withContext(NonCancellable) { session.stopVoice() }
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
    private fun markUsed(session: AgentSession) {
        // The session is open after all, so why it couldn't open no longer holds.
        failed.value = null
        unused = null
        if (openedFor != null || isAnnounced) return
        isAnnounced = true
        shown.value = session.id
    }

    /** The conversation's session, starting one for a new conversation, or null when it can't. */
    private suspend fun open(): AgentSession? {
        failed.value = null
        return try {
            transport.open()
        } catch (e: AgentsException) {
            failed.value = e.message
            null
        }
    }

    /** Carries on the conversation's session, or starts one for a new conversation. */
    private fun newTransport(): AgentTransport {
        val sessionId = openedFor
        return AgentTransport(
            context = getApplication(),
            scope = screenScope,
            start = {
                sessionId?.let { Assistant.agents.attach(it, tools) } ?: Assistant.agents.agent(Assistant.AGENT_NAME).chat(
                    SessionOptions(modelOverwrites = ModelOverwrites(llm = Assistant.CHAT_MODEL), tools = tools, returnUrl = Assistant.RETURN_URL),
                )
            },
            onUsed = ::markUsed,
        )
    }

    private fun newConversation(): AIConversationState<AgentMessage> =
        AIConversationState(transport, screenScope, approver(transport))

    /**
     * Answers the questions of the tools that ask first, as the person signed in here, and signs
     * them in when a connector's call waits for their account.
     *
     * The router addresses a question to the person whose message the call answers, and Stream's
     * components show it only to them. The answer goes back over the session's socket, which is
     * also what runs the tool once it is allowed. Connect opens the sign-in page in an Auth Tab;
     * the page comes back to this app with `status=connected` when the account is connected, and
     * the call carries on at the router. Not now ends the call there.
     */
    private fun approver(transport: AgentTransport): AIToolApprover {
        val authorizer = AIToolAuthorizer(
            connect = { _, url -> signIn?.authenticate(Uri.parse(url))?.getQueryParameter("status") == "connected" },
            decline = { call -> transport.session.value?.declineLogin(call.id) },
        )
        val installId = AIClientIdentity.installId(getApplication())
        return AIToolApprover(Assistant.user.id, installId, screenScope, authorizer) { call, allowed ->
            transport.session.value?.decide(call.id, allowed)
        }
    }

    /**
     * Closes the conversation on screen, and deletes a new one's session if it was never used.
     * The screen's scope is gone by then, and closing tells the router the conversation is over.
     */
    private fun leave() {
        val closing = transport
        val abandoned = unused
        screenScope.cancel()
        CoroutineScope(Dispatchers.IO).launch {
            withContext(NonCancellable) {
                closing.close()
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
}
