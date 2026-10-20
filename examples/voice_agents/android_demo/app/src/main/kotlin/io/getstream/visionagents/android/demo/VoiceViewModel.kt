package io.getstream.visionagents.android.demo

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.getstream.visionagents.android.core.AgentsException
import io.getstream.visionagents.android.core.Conversation
import io.getstream.visionagents.android.core.SessionOptions
import io.getstream.visionagents.android.core.VoiceSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * A spoken conversation, held for as long as the app is open.
 *
 * Talking does two things: the router starts a session, which puts the agent on a call, and
 * this device joins it as the user `setUser` named, with the same token. The transcript comes
 * off the session socket rather than out of the call. A tool that asks first asks here, over the
 * call, and the agent waits for the answer.
 */
class VoiceViewModel(application: Application) : AndroidViewModel(application) {
    private val current = MutableStateFlow<VoiceSession?>(null)
    private val starting = MutableStateFlow(false)
    private val failed = MutableStateFlow<String?>(null)
    private var watching: Job? = null

    /** The call in progress, or null before one starts and after it ends. */
    val voice: StateFlow<VoiceSession?> = current.asStateFlow()

    val isStarting: StateFlow<Boolean> = starting.asStateFlow()

    /** Why the last call could not start, or null. */
    val failure: StateFlow<String?> = failed.asStateFlow()

    fun start() {
        if (starting.value || current.value != null) return
        starting.value = true
        failed.value = null
        viewModelScope.launch {
            try {
                val voice = VoiceSession.start(
                    getApplication(),
                    Demo.agents,
                    SessionOptions(agent = Demo.AGENT_NAME, tools = Demo.tools),
                )
                current.value = voice
                voice.join()
                // The agent leaving is the call being over, so let go of it and offer to start
                // another rather than leaving dead controls on screen.
                watching = launch {
                    voice.session.conversation.first { it.state == Conversation.State.Ended }
                    hangUp()
                }
            } catch (e: AgentsException) {
                failed.value = e.message
            } finally {
                starting.value = false
            }
        }
    }

    fun setMuted(muted: Boolean) {
        current.value?.setMuted(muted)
    }

    fun hangUp() {
        val voice = current.value ?: return
        current.value = null
        watching?.cancel()
        viewModelScope.launch { withContext(NonCancellable) { voice.end() } }
    }

    override fun onCleared() {
        val voice = current.value ?: return
        // The screen's scope is gone by now, and ending tells the router the call is over.
        CoroutineScope(Dispatchers.IO).launch {
            withContext(NonCancellable) { voice.end() }
        }
    }
}
