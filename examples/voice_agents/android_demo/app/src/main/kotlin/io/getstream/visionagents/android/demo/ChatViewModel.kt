package io.getstream.visionagents.android.demo

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.getstream.visionagents.android.core.AgentsException
import io.getstream.visionagents.android.core.SessionOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Opens a written conversation and holds it for as long as the screen is around.
 *
 * Nothing is transcribed and nothing is spoken, so no call is joined. Everything between
 * hearing a question and answering it is the same as on a call: the same instructions, the
 * same knowledge, the same skills and the same tools running on this phone. What is new in
 * writing is that the reply shows its working, with Stream's AI components: the agent's
 * reasoning as it thinks, each tool call as it runs, and a question when a tool asks first.
 */
class ChatViewModel(application: Application) : AndroidViewModel(application) {
    sealed interface State {
        data object Opening : State
        data class Open(val chat: AgentChat) : State
        data class Failed(val reason: String) : State
    }

    private val state = MutableStateFlow<State>(State.Opening)
    val chat: StateFlow<State> = state.asStateFlow()

    init {
        viewModelScope.launch {
            state.value = try {
                // A model that streams its reasoning, which is what fills the reply's thinking
                // panel. A call keeps the agent's own model, which answers sooner.
                val options = SessionOptions(agent = Demo.AGENT_NAME, llm = "xai/grok-4.7", tools = Demo.tools)
                val session = Demo.agents.chat(options)
                State.Open(AgentChat.open(getApplication(), session, viewModelScope))
            } catch (e: AgentsException) {
                State.Failed(e.message.orEmpty())
            }
        }
    }

    override fun onCleared() {
        val open = state.value as? State.Open ?: return
        // The screen's scope is gone by now, and closing tells the router the conversation is over.
        CoroutineScope(Dispatchers.IO).launch {
            withContext(NonCancellable) { open.chat.close() }
        }
    }
}
