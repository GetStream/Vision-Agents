package io.getstream.agents.androidassistant

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.getstream.agents.android.core.AgentsException
import io.getstream.agents.android.core.Session
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * This person's conversations with the assistant, most recently active first.
 *
 * The router keeps them: a session belongs to whoever opened it, so this lists this phone's
 * user's and nobody else's. It titles each one itself once there is something to go on, and a
 * conversation that ended carries on when it is next asked something, so there is nothing to
 * keep here but the list.
 */
class Conversations : ViewModel() {
    private val found = MutableStateFlow<List<Session>>(emptyList())
    private val search = MutableStateFlow("")
    private val failed = MutableStateFlow<String?>(null)

    /** The conversations found, newest first. */
    val items: StateFlow<List<Session>> = found.asStateFlow()

    /** What the list is narrowed to, or empty for all of them. */
    val searchText: StateFlow<String> = search.asStateFlow()

    /** Why the list could not be read, or null. */
    val failure: StateFlow<String?> = failed.asStateFlow()

    /** Narrows the list to conversations called something like [text]; [reload] reads it. */
    fun search(text: String) {
        search.value = text
    }

    /** Reads the list again, narrowed to [searchText] when there is one. */
    suspend fun reload() {
        try {
            val text = search.value
            val page = if (text.isEmpty()) Assistant.sessions.query(limit = 50) else Assistant.sessions.search(text, limit = 50)
            // Only the ones kept in writing can be opened again as a chat: a session that was
            // only ever a call, or incognito, has no channel to read back.
            found.value = page.items.filter { it.conversationId.isNotEmpty() }
            failed.value = null
        } catch (e: AgentsException) {
            failed.value = e.message
        }
    }

    /** Deletes a conversation: its turns, and what it remembered, go with it. */
    fun delete(session: Session) {
        found.update { items -> items.filterNot { it.id == session.id } }
        viewModelScope.launch {
            try {
                Assistant.sessions.delete(session.id)
            } catch (e: AgentsException) {
                failed.value = e.message
                reload()
            }
        }
    }
}

/** What a conversation is called in the list: the router's title once it has given it one. */
val Session.displayTitle: String get() = title.ifEmpty { "New conversation" }
