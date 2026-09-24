package io.getstream.visionagents.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.getstream.visionagents.core.AgentSession
import io.getstream.visionagents.core.AgentsException

/**
 * A whole conversation: the transcript, what the agent is doing, and somewhere to type.
 *
 * The three parts are public and work on their own, so a host that wants a different
 * arrangement can take them apart rather than fight this. It opens the socket when it first
 * appears; closing the session is the owner's, usually a `ViewModel` in `onCleared`, because a
 * rotation recomposes this and must not end the conversation.
 */
@Composable
public fun ConversationView(
    session: AgentSession,
    modifier: Modifier = Modifier,
    prompt: String = "Message",
) {
    val conversation by session.conversation.collectAsStateWithLifecycle()
    val connected by session.isConnected.collectAsStateWithLifecycle()
    val failure by session.failure.collectAsStateWithLifecycle()
    var opening by remember(session) { mutableStateOf<AgentsException?>(null) }

    LaunchedEffect(session) {
        try {
            session.start()
        } catch (e: AgentsException) {
            opening = e
        }
    }

    Column(modifier = modifier) {
        TranscriptView(conversation.turns, Modifier.weight(1f).fillMaxWidth())

        val problem = (failure ?: opening)?.message ?: conversation.failure
        if (problem != null) {
            Text(
                problem,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }

        AgentStatusView(conversation.state, Modifier.padding(horizontal = 16.dp))

        Composer(
            send = {
                try {
                    session.send(it)
                } catch (_: AgentsException) {
                    // The socket is gone, which `failure` and `enabled` already show.
                }
            },
            modifier = Modifier.padding(16.dp),
            prompt = prompt,
            enabled = connected,
        )
    }
}
