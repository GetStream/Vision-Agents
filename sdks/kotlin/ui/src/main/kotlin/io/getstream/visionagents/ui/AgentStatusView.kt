package io.getstream.visionagents.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import io.getstream.visionagents.core.Conversation

/** What the agent is doing, in a line. */
@Composable
public fun AgentStatusView(state: Conversation.State, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier,
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        if (state.isBusy) {
            CircularProgressIndicator(modifier = Modifier.size(12.dp), strokeWidth = 1.5.dp)
        }
        Text(
            state.label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private val Conversation.State.isBusy: Boolean
    get() = when (this) {
        Conversation.State.Responding, Conversation.State.Listening, is Conversation.State.Working -> true
        Conversation.State.Idle, Conversation.State.Ended -> false
    }

private val Conversation.State.label: String
    get() = when (this) {
        Conversation.State.Idle -> "ready"
        Conversation.State.Listening -> "listening"
        Conversation.State.Responding -> "answering"
        is Conversation.State.Working ->
            if (skills.isEmpty()) "thinking" else "thinking (${skills.joinToString()})"
        Conversation.State.Ended -> "the conversation ended"
    }
