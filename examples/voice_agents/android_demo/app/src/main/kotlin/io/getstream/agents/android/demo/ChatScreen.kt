package io.getstream.agents.android.demo

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Inventory2
import androidx.compose.material.icons.outlined.WarningAmber
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.getstream.chat.android.ai.compose.conversation.AIAgentError
import io.getstream.chat.android.ai.compose.conversation.AIConversationState
import io.getstream.chat.android.ai.compose.conversation.AIConversationStatus
import io.getstream.chat.android.ai.compose.ui.component.AIConversation
import io.getstream.chat.android.ai.compose.ui.component.AISuggestions
import io.getstream.chat.android.ai.compose.ui.component.ChatAiComponentFactory
import io.getstream.chat.android.ai.compose.ui.component.ChatComposer
import io.getstream.chat.android.ai.compose.ui.component.ComposerLeadingContentParams
import io.getstream.chat.android.ai.compose.ui.component.LocalChatAiComponentFactory

/** A written conversation with the agent, opened when the screen is first shown. */
@Composable
fun ChatScreen(modifier: Modifier = Modifier, viewModel: ChatViewModel = viewModel()) {
    val state by viewModel.chat.collectAsStateWithLifecycle()
    when (val current = state) {
        ChatViewModel.State.Opening -> Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        is ChatViewModel.State.Failed -> EmptyState(
            icon = Icons.Outlined.WarningAmber,
            title = "Could not start",
            message = current.reason,
            modifier = modifier,
        )
        is ChatViewModel.State.Open -> AgentConversation(current.conversation, modifier)
    }
}

private val starters = listOf(
    "Where is order A-1042?",
    "Can I return order A-1043?",
    "I'd like a refund for order A-1042",
)

@Composable
private fun AgentConversation(conversation: AIConversationState<AgentMessage>, modifier: Modifier = Modifier) {
    val messages by conversation.messages.collectAsStateWithLifecycle()
    val pending by conversation.pendingMessage.collectAsStateWithLifecycle()
    val status by conversation.status.collectAsStateWithLifecycle()
    val error by conversation.error.collectAsStateWithLifecycle()

    CompositionLocalProvider(LocalChatAiComponentFactory provides DemoViews) {
        Column(modifier.fillMaxSize().imePadding()) {
            // A question moves to the top as it is asked, and its reply writes itself out below
            // it without moving what is on screen. While the agent thinks, the list shows its
            // thinking, which the reply's first step takes over once it arrives.
            AIConversation(
                state = conversation,
                modifier = Modifier.weight(1f),
                footer = {
                    error?.takeIf { status == AIConversationStatus.Error }?.let {
                        Failure(it, onRetry = conversation::retry)
                    }
                },
                emptyContent = {
                    EmptyState(
                        icon = Icons.Outlined.Inventory2,
                        title = "Ask about an order",
                        message = "Orders A-1042 and A-1043 are on this account.",
                    )
                },
            )

            if (messages.isEmpty() && pending == null) {
                AISuggestions(suggestions = starters, onSuggestionClick = { conversation.send(it) })
            }
            ChatComposer(state = conversation, modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp))
        }
    }
}

/**
 * Why the latest question failed, under it. A question the router never had is sent again; one
 * the agent couldn't answer, or older messages that couldn't load, are let go, and the person asks
 * again.
 */
@Composable
private fun Failure(error: Throwable, onRetry: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.fillMaxWidth().padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            if (error is AIAgentError) "The agent couldn't answer." else error.message ?: "Something went wrong.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.error,
            modifier = Modifier.weight(1f, fill = false),
        )
        TextButton(onClick = onRetry) { Text(if (error is AIAgentError) "OK" else "Try again") }
    }
}

@Composable
private fun EmptyState(
    icon: ImageVector,
    title: String,
    message: String,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(bottom = 8.dp).size(56.dp),
        )
        Text(title, style = MaterialTheme.typography.titleLarge)
        Text(message, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/**
 * Stream's views, with the composer without its attachment button: a question goes to the agent
 * as text, and dictating one stays. Under each finished reply the conversation offers to copy,
 * share and rate it; asking for a reply again needs the router to replace one, which it does not
 * yet.
 */
private object DemoViews : ChatAiComponentFactory {
    @Composable
    override fun RowScope.ComposerLeadingContent(params: ComposerLeadingContentParams) = Unit
}
