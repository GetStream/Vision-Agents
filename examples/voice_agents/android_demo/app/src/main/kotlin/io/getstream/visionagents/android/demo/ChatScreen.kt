package io.getstream.visionagents.android.demo

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.getstream.chat.android.ai.compose.conversation.AIReplyActions
import io.getstream.chat.android.ai.compose.parts.AIClientIdentity
import io.getstream.chat.android.ai.compose.ui.component.AIConversation
import io.getstream.chat.android.ai.compose.ui.component.AIConversationReplyActionsParams
import io.getstream.chat.android.ai.compose.ui.component.AIReplyActionsRow
import io.getstream.chat.android.ai.compose.ui.component.AISuggestions
import io.getstream.chat.android.ai.compose.ui.component.AIToolApprover
import io.getstream.chat.android.ai.compose.ui.component.AITypingIndicator
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
        is ChatViewModel.State.Open -> AgentConversation(current.chat, modifier)
    }
}

private val starters = listOf(
    "Where is order A-1042?",
    "Can I return order A-1043?",
    "I'd like a refund for order A-1042",
)

@Composable
private fun AgentConversation(chat: AgentChat, modifier: Modifier = Modifier) {
    val messages by chat.conversation.collectAsStateWithLifecycle()
    val activity by chat.activity.collectAsStateWithLifecycle()
    val answering by chat.isAnswering.collectAsStateWithLifecycle()
    val failure by chat.failure.collectAsStateWithLifecycle()
    val hasOlder by chat.hasOlderMessages.collectAsStateWithLifecycle()
    val approver = rememberApprover(chat)
    val views = remember(chat) { DemoViews(chat) }

    CompositionLocalProvider(LocalChatAiComponentFactory provides views) {
        Column(modifier.fillMaxSize().imePadding()) {
            // A question moves to the top as it is asked, and its reply writes itself out below
            // it without moving what is on screen. While the agent thinks, the list shows its
            // thinking, which the reply's first step takes over once it arrives.
            AIConversation(
                messages = messages,
                modifier = Modifier.weight(1f),
                isThinking = activity == AgentChat.Activity.Thinking,
                approver = approver,
                loadOlderMessages = if (hasOlder) chat::loadOlderMessages else null,
                footer = {
                    (activity as? AgentChat.Activity.Working)?.let { working ->
                        AITypingIndicator(label = {
                            Text(
                                working.text,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        })
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

            if (messages.isEmpty()) {
                AISuggestions(suggestions = starters, onSuggestionClick = chat::ask)
            }
            failure?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(horizontal = 16.dp),
                )
            }
            ChatComposer(
                onSendClick = { chat.ask(it.text) },
                onStopClick = chat::stop,
                isGenerating = answering,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp),
            )
        }
    }
}

/**
 * Answers the questions of the tools that ask first, as the person signed in here.
 *
 * The router addresses a question to the person whose message the call answers, and Stream's
 * components show it only to them. The answer goes back over the session's socket, which is
 * also what runs the tool once it is allowed.
 */
@Composable
private fun rememberApprover(chat: AgentChat): AIToolApprover {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    return remember(chat) {
        AIToolApprover(Demo.user.id, AIClientIdentity.installId(context), scope) { call, allowed ->
            chat.session.decide(call.id, allowed)
        }
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
 * Stream's views, with what the person can do under each finished reply (copy it, share it and
 * rate it) and the composer without its attachment button: a question goes to the agent as text,
 * and dictating one stays. Asking for a reply again needs the router to replace one, which it
 * does not yet.
 */
private class DemoViews(private val chat: AgentChat) : ChatAiComponentFactory {
    @Composable
    override fun RowScope.ComposerLeadingContent(params: ComposerLeadingContentParams) = Unit

    @Composable
    override fun AIConversationReplyActions(params: AIConversationReplyActionsParams) {
        // Read here, so the thumbs change as soon as the rating does.
        val ratings by chat.ratings.collectAsStateWithLifecycle()
        val reply = params.message.id
        AIReplyActionsRow(
            text = params.text,
            actions = AIReplyActions(rating = ratings[reply], onRate = { chat.rate(reply, it) }),
        )
    }
}
