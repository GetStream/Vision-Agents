package io.getstream.visionagents.android.demo

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Inventory2
import androidx.compose.material.icons.outlined.WarningAmber
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.getstream.chat.android.ai.compose.parts.AIClientIdentity
import io.getstream.chat.android.ai.compose.ui.component.AIMessagePartItem
import io.getstream.chat.android.ai.compose.ui.component.AIMessageParts
import io.getstream.chat.android.ai.compose.ui.component.AISuggestions
import io.getstream.chat.android.ai.compose.ui.component.AIToolApprover
import io.getstream.chat.android.ai.compose.ui.component.AITypingIndicator
import io.getstream.chat.android.ai.compose.ui.component.ChatAiComponentFactory
import io.getstream.chat.android.ai.compose.ui.component.ChatComposer
import io.getstream.chat.android.ai.compose.ui.component.ComposerLeadingContentParams
import io.getstream.chat.android.ai.compose.ui.component.LocalChatAiComponentFactory
import io.getstream.chat.android.ai.compose.ui.component.StreamingReasoning
import io.getstream.chat.android.ai.compose.ui.component.StreamingText
import io.getstream.chat.android.models.Message
import kotlinx.coroutines.delay

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
    val messages by chat.messages.collectAsStateWithLifecycle()
    val indicator by chat.indicator.collectAsStateWithLifecycle()
    val answering by chat.isAnswering.collectAsStateWithLifecycle()
    val failure by chat.failure.collectAsStateWithLifecycle()
    val approver = rememberApprover(chat)

    Column(modifier.fillMaxSize().imePadding()) {
        Box(Modifier.weight(1f)) {
            // Reversed, so the conversation starts from the bottom and the newest stays in view.
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                reverseLayout = true,
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.Bottom),
            ) {
                indicator?.let { status ->
                    item(key = "indicator") {
                        AITypingIndicator(label = {
                            Text(status, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        })
                    }
                }
                items(messages.asReversed(), key = { it.id }) { message ->
                    MessageRow(message, chat, approver)
                }
            }
            if (messages.isEmpty()) {
                EmptyState(
                    icon = Icons.Outlined.Inventory2,
                    title = "Ask about an order",
                    message = "Orders A-1042 and A-1043 are on this account.",
                )
            }
        }

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
        // The composer without its attachment button: a question goes to the agent as text, and
        // dictating one stays.
        CompositionLocalProvider(LocalChatAiComponentFactory provides TextOnlyComposer) {
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

/** One message: the person's in a bubble, the agent's as its steps and then its answer. */
@Composable
private fun MessageRow(message: Message, chat: AgentChat, approver: AIToolApprover) {
    if (!message.isAgentReply) {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.CenterEnd) {
            Text(
                message.text,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier
                    .widthIn(max = 320.dp)
                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.15f), RoundedCornerShape(18.dp))
                    .padding(horizontal = 14.dp, vertical = 10.dp),
            )
        }
        return
    }

    // Whether the text is shown as still being written, which is what types it out. Stream
    // folds a fast reply's live updates into one, so its text can arrive in the same update
    // that says it is finished. StreamingText types only what arrives while it is told the reply
    // is being written, so it is told that until the text has reached it.
    var writing by remember(message.id) { mutableStateOf(message.isGenerating) }
    LaunchedEffect(message.isGenerating) {
        if (message.isGenerating) {
            writing = true
        } else {
            delay(100)
            writing = false
        }
    }
    // Read so that a step redraws as more of its thinking arrives.
    val version by chat.reasoningVersion.collectAsStateWithLifecycle()

    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val parts = message.parts
        if (parts.isNotEmpty()) {
            AIMessageParts(parts = parts) { part ->
                // The step keeps only the opening of its thinking; the whole of it came with the
                // live updates, while the reply was being written.
                val step = part.reasoning
                if (step != null) {
                    StreamingReasoning(
                        text = version.let { chat.reasoning[step.id] } ?: step.preview.orEmpty(),
                        isThinking = step.isStreaming,
                        durationSeconds = step.durationSeconds,
                        summary = step.summary,
                    )
                } else {
                    AIMessagePartItem(part = part, approver = approver)
                }
            }
        }
        if (message.text.isNotEmpty()) {
            StreamingText(text = message.text, animate = writing)
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

private object TextOnlyComposer : ChatAiComponentFactory {
    @Composable
    override fun RowScope.ComposerLeadingContent(params: ComposerLeadingContentParams) = Unit
}
