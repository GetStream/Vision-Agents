package io.getstream.visionagents.androidassistant

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
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
import io.getstream.chat.android.ai.compose.ui.component.LocalChatAiComponentFactory
import io.getstream.chat.android.ai.compose.ui.component.MessageData
import java.util.Calendar

private val starters = listOf(
    "What's on my calendar today?",
    "Remind me to call mum at 6pm",
    "Plan a weekend in Lisbon",
    "What should I cook tonight?",
)

/**
 * One conversation: the messages, and the composer floating over the bottom of them.
 *
 * The person's messages are in bubbles, the agent's as their steps and then their answer. What
 * the person asks shows at once, and Stream's copy takes its place. Each question goes to the top
 * as it is asked, and its reply writes itself out below it without moving what is on screen.
 * While the agent thinks, the list shows its thinking, which the reply's first step takes over
 * once it arrives.
 */
@Composable
fun ChatScreen(viewModel: ChatViewModel, modifier: Modifier = Modifier) {
    val messages by viewModel.messages.collectAsStateWithLifecycle()
    val pending by viewModel.pending.collectAsStateWithLifecycle()
    val activity by viewModel.activity.collectAsStateWithLifecycle()
    val answering by viewModel.isAnswering.collectAsStateWithLifecycle()
    val failure by viewModel.failure.collectAsStateWithLifecycle()
    val hasOlder by viewModel.hasOlderMessages.collectAsStateWithLifecycle()
    val chat by viewModel.chat.collectAsStateWithLifecycle()
    val approver = chat?.let { rememberApprover(it) }
    val views = remember(viewModel) { AssistantViews(viewModel) }
    var draft by rememberSaveable(stateSaver = MessageData.Saver) { mutableStateOf(MessageData()) }
    // How tall the composer is, keyboard included, which the end of the conversation keeps
    // clear of so the latest message is never under it.
    var barHeight by remember { mutableStateOf(0.dp) }
    val density = LocalDensity.current
    val isWelcoming = messages.isEmpty() && pending == null

    CompositionLocalProvider(LocalChatAiComponentFactory provides views) {
        Box(modifier.fillMaxSize()) {
            AIConversation(
                messages = messages,
                modifier = Modifier.fillMaxSize(),
                pendingMessage = pending,
                isThinking = activity == AgentChat.Activity.Thinking,
                approver = approver,
                loadOlderMessages = if (hasOlder) viewModel::loadOlderMessages else null,
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
                    Spacer(Modifier.height(barHeight))
                },
                emptyContent = { Welcome(Modifier.padding(bottom = barHeight)) },
            )

            // The conversation fades away under the composer rather than stopping at a hairline,
            // so the composer does not read as a bar of its own.
            Column(
                Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .onSizeChanged { barHeight = with(density) { it.height.toDp() } }
                    .background(
                        Brush.verticalGradient(
                            listOf(
                                Color.Transparent,
                                MaterialTheme.colorScheme.background.copy(alpha = 0.85f),
                                MaterialTheme.colorScheme.background,
                            ),
                        ),
                    ),
            ) {
                if (isWelcoming) {
                    AISuggestions(suggestions = starters, onSuggestionClick = { viewModel.send(it) })
                }
                failure?.let {
                    Surface(
                        shape = CircleShape,
                        color = MaterialTheme.colorScheme.surfaceContainer,
                        modifier = Modifier.padding(horizontal = 16.dp).align(Alignment.CenterHorizontally),
                    ) {
                        Text(
                            it,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                        )
                    }
                }
                ChatComposer(
                    messageData = draft,
                    onMessageDataChange = {
                        draft = it
                        viewModel.draftChanged(it.text)
                    },
                    onSendClick = { viewModel.send(it.text, it.attachments) },
                    onStopClick = viewModel::stop,
                    isGenerating = answering,
                )
            }
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
        AIToolApprover(Assistant.user.id, AIClientIdentity.installId(context), scope) { call, allowed ->
            chat.session.decide(call.id, allowed)
        }
    }
}

/** What a new conversation shows before anything is said. */
@Composable
private fun Welcome(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.padding(32.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Outlined.AutoAwesome,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(44.dp),
        )
        Text(greeting(), style = MaterialTheme.typography.titleLarge)
        Text(
            "Ask me anything, or show me a photo.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

private fun greeting(): String {
    val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
    val part = if (hour < 12) "morning" else if (hour < 18) "afternoon" else "evening"
    return "Good $part, ${Assistant.user.name}"
}

/**
 * Stream's views, with what the person can do under each finished reply: copy it, share it and
 * rate it. Asking for a reply again needs the router to replace one, which it does not yet.
 */
private class AssistantViews(private val viewModel: ChatViewModel) : ChatAiComponentFactory {
    @Composable
    override fun AIConversationReplyActions(params: AIConversationReplyActionsParams) {
        // Read here, so the thumbs change as soon as the rating does.
        val ratings by viewModel.ratings.collectAsStateWithLifecycle()
        val reply = params.message.id
        AIReplyActionsRow(
            text = params.text,
            actions = AIReplyActions(rating = ratings[reply], onRate = { viewModel.rate(reply, it) }),
        )
    }
}
