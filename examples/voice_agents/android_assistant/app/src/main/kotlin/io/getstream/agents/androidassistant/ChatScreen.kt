package io.getstream.agents.androidassistant

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.getstream.chat.android.ai.compose.conversation.AIConversationStatus
import io.getstream.chat.android.ai.compose.ui.component.AIConversation
import io.getstream.chat.android.ai.compose.ui.component.AISuggestions
import io.getstream.chat.android.ai.compose.ui.component.ChatAiComponentFactory
import io.getstream.chat.android.ai.compose.ui.component.ChatComposer
import io.getstream.chat.android.ai.compose.ui.component.ComposerTrailingContentParams
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
    val conversation by viewModel.conversation.collectAsStateWithLifecycle()
    val messages by conversation.messages.collectAsStateWithLifecycle()
    val pending by conversation.pendingMessage.collectAsStateWithLifecycle()
    val status by conversation.status.collectAsStateWithLifecycle()
    val error by conversation.error.collectAsStateWithLifecycle()
    val opening by viewModel.failure.collectAsStateWithLifecycle()
    val signIn = LocalSignIn.current
    DisposableEffect(viewModel, signIn) {
        viewModel.signIn = signIn
        onDispose { if (viewModel.signIn === signIn) viewModel.signIn = null }
    }
    val views = remember(viewModel) { AssistantViews(viewModel) }
    var draft by rememberSaveable(stateSaver = MessageData.Saver) { mutableStateOf(MessageData()) }
    // How tall the composer is, keyboard included, which the conversation keeps clear of so the
    // latest message and its scroll-to-latest button are never under it.
    var barHeight by remember { mutableStateOf(0.dp) }
    val density = LocalDensity.current
    val isWelcoming = messages.isEmpty() && pending == null
    // Older messages that could not load show with the other failures, over the composer; a
    // question that went unanswered, under it with a way to try again.
    val failure = opening ?: (error as? LoadFailed)?.message
    val unanswered = error?.takeIf { it !is LoadFailed && status == AIConversationStatus.Error }

    CompositionLocalProvider(LocalChatAiComponentFactory provides views) {
        Box(modifier.fillMaxSize()) {
            AIConversation(
                state = conversation,
                modifier = Modifier.fillMaxSize(),
                footer = { unanswered?.let { Unanswered(UNANSWERED, onRetry = viewModel::retry) } },
                emptyContent = { Welcome() },
                contentPadding = PaddingValues(bottom = barHeight),
                ui = viewModel.uiHost,
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
                    onStopClick = conversation::stop,
                    isGenerating = status.isAnswering,
                )
            }
        }
    }
}

/** Why the latest question went unanswered, under it, and the way to ask it again. */
@Composable
private fun Unanswered(reason: String, onRetry: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.fillMaxWidth().padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            reason,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.error,
            modifier = Modifier.weight(1f, fill = false),
        )
        TextButton(onClick = onRetry) { Text("Try again") }
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
            "Ask me anything, show me a photo, or tap the waveform to talk.",
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
 * Stream's views, with the way into voice mode beside the composer. Under each finished reply the
 * conversation offers to copy, share and rate it; asking for a reply again needs the router to
 * replace one, which it does not yet.
 */
private class AssistantViews(private val viewModel: ChatViewModel) : ChatAiComponentFactory {
    /** The way into voice mode, beside the message box. */
    @Composable
    override fun RowScope.ComposerTrailingContent(params: ComposerTrailingContentParams) {
        FilledIconButton(onClick = viewModel::talk) {
            Icon(Icons.Rounded.GraphicEq, contentDescription = "Talk to Juno")
        }
    }
}
