package io.getstream.visionagents.android.demo

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.CallEnd
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.getstream.chat.android.ai.compose.parts.AIToolApproval
import io.getstream.chat.android.ai.compose.ui.component.AIToolApprovalCard
import io.getstream.chat.android.ai.compose.ui.component.AIToolApprovalState
import io.getstream.chat.android.ai.compose.ui.component.AITypingIndicator
import io.getstream.chat.android.ai.compose.ui.component.StreamingText
import io.getstream.visionagents.android.core.AgentSession
import io.getstream.visionagents.android.core.AgentsException
import io.getstream.visionagents.android.core.Conversation
import io.getstream.visionagents.android.core.ToolApprovalRequest
import io.getstream.visionagents.android.core.VoiceSession
import kotlinx.coroutines.launch

/** A spoken conversation, with what was said written out as it happens. */
@Composable
fun VoiceScreen(modifier: Modifier = Modifier, viewModel: VoiceViewModel = viewModel()) {
    val voice by viewModel.voice.collectAsStateWithLifecycle()
    val starting by viewModel.isStarting.collectAsStateWithLifecycle()
    val failure by viewModel.failure.collectAsStateWithLifecycle()

    val current = voice
    if (current != null) {
        OnCall(current, onMute = viewModel::setMuted, onHangUp = viewModel::hangUp, modifier = modifier)
    } else {
        Talk(starting = starting, failure = failure, onTalk = viewModel::start, modifier = modifier)
    }
}

/** The way into a call: the microphone, asked for first, and then the agent. */
@Composable
private fun Talk(starting: Boolean, failure: String?, onTalk: () -> Unit, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var refused by remember { mutableStateOf(false) }
    val microphone = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        refused = !granted
        if (granted) onTalk()
    }

    Column(
        modifier = modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        val reason = if (refused) "Talking to the agent needs the microphone." else failure
        reason?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center)
        }
        Button(
            enabled = !starting,
            onClick = {
                val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                    PackageManager.PERMISSION_GRANTED
                if (granted) onTalk() else microphone.launch(Manifest.permission.RECORD_AUDIO)
            },
            contentPadding = PaddingValues(horizontal = 24.dp, vertical = 12.dp),
        ) {
            Icon(Icons.Filled.Call, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.size(8.dp))
            Text("Talk to the agent")
        }
    }
}

@Composable
private fun OnCall(
    voice: VoiceSession,
    onMute: (Boolean) -> Unit,
    onHangUp: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val approvals by voice.session.approvals.collectAsStateWithLifecycle()

    Column(modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        SpokenTranscript(voice.session, Modifier.weight(1f))
        approvals.firstOrNull()?.let { request ->
            // Keyed by the call, so a second question starts with no answer sending.
            key(request.id) {
                ApprovalCard(request, voice.session, Modifier.padding(horizontal = 16.dp))
            }
        }
        CallControls(voice, onMute, onHangUp, Modifier.padding(bottom = 24.dp))
    }
}

/**
 * What was said on the call, the agent's words streaming in as they are written.
 *
 * Read from the top, as a transcript is, and kept scrolled to the newest line as it grows.
 */
@Composable
private fun SpokenTranscript(session: AgentSession, modifier: Modifier = Modifier) {
    val conversation by session.conversation.collectAsStateWithLifecycle()
    val turns = conversation.turns
    val status = conversation.status
    val list = rememberLazyListState()
    val lines = turns.size + if (status != null) 1 else 0
    LaunchedEffect(lines, turns.lastOrNull()?.text?.length) {
        if (lines > 0) list.animateScrollToItem(lines - 1)
    }

    LazyColumn(
        state = list,
        modifier = modifier.fillMaxWidth(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        items(turns, key = { it.id }) { turn ->
            if (turn.isAgent) {
                val writing = turn.id == turns.lastOrNull()?.id && conversation.state == Conversation.State.Responding
                StreamingText(text = turn.text, animate = writing)
            } else {
                Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.CenterEnd) {
                    Text(
                        turn.text,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier
                            .widthIn(max = 320.dp)
                            .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.15f), RoundedCornerShape(18.dp))
                            .padding(horizontal = 14.dp, vertical = 10.dp),
                    )
                }
            }
        }
        status?.let { text ->
            item(key = "status") {
                AITypingIndicator(label = {
                    Text(text, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                })
            }
        }
    }
}

/** What the agent is doing while there is nothing of it to read yet. */
private val Conversation.status: String?
    get() = when (val current = state) {
        Conversation.State.Listening -> "Listening"
        Conversation.State.Responding ->
            if (turns.lastOrNull()?.let { !it.isAgent || it.text.isEmpty() } != false) "Thinking" else null
        is Conversation.State.Working ->
            "Working on " + current.skills.joinToString(" and ") { it.replace('_', ' ') }
        else -> null
    }

/**
 * A tool's question, asked over the call with Stream's approval card. The tool runs only once it
 * is allowed, and the agent is told when it is not.
 */
@Composable
private fun ApprovalCard(request: ToolApprovalRequest, session: AgentSession, modifier: Modifier = Modifier) {
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf(AIToolApprovalState()) }
    AIToolApprovalCard(
        approval = AIToolApproval(
            title = request.approval.title,
            message = request.approval.message,
            reason = request.reason.ifEmpty { null },
            allowTitle = request.approval.allowTitle,
            declineTitle = request.approval.declineTitle,
        ),
        state = state,
        decide = { allowed ->
            state = AIToolApprovalState(isSending = true)
            scope.launch {
                try {
                    session.decide(request.id, allowed)
                } catch (_: AgentsException) {
                    state = AIToolApprovalState(failed = true)
                }
            }
        },
        modifier = modifier,
    )
}

/** Mute and hang up. Nothing here plays the agent: joining the call is what does that. */
@Composable
private fun CallControls(
    voice: VoiceSession,
    onMute: (Boolean) -> Unit,
    onHangUp: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val call by voice.call.collectAsStateWithLifecycle()
    val muted by voice.isMuted.collectAsStateWithLifecycle()
    val failure by voice.failure.collectAsStateWithLifecycle()

    Column(modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
        val reason = failure
        when {
            reason != null -> Text(
                reason.message.orEmpty(),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
            call == null -> Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(12.dp), strokeWidth = 1.5.dp)
                Text("joining", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
            FilledTonalIconButton(
                onClick = { onMute(!muted) },
                enabled = call != null,
                modifier = Modifier.size(56.dp),
            ) {
                Icon(
                    if (muted) Icons.Filled.MicOff else Icons.Filled.Mic,
                    contentDescription = if (muted) "Unmute" else "Mute",
                )
            }
            FilledIconButton(
                onClick = onHangUp,
                modifier = Modifier.size(56.dp),
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = MaterialTheme.colorScheme.error,
                    contentColor = MaterialTheme.colorScheme.onError,
                ),
            ) {
                Icon(Icons.Filled.CallEnd, contentDescription = "Hang up")
            }
        }
    }
}
