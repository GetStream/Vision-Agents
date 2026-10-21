package io.getstream.visionagents.android.demo

import android.Manifest
import android.content.pm.PackageManager
import android.os.SystemClock
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.CallEnd
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedCard
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.getstream.chat.android.ai.compose.parts.AIToolApproval
import io.getstream.chat.android.ai.compose.ui.component.AIToolApprovalCard
import io.getstream.chat.android.ai.compose.ui.component.AIToolApprovalState
import io.getstream.chat.android.ai.compose.ui.component.AIVoiceAgentState
import io.getstream.chat.android.ai.compose.ui.component.AIVoiceGlow
import io.getstream.chat.android.ai.compose.ui.component.AIVoiceVisualizationDefaults
import io.getstream.chat.android.ai.compose.ui.component.AIVoiceWave
import io.getstream.video.android.core.Call
import io.getstream.visionagents.android.core.AgentSession
import io.getstream.visionagents.android.core.AgentsException
import io.getstream.visionagents.android.core.Conversation
import io.getstream.visionagents.android.core.ToolApprovalRequest
import io.getstream.visionagents.android.core.VoiceSession
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * A spoken conversation, with the agent shown as Stream's voice glow and what is said captioned
 * underneath.
 *
 * The glow shows what the agent is doing: connecting while the call is joined, thinking while it
 * answers, and moving with whoever is talking. A tool that asks first asks here, over the call,
 * and the agent waits for the answer.
 */
@Composable
fun VoiceScreen(visual: VoiceVisual, modifier: Modifier = Modifier, viewModel: VoiceViewModel = viewModel()) {
    val voice by viewModel.voice.collectAsStateWithLifecycle()
    val starting by viewModel.isStarting.collectAsStateWithLifecycle()
    val failure by viewModel.failure.collectAsStateWithLifecycle()

    Column(modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        AgentPresence(voice, visual, isStarting = starting, modifier = Modifier.weight(1f))

        val current = voice
        if (current != null) {
            val approvals by current.session.approvals.collectAsStateWithLifecycle()
            approvals.firstOrNull()?.let { request ->
                // Keyed by the call, so a second question starts with no answer sending.
                key(request.id) {
                    ApprovalCard(request, current.session, Modifier.padding(horizontal = 16.dp))
                }
            }
            CallControls(current, onMute = viewModel::setMuted, onHangUp = viewModel::hangUp, modifier = Modifier.padding(bottom = 24.dp))
        } else {
            Talk(starting = starting, failure = failure, onTalk = viewModel::start, modifier = Modifier.padding(bottom = 24.dp))
        }
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
        modifier = modifier.fillMaxWidth().padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
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

/** How the agent is shown: Stream's voice glow, an orb, or its voice wave. */
enum class VoiceVisual(val label: String) { Orb("Orb"), Wave("Wave") }

/**
 * Settings for the Voice tab, at the top right: a sheet showing the orb and the wave, each moving
 * as it would on a call, to pick how the agent is shown.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VoiceVisualPicker(visual: VoiceVisual, onPick: (VoiceVisual) -> Unit) {
    var open by remember { mutableStateOf(false) }
    IconButton(onClick = { open = true }) {
        Icon(Icons.Outlined.Tune, contentDescription = "Voice style")
    }
    if (!open) return
    ModalBottomSheet(onDismissRequest = { open = false }) {
        Column(Modifier.fillMaxWidth().padding(start = 24.dp, end = 24.dp, bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Voice style", style = MaterialTheme.typography.titleLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                VoiceVisual.entries.forEach { option ->
                    VisualOption(
                        option,
                        selected = option == visual,
                        onClick = {
                            onPick(option)
                            open = false
                        },
                        modifier = Modifier.weight(1f),
                    )
                }
            }
        }
    }
}

/** One choice in the sheet: the visual itself, speaking, over its name. */
@Composable
private fun VisualOption(option: VoiceVisual, selected: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val border = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
    OutlinedCard(
        onClick = onClick,
        modifier = modifier,
        border = BorderStroke(if (selected) 2.dp else 1.dp, border),
    ) {
        Column(Modifier.fillMaxWidth().padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            val preview = Modifier.fillMaxWidth().aspectRatio(1f)
            when (option) {
                VoiceVisual.Orb -> AIVoiceGlow(state = AIVoiceAgentState.Speaking, modifier = preview)
                VoiceVisual.Wave -> AIVoiceWave(state = AIVoiceAgentState.Speaking, modifier = preview)
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                if (selected) Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
                Text(option.label, style = MaterialTheme.typography.titleSmall)
            }
        }
    }
}

/** What the glow shows during a call. */
private sealed interface Phase {
    data object Ready : Phase
    data object Listening : Phase
    data object Thinking : Phase
    data class Working(val skills: List<String>) : Phase
    data object Speaking : Phase
}

/**
 * The glow, moving with whoever is talking, a word for what the agent is doing, and the captions.
 *
 * While the agent speaks, the glow moves with its voice in Stream's colors; while the person
 * holding the phone speaks, with theirs in [PersonColors].
 *
 * The agent speaking comes from its audio, since a reply is written before it is spoken. The
 * person speaking comes from the call, which marks them speaking from the audio it sends, after
 * echo cancellation; the microphone here is heard before it, along with the agent coming out of
 * the speaker, so it only sets how much the glow moves. From the moment the person has been heard
 * until the agent's voice starts, the agent is thinking, including the gap after its reply is
 * written and before it is spoken. A phase is held for at least [MIN_PHASE_MILLIS], so events
 * arriving close together do not make the glow flicker.
 */
@Composable
private fun AgentPresence(voice: VoiceSession?, visual: VoiceVisual, isStarting: Boolean, modifier: Modifier = Modifier) {
    val call by remember(voice) { voice?.call ?: MutableStateFlow(null) }.collectAsStateWithLifecycle()

    val agentMeter = remember { VoiceMeter(audible = 0.04f) }
    val personMeter = remember { VoiceMeter(audible = 0.02f) }
    var phase by remember { mutableStateOf<Phase>(Phase.Ready) }
    // Read by the glow once a frame, so a new level redraws the glow and nothing else.
    val level = remember { mutableStateOf<Float?>(null) }
    val captions = remember(call) { Captions() }

    val active = call
    LaunchedEffect(active) { if (active != null) meter(active, agentMeter, personMeter) }
    LaunchedEffect(voice, active) {
        phase = Phase.Ready
        if (voice == null || active == null) return@LaunchedEffect
        var since = SystemClock.elapsedRealtime()
        var wanted: Phase = Phase.Ready
        var wantedSince = since
        var spokenTurn: String? = null
        var writtenTurn: String? = null
        var writtenAt = since
        var last = since
        while (isActive) {
            val now = SystemClock.elapsedRealtime()
            val agent = agentMeter.read()
            val person = personMeter.read()
            val current = voice.session.conversation.value
            val turn = current.turns.lastOrNull()
            if (turn?.isAgent == true && turn.id != writtenTurn) {
                writtenTurn = turn.id
                writtenAt = now
            }
            if (agent != null && turn?.isAgent == true) spokenTurn = turn.id

            val session = current.state
            val personSpeaking = active.state.me.value?.speaking?.value == true
            val target = when {
                agent != null -> Phase.Speaking
                personSpeaking || session is Conversation.State.Listening -> Phase.Listening
                session is Conversation.State.Working -> Phase.Working(session.skills)
                session is Conversation.State.Responding -> Phase.Thinking
                // Heard, and nothing written back yet.
                turn != null && !turn.isAgent -> Phase.Thinking
                // Written, and not spoken yet: the voice is still on its way.
                turn != null && turn.id != spokenTurn && now - writtenAt < AWAIT_VOICE_MILLIS -> Phase.Thinking
                else -> Phase.Ready
            }
            if (target != wanted) {
                wanted = target
                wantedSince = now
            }
            val settle = if (wanted == Phase.Ready) READY_SETTLE_MILLIS else SETTLE_MILLIS
            if (wanted != phase && now - since >= MIN_PHASE_MILLIS && now - wantedSince >= settle) {
                phase = wanted
                since = now
            }
            level.value = when (phase) {
                Phase.Speaking -> agent ?: 0f
                Phase.Listening -> person ?: 0f
                else -> null
            }

            captions.update(
                turn = turn,
                agentIsAudible = agent != null,
                agentIsAnswering = current.state is Conversation.State.Responding || current.state is Conversation.State.Working,
                now = now,
                elapsed = now - last,
            )
            last = now
            delay(33)
        }
    }

    val state = glowState(voice, isStarting, active, phase)
    val status = status(voice, isStarting, active, phase)

    Column(modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
        Spacer(Modifier.weight(1f))
        val colors = if (phase == Phase.Listening) PersonColors else AIVoiceVisualizationDefaults.colors()
        val size = Modifier.widthIn(max = 320.dp).fillMaxWidth().aspectRatio(1f)
        when (visual) {
            VoiceVisual.Orb -> AIVoiceGlow(state = state, colors = colors, audioLevel = { level.value }, modifier = size)
            VoiceVisual.Wave -> AIVoiceWave(state = state, colors = colors, audioLevel = { level.value }, modifier = size)
        }
        AnimatedContent(status, transitionSpec = { fadeIn(tween(200)) togetherWith fadeOut(tween(200)) }, label = "status") {
            // Nothing before a call, but the line keeps its height so the glow does not move.
            Text(it.ifEmpty { " " }, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.weight(1f))
        CaptionView(captions.caption, captions.isVisible)
    }
}

/**
 * Meters the call until it is left: this device's microphone as the call's audio filter, and the
 * agent, the call's other participant, through a sink on its audio track. The track can change
 * when the call resubscribes, so the sink follows whichever one is current.
 */
@OptIn(ExperimentalCoroutinesApi::class)
private suspend fun meter(call: Call, agentMeter: VoiceMeter, personMeter: VoiceMeter) {
    call.audioFilter = personMeter.filter
    var metered: org.webrtc.AudioTrack? = null
    try {
        call.state.remoteParticipants
            .flatMapLatest { participants ->
                if (participants.isEmpty()) {
                    flowOf(null)
                } else {
                    combine(participants.map { it.audioTrack }) { tracks -> tracks.firstNotNullOfOrNull { it?.audio } }
                }
            }
            .distinctUntilChanged()
            .collect { track ->
                metered?.let { runCatching { it.removeSink(agentMeter.sink) } }
                track?.addSink(agentMeter.sink)
                metered = track
            }
    } finally {
        // A track the call has already disposed refuses to be touched; it hears nothing anyway.
        metered?.let { runCatching { it.removeSink(agentMeter.sink) } }
        if (call.audioFilter === personMeter.filter) call.audioFilter = null
    }
}

private fun glowState(voice: VoiceSession?, isStarting: Boolean, call: Call?, phase: Phase): AIVoiceAgentState = when {
    voice == null -> if (isStarting) AIVoiceAgentState.Connecting else AIVoiceAgentState.Idle
    call == null -> AIVoiceAgentState.Connecting
    else -> when (phase) {
        Phase.Ready -> AIVoiceAgentState.Idle
        Phase.Listening -> AIVoiceAgentState.Listening
        Phase.Thinking, is Phase.Working -> AIVoiceAgentState.Thinking
        Phase.Speaking -> AIVoiceAgentState.Speaking
    }
}

/** What the glow is showing, in words. */
private fun status(voice: VoiceSession?, isStarting: Boolean, call: Call?, phase: Phase): String = when {
    voice == null -> if (isStarting) "Connecting" else ""
    call == null -> "Connecting"
    else -> when (phase) {
        Phase.Ready -> "Ready"
        Phase.Listening -> "Listening"
        Phase.Thinking -> "Thinking"
        is Phase.Working -> "Working on " + phase.skills.joinToString(" and ") { it.replace('_', ' ') }
        Phase.Speaking -> "Speaking"
    }
}

/** The least time a phase is shown before the glow moves on. */
private const val MIN_PHASE_MILLIS = 400L

/** How long a new phase has to hold before it is shown. */
private const val SETTLE_MILLIS = 150L

/** How long the line has to stay quiet before the glow goes back to ready. */
private const val READY_SETTLE_MILLIS = 800L

/** How long a written reply waits for its voice before the glow stops thinking. */
private const val AWAIT_VOICE_MILLIS = 10_000L

/** The colors the glow takes while the person is talking. The agent's are Stream's. */
private val PersonColors = listOf(Color.hsv(0f, 0.85f, 1f))

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

/**
 * Mute and hang up. Nothing here plays the agent: joining the call is what does that. The glow
 * shows the call being joined.
 */
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
        failure?.let { reason ->
            Text(
                reason.message.orEmpty(),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
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
