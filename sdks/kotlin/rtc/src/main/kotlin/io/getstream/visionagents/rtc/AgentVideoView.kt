package io.getstream.visionagents.rtc

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.getstream.video.android.compose.theme.VideoTheme
import io.getstream.video.android.compose.ui.components.call.renderer.MirrorMode
import io.getstream.video.android.compose.ui.components.call.renderer.ParticipantVideo
import io.getstream.video.android.core.Call
import io.getstream.video.android.core.ParticipantState

/**
 * The device camera, with the agent's annotated track inset beside it.
 *
 * There is no overlay drawing here: bounding boxes are already burned into the agent's published
 * track. That track has crossed the network to the processor and back, so it lags the camera by
 * a round trip and is inset rather than filling the frame. What fills the frame is this
 * device's own camera, which is immediate. A call where this device publishes nothing shows the
 * agent's track full-frame instead.
 */
@Composable
public fun AgentVideoView(voice: VoiceSession, modifier: Modifier = Modifier) {
    val call by voice.call.collectAsStateWithLifecycle()
    val joined = call
    if (joined == null) {
        WaitingForVideo(modifier)
    } else {
        VideoTheme { CallCanvas(joined, modifier) }
    }
}

@Composable
private fun CallCanvas(call: Call, modifier: Modifier) {
    val me by call.state.me.collectAsStateWithLifecycle()
    val cameraOn by call.camera.isEnabled.collectAsStateWithLifecycle()
    val remote by call.state.remoteParticipants.collectAsStateWithLifecycle()
    val agent = agentVideo(remote)
    // The camera being on is what this device decided, true from the moment it is turned on
    // rather than once a track has been negotiated and reported back.
    val local = me?.takeIf { cameraOn }

    Box(modifier.background(Color.Black)) {
        when {
            local != null -> {
                ParticipantVideo(call, local, Modifier.fillMaxSize(), labelContent = {}, mirrorMode = MirrorMode.AUTO)
                if (agent != null) {
                    ParticipantVideo(
                        call,
                        agent,
                        Modifier
                            .align(Alignment.BottomEnd)
                            .padding(16.dp)
                            .size(width = 180.dp, height = 240.dp)
                            .clip(RoundedCornerShape(12.dp)),
                        labelContent = {},
                        mirrorMode = MirrorMode.AUTO,
                    )
                }
            }
            agent != null -> ParticipantVideo(call, agent, Modifier.fillMaxSize(), labelContent = {}, mirrorMode = MirrorMode.AUTO)
            else -> WaitingForVideo(Modifier.fillMaxSize())
        }
    }
}

/**
 * The video worker joins as `{agent_user_id}-video`; anyone else with video is a fallback for a
 * call where the annotated track has not arrived yet.
 */
@Composable
private fun agentVideo(remote: List<ParticipantState>): ParticipantState? {
    val withVideo = remote.filter { it.videoEnabled.collectAsStateWithLifecycle().value }
    return withVideo.firstOrNull { it.userId.value.endsWith("-video") } ?: withVideo.firstOrNull()
}

@Composable
private fun WaitingForVideo(modifier: Modifier = Modifier) {
    Box(modifier.background(Color.Black), contentAlignment = Alignment.Center) {
        Text("waiting for video", style = MaterialTheme.typography.labelSmall, color = Color.White)
    }
}
