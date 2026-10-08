package io.getstream.visionagents.rtc

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.launch

/**
 * The controls for a spoken conversation: mute, camera, and hang up.
 *
 * It draws no transcript. Showing what was said is the ui module's job, over the same
 * [VoiceSession.session], so a host can put the two together however it likes. There is
 * nothing here for playing the agent: joining the call is what does that.
 *
 * @param credentials asked for the token this device joins with. Minting one is server-side
 *   only, so it comes from the app's own backend.
 */
@Composable
public fun VoiceCallView(
    voice: VoiceSession,
    credentials: CallCredentialsProvider,
    modifier: Modifier = Modifier,
    camera: Boolean = false,
) {
    val call by voice.call.collectAsStateWithLifecycle()
    val muted by voice.isMuted.collectAsStateWithLifecycle()
    val cameraOn by voice.isCameraEnabled.collectAsStateWithLifecycle()
    val failure by voice.failure.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()

    LaunchedEffect(voice) { voice.join(camera, credentials) }

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        val problem = failure
        if (problem != null) {
            Text(
                problem.message ?: problem.toString(),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.error,
            )
        } else if (call == null) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                CircularProgressIndicator(modifier = Modifier.size(12.dp), strokeWidth = 1.5.dp)
                Text("joining", style = MaterialTheme.typography.labelMedium)
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
            FilledTonalButton(onClick = { voice.setMuted(!muted) }, enabled = call != null) {
                Text(if (muted) "Unmute" else "Mute")
            }
            if (camera) {
                FilledTonalButton(onClick = { voice.setCameraEnabled(!cameraOn) }, enabled = call != null) {
                    Text(if (cameraOn) "Camera off" else "Camera on")
                }
            }
            Button(
                onClick = { scope.launch { voice.end() } },
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
            ) {
                Text("End")
            }
        }
    }
}
