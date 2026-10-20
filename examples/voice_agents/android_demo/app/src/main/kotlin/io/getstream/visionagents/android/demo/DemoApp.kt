package io.getstream.visionagents.android.demo

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Chat
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.outlined.Key
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

/**
 * Talks to the agent, in writing or out loud.
 *
 * There is no agent picker. Reading the configs is server-side only, so a device cannot ask
 * which agents exist; it is told which one it talks to, the way a real app would be, in
 * [Demo.AGENT_NAME].
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DemoApp() {
    var tab by rememberSaveable { mutableStateOf(Tab.Chat) }
    val hasKey = Demo.streamApiKey.isNotEmpty()
    Scaffold(
        topBar = { CenterAlignedTopAppBar(title = { Text("Larkspur support") }) },
        bottomBar = {
            if (hasKey) {
                NavigationBar {
                    Tab.entries.forEach {
                        NavigationBarItem(
                            selected = tab == it,
                            onClick = { tab = it },
                            icon = { Icon(it.icon, contentDescription = null) },
                            label = { Text(it.label) },
                        )
                    }
                }
            }
        },
    ) { padding ->
        val content = Modifier.padding(padding).consumeWindowInsets(padding)
        when {
            !hasKey -> NoKey(content)
            // Each keeps its conversation while the other is shown: a call carries on under Chat.
            tab == Tab.Chat -> ChatScreen(content)
            else -> VoiceScreen(content)
        }
    }
}

private enum class Tab(val label: String, val icon: ImageVector) {
    Chat("Chat", Icons.AutoMirrored.Outlined.Chat),
    Voice("Voice", Icons.Rounded.GraphicEq),
}

@Composable
private fun NoKey(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Outlined.Key,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(56.dp),
        )
        Text("No Stream key yet", style = MaterialTheme.typography.titleLarge)
        Text(
            "Put the STREAM_API_KEY the router runs with in local.properties, as demo.streamApiKey.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}
