package io.getstream.visionagents.android.demo

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Key
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

/**
 * Talks to the agent in writing.
 *
 * There is no agent picker. Reading the configs is server-side only, so a device cannot ask
 * which agents exist; it is told which one it talks to, the way a real app would be, in
 * [Demo.AGENT_NAME].
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DemoApp() {
    Scaffold(
        topBar = { CenterAlignedTopAppBar(title = { Text("Larkspur support") }) },
    ) { padding ->
        if (Demo.streamApiKey.isEmpty()) {
            NoKey(Modifier.padding(padding))
        } else {
            ChatScreen(Modifier.padding(padding))
        }
    }
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
