package io.getstream.visionagents.androidassistant

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Key
import androidx.compose.material.icons.outlined.Menu
import androidx.compose.material.icons.outlined.WifiOff
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch

/**
 * The conversation in front, and the ones there were in a drawer that slides over it.
 *
 * There is no agent picker. Reading the configs is server-side only, so a phone cannot ask which
 * agents exist; it is told which one it talks to, in [Assistant.AGENT_NAME].
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AssistantApp(chat: ChatViewModel = viewModel(), conversations: Conversations = viewModel()) {
    if (Assistant.streamApiKey.isEmpty()) {
        Notice(
            Icons.Outlined.Key,
            "No Stream key yet",
            "Put the STREAM_API_KEY the router runs with in local.properties, as assistant.streamApiKey.",
        )
        return
    }
    when (rememberLocalNetwork()) {
        LocalNetwork.Asking -> return
        LocalNetwork.Refused -> {
            Notice(
                Icons.Outlined.WifiOff,
                "No local network access",
                "The router runs on your Mac, which the app reaches over the local network. " +
                    "Allow it in the app's settings.",
            )
            return
        }
        LocalNetwork.Allowed -> Unit
    }

    val drawer = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val focus = LocalFocusManager.current
    val selected by chat.selected.collectAsStateWithLifecycle()
    val screen by chat.screen.collectAsStateWithLifecycle()
    val items by conversations.items.collectAsStateWithLifecycle()
    val title = items.firstOrNull { it.id == selected }?.displayTitle ?: "Juno"

    LaunchedEffect(drawer) {
        // The router titles a conversation as it goes, so the list is read again each time it is
        // shown. The drawer slides over the composer without ending its editing, so the keyboard
        // is put away here.
        snapshotFlow { drawer.targetValue }.collect {
            if (it == DrawerValue.Open) {
                focus.clearFocus()
                conversations.reload()
            }
        }
    }

    fun show(sessionId: String?) {
        chat.show(sessionId)
        scope.launch { drawer.close() }
    }

    val voiceShown by chat.isVoiceShown.collectAsStateWithLifecycle()
    val open by chat.chat.collectAsStateWithLifecycle()

    ModalNavigationDrawer(
        drawerState = drawer,
        // Voice mode is over everything, the drawer's edge included.
        gesturesEnabled = !voiceShown,
        drawerContent = {
            ModalDrawerSheet {
                ConversationList(
                    conversations = conversations,
                    selected = selected,
                    onNew = { show(null) },
                    onOpen = { show(it.id) },
                    onDelete = {
                        conversations.delete(it)
                        if (it.id == selected) chat.show(null)
                    },
                )
            }
        },
    ) {
        Scaffold(
            topBar = {
                CenterAlignedTopAppBar(
                    title = { Text(title, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawer.open() } }) {
                            Icon(Icons.Outlined.Menu, contentDescription = "Conversations")
                        }
                    },
                )
            },
            // The conversation runs on under the navigation bar and the keyboard, which the
            // composer floating over it keeps clear of itself.
            contentWindowInsets = WindowInsets(0),
        ) { padding ->
            key(screen) {
                ChatScreen(chat, Modifier.padding(padding).consumeWindowInsets(padding))
            }
        }
    }
    val talking = open
    if (voiceShown && talking != null) VoiceMode(chat, talking)
}

private enum class LocalNetwork { Asking, Allowed, Refused }

/**
 * Asks for the local network, which an app targeting Android 17 needs to reach the Mac at its
 * address on the Wi-Fi, before a connection that would otherwise time out rather than fail.
 * Through adb reverse the router and the backend are on the phone's own localhost, which needs
 * no asking.
 */
@Composable
private fun rememberLocalNetwork(): LocalNetwork {
    // Android 17, which compileSdk 37 knows as a number.
    val onLoopback = listOf(Assistant.routerUrl, Assistant.backendUrl).all {
        Uri.parse(it).host in setOf("localhost", "127.0.0.1")
    }
    if (Build.VERSION.SDK_INT < 37 || onLoopback) return LocalNetwork.Allowed
    val context = LocalContext.current
    val permission = Manifest.permission.ACCESS_LOCAL_NETWORK
    var state by remember {
        mutableStateOf(
            if (context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED) {
                LocalNetwork.Allowed
            } else {
                LocalNetwork.Asking
            },
        )
    }
    val ask = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        state = if (granted) LocalNetwork.Allowed else LocalNetwork.Refused
    }
    LaunchedEffect(Unit) {
        if (state == LocalNetwork.Asking) ask.launch(permission)
    }
    return state
}

@Composable
private fun Notice(icon: ImageVector, title: String, message: String) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(56.dp),
        )
        Text(title, style = MaterialTheme.typography.titleLarge)
        Text(
            message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}
