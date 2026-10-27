package io.getstream.visionagents.androidassistant

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.EditNote
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.getstream.visionagents.android.core.Session
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * The drawer: a new conversation, a search, and the conversations there were, by day.
 *
 * [selected] is the conversation open beside this, or null for a new one. Holding a
 * conversation offers to delete it.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversationList(
    conversations: Conversations,
    selected: String?,
    onNew: () -> Unit,
    onOpen: (Session) -> Unit,
    onDelete: (Session) -> Unit,
    modifier: Modifier = Modifier,
) {
    val items by conversations.items.collectAsStateWithLifecycle()
    val searchText by conversations.searchText.collectAsStateWithLifecycle()
    val failure by conversations.failure.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    val focus = LocalFocusManager.current
    var refreshing by remember { mutableStateOf(false) }

    LaunchedEffect(searchText) {
        // Waits for typing to pause before searching.
        if (searchText.isNotEmpty()) delay(300)
        conversations.reload()
    }

    Column(modifier.fillMaxSize()) {
        NavigationDrawerItem(
            label = { Text("New chat", style = MaterialTheme.typography.titleMedium) },
            icon = { Icon(Icons.Outlined.EditNote, contentDescription = null) },
            selected = false,
            onClick = onNew,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
        )

        OutlinedTextField(
            value = searchText,
            onValueChange = conversations::search,
            placeholder = { Text("Search conversations") },
            leadingIcon = { Icon(Icons.Outlined.Search, contentDescription = null) },
            singleLine = true,
            shape = CircleShape,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
            keyboardActions = KeyboardActions(onSearch = { focus.clearFocus() }),
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
        )

        PullToRefreshBox(
            isRefreshing = refreshing,
            onRefresh = {
                scope.launch {
                    refreshing = true
                    conversations.reload()
                    refreshing = false
                }
            },
            modifier = Modifier.weight(1f),
        ) {
            LazyColumn(Modifier.fillMaxSize().padding(top = 8.dp)) {
                days(items).forEach { day ->
                    item(key = "day:${day.id}") {
                        Text(
                            day.title,
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(start = 28.dp, end = 28.dp, top = 16.dp, bottom = 4.dp),
                        )
                    }
                    items(day.items, key = { it.id }) { session ->
                        ConversationRow(
                            session = session,
                            isSelected = session.id == selected,
                            onOpen = { onOpen(session) },
                            onDelete = { onDelete(session) },
                        )
                    }
                }
                failure?.let {
                    item(key = "failure") {
                        Text(
                            it,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                            modifier = Modifier.padding(horizontal = 28.dp, vertical = 8.dp),
                        )
                    }
                }
            }
        }
    }
}

/** One conversation in the list: open it with a tap, or hold it to delete it. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ConversationRow(session: Session, isSelected: Boolean, onOpen: () -> Unit, onDelete: () -> Unit) {
    var isMenuShown by remember { mutableStateOf(false) }
    Box(Modifier.padding(horizontal = 12.dp)) {
        Surface(
            shape = CircleShape,
            color = if (isSelected) MaterialTheme.colorScheme.secondaryContainer else Color.Transparent,
            modifier = Modifier
                .fillMaxWidth()
                .clip(CircleShape)
                .combinedClickable(onClick = onOpen, onLongClick = { isMenuShown = true }),
        ) {
            Text(
                session.displayTitle,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 14.dp),
            )
        }
        DropdownMenu(expanded = isMenuShown, onDismissRequest = { isMenuShown = false }) {
            DropdownMenuItem(
                text = { Text("Delete") },
                leadingIcon = { Icon(Icons.Outlined.Delete, contentDescription = null) },
                onClick = {
                    isMenuShown = false
                    onDelete()
                },
            )
        }
    }
}

/**
 * A run of conversations under one heading. A heading can appear twice: the list is in order of
 * activity, and the days are when each conversation started, which is all a [Session] says.
 */
private class Day(val title: String, val items: List<Session>) {
    val id: String get() = items.first().id
}

/** The conversations under Today, Yesterday and then their week, in the router's order. */
private fun days(sessions: List<Session>): List<Day> {
    val days = mutableListOf<Day>()
    for (session in sessions) {
        val title = dayTitle(Date(session.createdAt.toEpochMilliseconds()))
        val last = days.lastOrNull()
        if (last?.title == title) {
            days[days.lastIndex] = Day(title, last.items + session)
        } else {
            days += Day(title, listOf(session))
        }
    }
    return days
}

private fun dayTitle(date: Date): String {
    val today = startOfDay(Date())
    val day = startOfDay(date)
    // Rounded, since a day the clocks change on is an hour longer or shorter.
    val daysAgo = Math.round((today.timeInMillis - day.timeInMillis) / DAY_MILLIS.toDouble()).toInt()
    return when {
        daysAgo <= 0 -> "Today"
        daysAgo == 1 -> "Yesterday"
        daysAgo < 7 -> "This week"
        else -> SimpleDateFormat("LLLL yyyy", Locale.getDefault()).format(date)
    }
}

private fun startOfDay(date: Date): Calendar = Calendar.getInstance().apply {
    time = date
    set(Calendar.HOUR_OF_DAY, 0)
    set(Calendar.MINUTE, 0)
    set(Calendar.SECOND, 0)
    set(Calendar.MILLISECOND, 0)
}

private const val DAY_MILLIS = 24 * 60 * 60 * 1000L
