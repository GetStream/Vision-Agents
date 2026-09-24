package io.getstream.visionagents.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import io.getstream.visionagents.core.Turn

/**
 * The conversation, scrolling as it grows.
 *
 * Keyed by turn id, so a reply streaming in recomposes only its own row. It brings no navigation
 * and no theme of its own, so it drops into whatever the host has.
 *
 * @param turns the conversation, oldest first.
 * @param bubble how to draw one line. Omit it for [TurnBubble].
 */
@Composable
public fun TranscriptView(
    turns: List<Turn>,
    modifier: Modifier = Modifier,
    bubble: @Composable (Turn) -> Unit = { TurnBubble(it) },
) {
    val list = rememberLazyListState()
    val last = turns.lastOrNull()
    // Growing text follows the bottom without animating; animating each delta is what makes a
    // transcript judder while it is being written. A whole new line is worth animating to.
    LaunchedEffect(turns.size) {
        if (turns.isNotEmpty()) list.animateScrollToItem(turns.lastIndex)
    }
    LaunchedEffect(last?.text) {
        if (turns.isNotEmpty()) list.scrollToItem(turns.lastIndex, Int.MAX_VALUE)
    }
    LazyColumn(
        modifier = modifier,
        state = list,
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        items(turns, key = { it.id }) { bubble(it) }
    }
}

/** One line of the conversation. */
@Composable
public fun TurnBubble(turn: Turn, modifier: Modifier = Modifier) {
    val colours = MaterialTheme.colorScheme
    Column(
        modifier = modifier.fillMaxWidth(),
        horizontalAlignment = if (turn.isAgent) Alignment.Start else Alignment.End,
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        val speaker = turn.speaker
        val name = (speaker as? Turn.Speaker.Person)?.participant?.display.orEmpty()
        if (name.isNotEmpty()) {
            Text(name, style = MaterialTheme.typography.labelSmall, color = colours.onSurfaceVariant)
        }
        SelectionContainer {
            Text(
                turn.text,
                color = if (turn.isAgent) colours.onSurface else colours.onPrimary,
                modifier = Modifier
                    .background(
                        if (turn.isAgent) colours.surfaceVariant else colours.primary,
                        RoundedCornerShape(16.dp),
                    )
                    .padding(horizontal = 12.dp, vertical = 8.dp),
            )
        }
    }
}
