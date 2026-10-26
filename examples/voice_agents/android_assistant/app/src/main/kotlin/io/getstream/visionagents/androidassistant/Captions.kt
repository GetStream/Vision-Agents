package io.getstream.visionagents.androidassistant

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.dp
import io.getstream.visionagents.android.core.Turn
import kotlin.math.min

/**
 * Closed captions for a spoken conversation: what is being said, gone a moment after the talking
 * stops.
 *
 * The agent's reply is written before it is spoken, so its words are revealed at the pace of its
 * voice, counting the time its audio is heard, rather than as they arrive. What the person said
 * appears once it has been transcribed.
 */
class Captions {
    data class Caption(val text: String, val isPerson: Boolean)

    // Compose state, so the caption band redraws when these change and nothing else does.
    var caption: Caption? by mutableStateOf(null)
        private set
    var isVisible: Boolean by mutableStateOf(false)
        private set

    private var turnId: String? = null
    private var spokenMillis = 0L
    private var lastAudible = Long.MIN_VALUE / 2
    private var lastChange = Long.MIN_VALUE / 2

    fun update(turn: Turn?, agentIsAudible: Boolean, agentIsAnswering: Boolean, now: Long, elapsed: Long) {
        val next = when {
            turn == null -> null
            turn.isAgent -> {
                if (turn.id != turnId) {
                    turnId = turn.id
                    spokenMillis = 0
                }
                if (agentIsAudible) {
                    spokenMillis += elapsed
                    lastAudible = now
                }
                // Once the agent has stopped for good, whatever the pacing has not reached yet is
                // shown, so a reply never ends half captioned.
                val isOver = spokenMillis > 0 && !agentIsAnswering && now - lastAudible > FINISHED_MILLIS
                val count = if (isOver) turn.text.length else min(turn.text.length, (spokenMillis * PACE / 1000).toInt())
                said(turn.text, count).takeIf { it.isNotEmpty() }?.let { Caption(it, isPerson = false) }
            }
            else -> said(turn.text, turn.text.length).takeIf { it.isNotEmpty() }?.let { Caption(it, isPerson = true) }
        }
        if (next != null && next != caption) {
            caption = next
            lastChange = now
        }
        isVisible = caption != null && (agentIsAudible || now - lastChange < LINGER_MILLIS)
    }

    companion object {
        /** About how fast the agent speaks, in characters a second. */
        private const val PACE = 15

        /** How long a caption stays up once nothing new is being said. */
        private const val LINGER_MILLIS = 2_500L

        /** How long the agent has to be quiet, with its reply written, for the reply to be over. */
        private const val FINISHED_MILLIS = 1_500L

        /** What has been said once the first [count] characters of [text] have been, in whole words. */
        fun said(text: String, count: Int): String {
            var said = text.take(count)
            if (count < text.length) {
                // Hold back a word until all of it has been said.
                val space = said.indexOfLast { it.isWhitespace() }
                said = if (space >= 0) said.substring(0, space) else ""
            }
            return said.trim()
        }
    }
}

/**
 * A caption, white on a dark band at the bottom of the screen like a film's. What the person said
 * is in italics.
 *
 * It shows two lines at a time, as subtitles do: the text is laid out at the band's width and
 * paged in twos, and the page holding the newest words is the one shown. Five lines are shown as
 * two, two more, then the last on its own.
 */
@Composable
fun CaptionView(caption: Captions.Caption?, isVisible: Boolean, modifier: Modifier = Modifier) {
    val alpha by animateFloatAsState(if (isVisible) 1f else 0f, tween(350), label = "caption")
    val style = MaterialTheme.typography.bodyLarge.copy(
        fontWeight = FontWeight.Medium,
        fontStyle = if (caption?.isPerson == true) FontStyle.Italic else FontStyle.Normal,
        color = Color.White,
        textAlign = TextAlign.Center,
    )
    val measurer = rememberTextMeasurer()
    // Two lines' room whether or not a caption is up, so the glow never moves.
    BoxWithConstraints(modifier.fillMaxWidth().height(72.dp).padding(horizontal = 16.dp), contentAlignment = Alignment.BottomCenter) {
        val width = constraints.maxWidth - with(LocalDensity.current) { BAND_PADDING.roundToPx() * 2 }
        val page = remember(caption?.text, style, width) {
            caption?.text?.let { page(measurer.measure(it, style, constraints = Constraints(maxWidth = width.coerceAtLeast(0))), it) }
        }
        if (!page.isNullOrEmpty()) {
            Text(
                page,
                style = style,
                maxLines = 2,
                modifier = Modifier
                    .alpha(alpha)
                    .background(Color.Black.copy(alpha = 0.7f), RoundedCornerShape(10.dp))
                    .padding(horizontal = BAND_PADDING, vertical = 8.dp),
            )
        }
    }
}

/** The last page of two lines of [text], as [layout] wraps it. */
private fun page(layout: TextLayoutResult, text: String): String {
    val first = (layout.lineCount - 1) / 2 * 2
    return text.substring(layout.getLineStart(first)).trim()
}

private val BAND_PADDING = 14.dp
