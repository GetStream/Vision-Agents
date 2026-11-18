@file:Suppress("MagicNumber") // The cards' layout, and the dial's.

package io.getstream.agents.androidassistant.components

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalLocale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponent
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponentDefinition
import io.getstream.chat.android.ai.openui.rendering.OpenUITheme
import kotlin.math.roundToInt

/** A restaurant's card, which asks for its menu when tapped. */
internal fun restaurantCard(definition: AIUIComponentDefinition) = AIUIComponent(definition) { arguments, context ->
    val name = arguments.string("name") ?: return@AIUIComponent
    val cuisine = arguments.string("cuisine") ?: return@AIUIComponent
    val rating = arguments.number("rating") ?: return@AIUIComponent
    RestaurantCard(name, cuisine, rating, arguments.string("price_tier")) {
        context.emit("inspect_venue", label = "Show me the menu and highlights for $name", payload = mapOf("name" to name))
    }
}

/** How busy a venue is right now, and the wait. */
internal fun availabilityGauge(definition: AIUIComponentDefinition) = AIUIComponent(definition) { arguments, _ ->
    val occupancy = arguments.number("occupancy_pct") ?: return@AIUIComponent
    val wait = arguments.string("wait_label") ?: return@AIUIComponent
    AvailabilityGauge(occupancy, wait)
}

/** A venue's tables by zone, which asks to reserve the one tapped. */
internal fun seatSelector(definition: AIUIComponentDefinition) = AIUIComponent(definition) { arguments, context ->
    val tables = arguments.objects("tables").mapNotNull(DiningTable::from)
    if (tables.isEmpty()) return@AIUIComponent
    SeatSelector(tables) { table ->
        context.emit(
            "select_table",
            label = "Reserve table ${table.id} (${table.zone} for ${table.seats})",
            payload = mapOf("id" to table.id, "zone" to table.zone, "seats" to table.seats),
        )
    }
}

/** A restaurant, its cuisine and its rating. Tapping it asks for its menu and highlights. */
@Composable
private fun RestaurantCard(name: String, cuisine: String, rating: Double, priceTier: String?, onInspect: () -> Unit) {
    val theme = OpenUITheme.current
    val locale = LocalLocale.current.platformLocale
    val badge = "★ " + String.format(locale, "%.1f", rating) + priceTier?.let { " · $it" }.orEmpty()
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(theme.radius.large))
            .background(theme.colors.subtleFill)
            .clickable(onClickLabel = "Ask Juno for its menu and highlights", onClick = onInspect)
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(name, style = theme.typography.heading, color = theme.colors.textPrimary, modifier = Modifier.weight(1f))
            Text(badge, style = theme.typography.caption.copy(fontWeight = FontWeight.SemiBold), color = theme.colors.accent)
        }
        Text(cuisine, style = theme.typography.bodySmall, color = theme.colors.textSecondary)
    }
}

/** How full a venue is, as a half dial in the brand's accent, with the wait under it. */
@Composable
private fun AvailabilityGauge(occupancy: Double, wait: String) {
    val theme = OpenUITheme.current
    val percent = occupancy.coerceIn(0.0, 100.0)
    val fraction by animateFloatAsState((percent / 100).toFloat(), tween(400), label = "occupancy")
    val rounded = percent.roundToInt()
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(theme.colors.subtleFill, RoundedCornerShape(theme.radius.large))
            .padding(12.dp)
            .clearAndSetSemantics { contentDescription = "$rounded percent full, $wait" },
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Box(Modifier.size(width = 140.dp, height = 78.dp), contentAlignment = Alignment.BottomCenter) {
            HalfDial(1f, theme.colors.sunkenSurface)
            HalfDial(fraction, theme.colors.accent)
            AnimatedContent(rounded, label = "percent") { value ->
                Text("$value%", style = theme.typography.metric, color = theme.colors.textPrimary)
            }
        }
        Text(wait, style = theme.typography.caption, color = theme.colors.textSecondary)
    }
}

/**
 * The top half of a circle standing on the bottom of its box, drawn from the left for [fraction]
 * of the way round.
 */
@Composable
private fun HalfDial(fraction: Float, color: Color) {
    Canvas(Modifier.fillMaxSize()) {
        val line = 10.dp.toPx()
        val radius = minOf(size.width / 2 - 8.dp.toPx(), size.height - 12.dp.toPx())
        val center = Offset(size.width / 2, size.height - 6.dp.toPx())
        drawArc(
            color = color,
            startAngle = 180f,
            sweepAngle = 180f * fraction,
            useCenter = false,
            topLeft = Offset(center.x - radius, center.y - radius),
            size = Size(radius * 2, radius * 2),
            style = Stroke(width = line, cap = StrokeCap.Round),
        )
    }
}

private class DiningTable(val id: String, val zone: String, val seats: Int, val available: Boolean) {
    companion object {
        fun from(json: Map<*, *>): DiningTable? {
            val id = (json["id"] as? String)?.takeIf { it.isNotEmpty() } ?: return null
            val zone = json["zone"] as? String ?: return null
            val seats = json.number("seats")?.roundToInt() ?: return null
            val available = json["available"] as? Boolean ?: return null
            return DiningTable(id, zone, seats, available)
        }
    }
}

/**
 * A venue's tables, by zone and seats. Tapping a free one chooses it and asks to reserve it; a
 * taken one can't be tapped. The first free one is chosen until the person picks another.
 */
@Composable
private fun SeatSelector(tables: List<DiningTable>, onSelect: (DiningTable) -> Unit) {
    val theme = OpenUITheme.current
    var picked by remember { mutableStateOf<String?>(null) }
    val selected = picked ?: tables.firstOrNull { it.available }?.id
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(theme.colors.subtleFill, RoundedCornerShape(theme.radius.large))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("Select a table zone", style = theme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold), color = theme.colors.textPrimary)
        // As many columns as fit at 150dp or more each.
        BoxWithConstraints(Modifier.fillMaxWidth()) {
            val columns = ((maxWidth + 8.dp) / (150.dp + 8.dp)).toInt().coerceAtLeast(1)
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                tables.chunked(columns).forEach { row ->
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        row.forEach { table ->
                            Table(table, isSelected = table.id == selected, modifier = Modifier.weight(1f)) {
                                picked = table.id
                                onSelect(table)
                            }
                        }
                        repeat(columns - row.size) { Spacer(Modifier.weight(1f)) }
                    }
                }
            }
        }
    }
}

@Composable
private fun Table(table: DiningTable, isSelected: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val theme = OpenUITheme.current
    val shape = RoundedCornerShape(theme.radius.small)
    Text(
        "${table.id} · ${table.zone} (${table.seats}p)",
        style = theme.typography.caption.copy(
            fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal,
            textDecoration = if (table.available) null else TextDecoration.LineThrough,
        ),
        color = if (table.available) theme.colors.textPrimary else theme.colors.textTertiary,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
        modifier = modifier
            .clip(shape)
            .background(if (isSelected) theme.colors.accent.copy(alpha = 0.16f) else theme.colors.sunkenSurface)
            .border(1.dp, if (isSelected) theme.colors.accent else Color.Transparent, shape)
            .selectable(selected = isSelected, enabled = table.available, role = Role.RadioButton, onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 7.dp),
    )
}
