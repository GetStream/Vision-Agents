@file:Suppress("MagicNumber") // The card's layout, and two weeks of days.

package io.getstream.agents.androidassistant.components

import android.Manifest
import android.content.ContentUris
import android.content.Context
import android.provider.CalendarContract
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CalendarMonth
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLocale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.getstream.agents.androidassistant.Permissions
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponent
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponentDefinition
import io.getstream.chat.android.ai.openui.rendering.OpenUITheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.text.DateFormat
import java.text.ParseException
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/** The person's calendar for a plan's days. */
internal fun calendarCheck(definition: AIUIComponentDefinition) = AIUIComponent(definition) { arguments, _ ->
    val start = arguments.string("start")?.let(CalendarDay::parse) ?: return@AIUIComponent
    val end = arguments.string("end")?.let(CalendarDay::parse) ?: return@AIUIComponent
    CalendarCheck(start, end)
}

/**
 * The person's own calendar for the days of a plan, read on the phone. The agent never sees it.
 * The calendar is asked for the first time a card needs it.
 */
@Composable
private fun CalendarCheck(start: Date, end: Date) {
    val theme = OpenUITheme.current
    val context = LocalContext.current.applicationContext
    var days by remember { mutableStateOf<List<CalendarDay>>(emptyList()) }
    var failure by remember { mutableStateOf<String?>(null) }
    LaunchedEffect(start, end) {
        val loaded = CalendarDay.days(context, start, end)
        days = loaded.orEmpty()
        failure = if (loaded == null) "Let Juno see your calendar in Settings to check these days." else null
    }
    val locale = LocalLocale.current.platformLocale
    val dayFormat = remember(locale) { SimpleDateFormat(android.text.format.DateFormat.getBestDateTimePattern(locale, "EEEdMMM"), locale) }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(theme.colors.subtleFill, RoundedCornerShape(theme.radius.large))
            .padding(12.dp)
            .animateContentSize(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Icon(Icons.Outlined.CalendarMonth, contentDescription = null, tint = theme.colors.accent, modifier = Modifier.size(18.dp))
            Text("Your calendar", style = theme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold), color = theme.colors.textPrimary)
        }
        failure?.let { Text(it, style = theme.typography.caption, color = theme.colors.textSecondary) }
        days.forEach { day ->
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    dayFormat.format(day.date),
                    style = theme.typography.caption.copy(fontWeight = FontWeight.Medium),
                    color = theme.colors.textPrimary,
                    modifier = Modifier.width(96.dp),
                )
                if (day.events.isEmpty()) {
                    Text("Free", style = theme.typography.caption, color = theme.colors.success)
                } else {
                    Text(day.events.joinToString(" · "), style = theme.typography.caption, color = theme.colors.textPrimary)
                }
            }
        }
        Text("Only you see this. Juno doesn't.", style = theme.typography.caption, color = theme.colors.textTertiary)
    }
}

/** One day of the person's calendar: what is on it, or nothing. */
private class CalendarDay(val date: Date, val events: List<String>) {
    companion object {
        private const val MAX_DAYS = 14
        private const val DAY_MILLIS = 24 * 60 * 60 * 1000L

        private fun isoDay(): SimpleDateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US).apply { isLenient = false }

        /** A day as YYYY-MM-DD, at its start in the person's timezone, or null when it is not one. */
        fun parse(text: String): Date? = try {
            isoDay().parse(text)
        } catch (_: ParseException) {
            null
        }

        /**
         * The days from [start] to [end], at most two weeks of them, or null when the person does
         * not let the app see their calendar.
         */
        suspend fun days(context: Context, start: Date, end: Date): List<CalendarDay>? {
            if (!Permissions.allOf(context, Manifest.permission.READ_CALENDAR)) return null
            return withContext(Dispatchers.IO) {
                val starts = mutableListOf<Long>()
                val day = Calendar.getInstance().apply { time = start }
                while (!day.time.after(maxOf(start, end)) && starts.size < MAX_DAYS) {
                    starts += day.timeInMillis
                    day.add(Calendar.DAY_OF_MONTH, 1)
                }
                val last = day.timeInMillis
                val events = events(context, starts.first(), last)
                starts.mapIndexed { index, from ->
                    val to = starts.getOrNull(index + 1) ?: last
                    CalendarDay(Date(from), events.filter { it.isOn(from, to) }.map { it.text })
                }
            }
        }

        private class Event(val title: String, val begin: Long, val end: Long, val allDay: Boolean) {
            private val time = DateFormat.getTimeInstance(DateFormat.SHORT)

            val text: String get() = if (allDay) title else "${time.format(Date(begin))} $title"

            /** Whether it is on the day from [from] to [to]. An all-day event's times are midnights in UTC. */
            fun isOn(from: Long, to: Long): Boolean {
                if (!allDay) return begin < to && end > from
                val local = isoDay()
                val utc = isoDay().apply { timeZone = TimeZone.getTimeZone("UTC") }
                val day = local.format(Date(from))
                return utc.format(Date(begin)) <= day && day < utc.format(Date(end))
            }
        }

        private fun events(context: Context, from: Long, to: Long): List<Event> {
            // All-day events are stored in UTC, so a day either side catches those of the first and last days.
            val uri = CalendarContract.Instances.CONTENT_URI.buildUpon().also {
                ContentUris.appendId(it, from - DAY_MILLIS)
                ContentUris.appendId(it, to + DAY_MILLIS)
            }.build()
            val columns = arrayOf(
                CalendarContract.Instances.TITLE,
                CalendarContract.Instances.BEGIN,
                CalendarContract.Instances.END,
                CalendarContract.Instances.ALL_DAY,
            )
            val events = mutableListOf<Event>()
            context.contentResolver.query(uri, columns, null, null, "${CalendarContract.Instances.BEGIN} ASC")?.use { rows ->
                while (rows.moveToNext()) {
                    val title = rows.getString(0)?.ifEmpty { null } ?: "Busy"
                    events += Event(title, rows.getLong(1), rows.getLong(2), rows.getInt(3) != 0)
                }
            }
            return events
        }
    }
}
