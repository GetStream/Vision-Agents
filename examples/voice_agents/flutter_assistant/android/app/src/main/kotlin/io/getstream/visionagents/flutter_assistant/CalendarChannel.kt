package io.getstream.visionagents.flutter_assistant

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.provider.CalendarContract
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.text.DateFormat
import java.text.ParseException
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.Executors

/**
 * The calendar, for the agent's list_events and add_reminder tools, as android_assistant's
 * tools use it. Android has no Reminders of its own for an app to add to, so a reminder is an
 * event in the calendar new events go in, at the time it is due, with an alert then.
 *
 * The app asks for the calendar before calling. What the phone could not do comes back as a
 * `tool_failure` error whose message the model reads.
 */
class CalendarChannel(context: Context) {
    private val context = context.applicationContext
    private val main = Handler(Looper.getMainLooper())

    fun handle(call: MethodCall, result: MethodChannel.Result) {
        executor.execute {
            try {
                val answer: Any? = when (call.method) {
                    "listEvents" -> eventsOn(call.argument<String>("day").orEmpty())
                    "addReminder" -> {
                        val due = call.argument<Number>("due")?.toLong()
                            ?: throw ToolFailure("A reminder needs a due time, such as 2026-10-16T18:00:00+02:00.")
                        addReminder(call.argument<String>("title").orEmpty(), Date(due))
                        null
                    }
                    else -> {
                        main.post { result.notImplemented() }
                        return@execute
                    }
                }
                main.post { result.success(answer) }
            } catch (failure: ToolFailure) {
                main.post { result.error("tool_failure", failure.message, null) }
            } catch (failure: SecurityException) {
                main.post {
                    result.error("tool_failure", "They have not let Juno use their calendar. It is in Settings.", null)
                }
            } catch (failure: Exception) {
                // Anything else the calendar provider throws still answers the call, or the
                // reply would wait on it until the router gives up.
                main.post { result.error("tool_failure", "The phone could not read its calendar just now.", null) }
            }
        }
    }

    /**
     * The events on [day], each as a line: its times or All day, its title, and where. An
     * all-day event is stored on its day in UTC, so it is matched by the date rather than the
     * hours.
     */
    private fun eventsOn(day: String): List<String> {
        val start = try {
            isoDay().parse(day)
        } catch (_: ParseException) {
            null
        } ?: throw ToolFailure("The day has to be a date like 2026-10-16.")
        val end = Calendar.getInstance().apply {
            time = start
            add(Calendar.DAY_OF_MONTH, 1)
        }.timeInMillis
        val utcDay = isoDay().apply { timeZone = TimeZone.getTimeZone("UTC") }
        val time = DateFormat.getTimeInstance(DateFormat.SHORT)
        val uri = CalendarContract.Instances.CONTENT_URI.buildUpon().also {
            ContentUris.appendId(it, start.time - DAY_MILLIS)
            ContentUris.appendId(it, end + DAY_MILLIS)
        }.build()
        val columns = arrayOf(
            CalendarContract.Instances.TITLE,
            CalendarContract.Instances.BEGIN,
            CalendarContract.Instances.END,
            CalendarContract.Instances.ALL_DAY,
            CalendarContract.Instances.EVENT_LOCATION,
        )
        val lines = mutableListOf<String>()
        context.contentResolver.query(uri, columns, null, null, "${CalendarContract.Instances.BEGIN} ASC")?.use { rows ->
            while (rows.moveToNext()) {
                val begin = rows.getLong(1)
                val finish = rows.getLong(2)
                val allDay = rows.getInt(3) != 0
                val onDay = if (allDay) utcDay.format(Date(begin)) == day else begin < end && finish > start.time
                if (!onDay) continue
                val title = rows.getString(0)?.ifEmpty { null } ?: "Busy"
                val place = rows.getString(4)?.ifEmpty { null }?.let { " at $it" }.orEmpty()
                val whenText = if (allDay) "All day" else "${time.format(Date(begin))}–${time.format(Date(finish))}"
                lines += "$whenText: $title$place"
            }
        }
        return lines
    }

    /**
     * Puts the reminder in the calendar new events go in: the primary one the person can write
     * to, or else the first. It lasts [REMINDER_MINUTES], and alerts as it starts.
     */
    private fun addReminder(title: String, due: Date) {
        val resolver = context.contentResolver
        val calendar = resolver.query(
            CalendarContract.Calendars.CONTENT_URI,
            arrayOf(CalendarContract.Calendars._ID),
            "${CalendarContract.Calendars.CALENDAR_ACCESS_LEVEL} >= ? AND ${CalendarContract.Calendars.VISIBLE} = 1",
            arrayOf(CalendarContract.Calendars.CAL_ACCESS_CONTRIBUTOR.toString()),
            "${CalendarContract.Calendars.IS_PRIMARY} DESC",
        )?.use { rows -> if (rows.moveToFirst()) rows.getLong(0) else null }
            ?: throw ToolFailure("There is no calendar on the phone to add a reminder to.")
        val event = ContentValues().apply {
            put(CalendarContract.Events.CALENDAR_ID, calendar)
            put(CalendarContract.Events.TITLE, title)
            put(CalendarContract.Events.DTSTART, due.time)
            put(CalendarContract.Events.DTEND, due.time + REMINDER_MINUTES * 60_000L)
            put(CalendarContract.Events.EVENT_TIMEZONE, TimeZone.getDefault().id)
            put(CalendarContract.Events.HAS_ALARM, 1)
        }
        val added = resolver.insert(CalendarContract.Events.CONTENT_URI, event)
            ?: throw ToolFailure("The calendar would not take the reminder.")
        val alert = ContentValues().apply {
            put(CalendarContract.Reminders.EVENT_ID, ContentUris.parseId(added))
            put(CalendarContract.Reminders.MINUTES, 0)
            put(CalendarContract.Reminders.METHOD, CalendarContract.Reminders.METHOD_ALERT)
        }
        resolver.insert(CalendarContract.Reminders.CONTENT_URI, alert)
    }

    companion object {
        const val NAME = "io.getstream.visionagents.flutterassistant/calendar"
        private const val REMINDER_MINUTES = 15
        private const val DAY_MILLIS = 24 * 60 * 60 * 1000L

        /** Where the calendar is read and written, off the main thread. */
        private val executor = Executors.newSingleThreadExecutor()

        /** A day as YYYY-MM-DD, in the person's timezone. */
        private fun isoDay(): SimpleDateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US).apply { isLenient = false }
    }
}

/** Why a tool could not answer, in words the model passes on. */
class ToolFailure(message: String) : Exception(message)
