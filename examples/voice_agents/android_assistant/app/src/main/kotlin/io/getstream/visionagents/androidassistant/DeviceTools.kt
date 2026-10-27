package io.getstream.visionagents.androidassistant

import android.Manifest
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.location.Address
import android.location.Geocoder
import android.location.Location
import android.location.LocationManager
import android.os.Build
import android.os.CancellationSignal
import android.provider.CalendarContract
import androidx.core.location.LocationManagerCompat
import io.getstream.visionagents.android.core.AgentTool
import java.text.DateFormat
import java.text.ParseException
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject

/**
 * What the agent can ask this phone: the time and place, the calendar and reminders, and to send
 * an email from the person's Gmail.
 *
 * The agent runs in the backend and knows none of this. Each tool runs here, when the model calls
 * it, and only its answer leaves the phone. The ones that share or do something personal ask
 * first: the call waits in the session's `approvals` until the person answers, on the reply's step
 * in a chat or on a card over a call. A permission the phone needs is asked for the first time a
 * tool needs it.
 */
class DeviceTools(context: Context) {
    private val context = context.applicationContext

    val all: List<AgentTool> get() = listOf(localContext, location, events, reminder, sendEmail)

    /**
     * The date, time and timezone, which a model in a data centre has no way to know.
     *
     * It takes no arguments, and still declares an empty object of them: a tool sent with no
     * parameters at all is refused by xAI, which fails the whole reply.
     */
    private val localContext = AgentTool(
        name = "get_local_context",
        description = "The person's current date, local time, timezone and language. Call it " +
            "before answering anything that depends on today or the time.",
        parameters = AgentTool.strings(emptyMap()),
    ) {
        val now = Date()
        val zone = TimeZone.getDefault()
        val day = SimpleDateFormat("EEEE d MMMM yyyy", Locale.getDefault()).format(now)
        val time = DateFormat.getTimeInstance(DateFormat.SHORT).format(now)
        val abbreviation = zone.getDisplayName(zone.inDaylightTime(now), TimeZone.SHORT)
        "It is $day, $time in ${zone.id} ($abbreviation). " +
            "Today as a date is ${isoDay().format(now)}. " +
            "Their language is ${Locale.getDefault()}."
    }

    /** Where the person is, to the street. It asks first, since it is where they are. */
    private val location = AgentTool(
        name = "get_location",
        description = "Where the person is: their street, neighbourhood and city. Only call it " +
            "when the answer depends on where they are, such as the weather or places nearby.",
        parameters = AgentTool.strings(
            mapOf("reason" to "why you need it, in a few words they would recognise"),
            required = listOf("reason"),
        ),
        displayTitle = "Checking where you are",
        approval = AgentTool.Approval(
            title = "Share your location?",
            message = "Juno gets your street and city, once.",
            reasonArgument = "reason",
            allowTitle = "Share",
            declineTitle = "Not now",
        ),
    ) {
        val allowed = Permissions.anyOf(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
        )
        if (!allowed) throw ToolFailure("They have not let Juno see where they are. It is in Settings.")
        val here = currentLocation() ?: throw ToolFailure("The phone could not tell where it is just now.")
        val place = address(here)
            ?: return@AgentTool "They are at latitude ${here.latitude}, longitude ${here.longitude}, " +
                "which has no address."
        val parts = listOfNotNull(place.thoroughfare, place.subLocality, place.locality, place.countryName)
        "They are on " + parts.joinToString(", ") + "."
    }

    /** What is on the calendar for one day. */
    private val events = AgentTool(
        name = "list_events",
        description = "The events on the person's calendar for one day.",
        parameters = AgentTool.strings(
            mapOf("day" to "the day as YYYY-MM-DD, in their timezone"),
            required = listOf("day"),
        ),
        displayTitle = "Checking your calendar",
    ) { arguments ->
        val start = try {
            isoDay().parse(arguments.string("day"))
        } catch (_: ParseException) {
            null
        } ?: throw ToolFailure("The day has to be a date like 2026-10-16.")
        if (!Permissions.allOf(context, Manifest.permission.READ_CALENDAR)) {
            throw ToolFailure("They have not let Juno see their calendar. It is in Settings.")
        }
        val found = eventsOn(start)
        if (found.isEmpty()) return@AgentTool "Nothing is on their calendar that day."
        found.joinToString("\n")
    }

    /**
     * Adds a reminder, once the person says it may: an event in their calendar at the time it is
     * due, with an alert then. Android has no Reminders of its own for an app to add to, and an
     * alert on a calendar event is what fires at a time on every phone.
     */
    private val reminder = AgentTool(
        name = "add_reminder",
        description = "Adds a reminder to the person's calendar, with an alert when it is due.",
        parameters = buildJsonObject {
            put("type", "object")
            putJsonObject("properties") {
                putJsonObject("title") {
                    put("type", "string")
                    put("description", "what to remind them of, such as Call mum")
                }
                putJsonObject("due") {
                    put("type", "string")
                    put(
                        "description",
                        "when, as an ISO 8601 date and time with their UTC offset, such as " +
                            "2026-10-16T18:00:00+02:00.",
                    )
                }
                putJsonObject("summary") {
                    put("type", "string")
                    put(
                        "description",
                        "the reminder and its time as they would read it, such as Call mum, today at 6pm",
                    )
                }
            }
            putJsonArray("required") {
                add("title")
                add("due")
                add("summary")
            }
        },
        displayTitle = "Adding a reminder",
        approval = AgentTool.Approval(
            title = "Add this reminder?",
            message = "It goes in your calendar, with an alert when it is due.",
            reasonArgument = "summary",
            allowTitle = "Add reminder",
            declineTitle = "Not now",
        ),
    ) { arguments ->
        val title = arguments.string("title")
        if (title.isEmpty()) throw ToolFailure("A reminder needs a title.")
        val due = parseInstant(arguments.string("due"))
            ?: throw ToolFailure("A reminder needs a due time, such as 2026-10-16T18:00:00+02:00.")
        val allowed = Permissions.allOf(context, Manifest.permission.READ_CALENDAR, Manifest.permission.WRITE_CALENDAR)
        if (!allowed) throw ToolFailure("They have not let Juno add reminders. It is in Settings.")
        addReminder(title, due)
        "Added the reminder \"$title\"."
    }

    /**
     * Sends an email from the person's Gmail, once they say it may.
     *
     * Google's Gmail MCP server writes drafts but cannot send, so sending is a tool here: the
     * approval is asked on the phone, and the backend sends through the router's proxy for the
     * person's Gmail connection, which holds the credential. The question shows who it goes to,
     * read from the call itself rather than from anything the model wrote about it.
     */
    private val sendEmail = AgentTool(
        name = "send_email",
        description = "Sends an email from the person's Gmail. Only call it once they have asked " +
            "for it to be sent, after you have shown them the subject and the body.",
        parameters = AgentTool.strings(
            mapOf(
                "to" to "who it is to: one or more email addresses, separated by commas",
                "subject" to "the subject line",
                "body" to "the email itself, in plain text",
            ),
            required = listOf("to", "subject", "body"),
        ),
        displayTitle = "Sending your email",
        approval = AgentTool.Approval(
            title = "Send this email?",
            message = "Juno sends it from your Gmail, as you.",
            reasonArgument = "to",
            allowTitle = "Send",
            declineTitle = "Not now",
        ),
    ) { arguments ->
        val to = arguments.string("to")
        val subject = arguments.string("subject")
        val body = arguments.string("body")
        if (to.isEmpty() || body.isEmpty()) throw ToolFailure("An email needs someone to go to and a body.")
        Gmail.send(to = to, subject = subject, body = body)
        "Sent \"$subject\" to $to from their Gmail."
    }

    /** One reading of where the phone is, or null when none came within 15 seconds. */
    @Suppress("MissingPermission") // Held: the tool asks for it first.
    private suspend fun currentLocation(): Location? {
        val manager = context.getSystemService(LocationManager::class.java)
        val provider = when {
            Build.VERSION.SDK_INT >= 31 && manager.hasProvider(LocationManager.FUSED_PROVIDER) ->
                LocationManager.FUSED_PROVIDER
            manager.isProviderEnabled(LocationManager.NETWORK_PROVIDER) -> LocationManager.NETWORK_PROVIDER
            manager.isProviderEnabled(LocationManager.GPS_PROVIDER) -> LocationManager.GPS_PROVIDER
            else -> return null
        }
        return withTimeoutOrNull(LOCATION_TIMEOUT_MILLIS) {
            suspendCancellableCoroutine { done ->
                val cancel = CancellationSignal()
                done.invokeOnCancellation { cancel.cancel() }
                LocationManagerCompat.getCurrentLocation(manager, provider, cancel, executor) { done.resume(it) }
            }
        }
    }

    /** The address at a place, or null where there is none or the phone has no geocoder. */
    private suspend fun address(location: Location): Address? {
        if (!Geocoder.isPresent()) return null
        val geocoder = Geocoder(context, Locale.getDefault())
        if (Build.VERSION.SDK_INT >= 33) {
            return suspendCancellableCoroutine { done ->
                geocoder.getFromLocation(
                    location.latitude,
                    location.longitude,
                    1,
                    object : Geocoder.GeocodeListener {
                        override fun onGeocode(addresses: MutableList<Address>) = done.resume(addresses.firstOrNull())
                        override fun onError(errorMessage: String?) = done.resume(null)
                    },
                )
            }
        }
        return withContext(Dispatchers.IO) {
            try {
                @Suppress("DEPRECATION") // The listener above is Android 13's.
                geocoder.getFromLocation(location.latitude, location.longitude, 1)?.firstOrNull()
            } catch (_: java.io.IOException) {
                null
            }
        }
    }

    /**
     * The events on the day that starts at [start], each as a line: its times or All day, its
     * title, and where. An all-day event is stored on its day in UTC, so it is matched by the
     * date rather than the hours.
     */
    private suspend fun eventsOn(start: Date): List<String> = withContext(Dispatchers.IO) {
        val end = Calendar.getInstance().apply {
            time = start
            add(Calendar.DAY_OF_MONTH, 1)
        }.timeInMillis
        val day = isoDay().format(start)
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
        lines
    }

    /**
     * Puts the reminder in the calendar new events go in: the primary one the person can write
     * to, or else the first. It lasts [REMINDER_MINUTES], and alerts as it starts.
     */
    private suspend fun addReminder(title: String, due: Date) = withContext(Dispatchers.IO) {
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

    private companion object {
        const val LOCATION_TIMEOUT_MILLIS = 15_000L
        const val REMINDER_MINUTES = 15
        const val DAY_MILLIS = 24 * 60 * 60 * 1000L

        /** Where the location's answer is delivered; the coroutine waiting for it resumes on its own. */
        val executor = Executors.newSingleThreadExecutor()

        /** A day as YYYY-MM-DD, in the person's timezone. */
        fun isoDay(): SimpleDateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US).apply { isLenient = false }

        /** An ISO 8601 date and time with its UTC offset, or null when it is not one. */
        fun parseInstant(text: String): Date? = listOf("yyyy-MM-dd'T'HH:mm:ssXXX", "yyyy-MM-dd'T'HH:mmXXX")
            .firstNotNullOfOrNull { pattern ->
                try {
                    SimpleDateFormat(pattern, Locale.US).apply { isLenient = false }.parse(text)
                } catch (_: ParseException) {
                    null
                }
            }

        fun JsonObject.string(key: String): String = this[key]?.jsonPrimitive?.contentOrNull.orEmpty()
    }
}

/** Why a tool could not answer, in words the model passes on. */
class ToolFailure(message: String) : Exception(message)
