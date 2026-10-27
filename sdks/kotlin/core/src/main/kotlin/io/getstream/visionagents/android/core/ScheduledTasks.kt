package io.getstream.visionagents.android.core

import io.getstream.visionagents.android.core.generated.ScheduledTask as TaskSchema
import io.getstream.visionagents.android.core.generated.ScheduledTaskNext
import io.getstream.visionagents.android.core.generated.ScheduledTaskPage
import io.getstream.visionagents.android.core.generated.ScheduledTaskQuietHours
import io.getstream.visionagents.android.core.generated.ScheduledTaskRun as RunSchema
import io.getstream.visionagents.android.core.generated.ScheduledTaskRunPage
import io.getstream.visionagents.android.core.generated.ScheduledTaskSchedule
import io.getstream.visionagents.android.core.generated.UpdateScheduledTaskRequest
import io.ktor.http.HttpMethod
import kotlin.time.Instant
import kotlinx.serialization.json.JsonObject

/**
 * The reminders and later work agents hold for the user [VisionAgents.setUser] named.
 *
 * The agent makes a task in a conversation, when the person asks for one, and the app's backend
 * can make one in their name. A device cannot: what the agent is told when a task comes due is
 * the backend's to write. What is here is what a person does with a reminder they already have:
 * find it, move it, pause it and cancel it.
 *
 *     val page = agents.scheduledTasks.list()
 *     agents.scheduledTasks.update(page.items[0].id, next = ScheduledTask.Next.After("PT15M"))
 */
public class ScheduledTasks internal constructor(private val backend: Backend) {
    /**
     * One page of this user's tasks, newest first, the ones their backend made included. Pass
     * the page's [Page.nextCursor] as [ScheduledTaskQuery.cursor] for the next one.
     */
    public suspend fun list(query: ScheduledTaskQuery = ScheduledTaskQuery()): Page<ScheduledTask> {
        val page = backend.get(path(), ScheduledTaskPage.serializer(), query.parameters())
        return Page(page.items.map(ScheduledTask::of), page.hasMore, page.nextCursor?.ifEmpty { null })
    }

    /** One task. Somebody else's is reported as not found. */
    public suspend fun get(id: String): ScheduledTask =
        ScheduledTask.of(backend.get(path(id), TaskSchema.serializer()))

    /**
     * Changes a task and returns it as it now is. Null leaves a field as it is.
     *
     * [schedule] replaces the whole schedule, but for a timezone it leaves out. [next] moves only
     * the next occurrence: a snooze, or a skip when it lands past the one after. Either lands 15
     * minutes from now at the soonest, unless it is no earlier than the current next occurrence.
     * An empty [deliverTo] is the conversation alone, and [quietHours] with both ends empty clears
     * the hold.
     */
    public suspend fun update(
        id: String,
        title: String? = null,
        schedule: ScheduledTask.Schedule? = null,
        next: ScheduledTask.Next? = null,
        deliverTo: List<String>? = null,
        quietHours: ScheduledTask.QuietHours? = null,
    ): ScheduledTask {
        val request = UpdateScheduledTaskRequest(
            deliverTo = deliverTo,
            next = next?.schema,
            quietHours = quietHours?.schema,
            schedule = schedule?.schema,
            title = title,
        )
        val updated = backend.send(
            HttpMethod.Patch,
            path(id),
            TaskSchema.serializer(),
            body = wire.encodeToString(UpdateScheduledTaskRequest.serializer(), request),
        )!!
        return ScheduledTask.of(updated)
    }

    /**
     * Holds a task: nothing comes due while it is paused, and what it slept through is never
     * replayed.
     */
    public suspend fun pause(id: String): ScheduledTask =
        ScheduledTask.of(backend.send(HttpMethod.Post, path(id, "pause"), TaskSchema.serializer())!!)

    /**
     * Starts a paused or failed task again, counting its next occurrence from now. A one-shot
     * whose time came while it was paused is completed instead, and a task with no occurrence
     * left needs a new schedule first.
     */
    public suspend fun resume(id: String): ScheduledTask =
        ScheduledTask.of(backend.send(HttpMethod.Post, path(id, "resume"), TaskSchema.serializer())!!)

    /** Cancels a task: it and its runs are gone. */
    public suspend fun delete(id: String) {
        backend.send<Unit>(HttpMethod.Delete, path(id), answer = null)
    }

    /**
     * One page of the times a task came due, newest first. Snoozes and pauses write none. Pass
     * the page's [Page.nextCursor] as [cursor] for the next one.
     */
    public suspend fun runs(id: String, limit: Int? = null, cursor: String? = null): Page<ScheduledTaskRun> {
        val page = backend.get(
            path(id, "runs"),
            ScheduledTaskRunPage.serializer(),
            queryOf("limit" to limit, "cursor" to cursor),
        )
        return Page(page.items.map(ScheduledTaskRun::of), page.hasMore, page.nextCursor?.ifEmpty { null })
    }

    private fun path(vararg rest: String) = listOf("v1", "agents", "scheduled-tasks") + rest
}

/** Which of this user's tasks to list. Null leaves a filter out. */
public data class ScheduledTaskQuery(
    /** Only the ones in this state. */
    val state: ScheduledTask.State? = null,
    /** Only the ones on this agent config. */
    val configId: String? = null,
    /** Only the ones following up in this conversation. */
    val sessionId: String? = null,
    /** Up to 200. Omitted is 25. */
    val limit: Int? = null,
    /** The [Page.nextCursor] of the page before, asked with the same filters. Omitted is the first page. */
    val cursor: String? = null,
) {
    internal fun parameters(): List<Pair<String, String>> = queryOf(
        "limit" to limit,
        "cursor" to cursor,
        "state" to when (state) {
            ScheduledTask.State.Unknown -> throw AgentsException.Configuration("an unknown state cannot be asked for")
            else -> state?.name?.lowercase()
        },
        "config_id" to configId,
        "session_id" to sessionId,
    )
}

/** Something an agent will do for the user later: a reminder, a check-in, a recurring report. */
public data class ScheduledTask(
    val id: String,
    /** A short name the user will recognise. */
    val title: String,
    /** What the agent is told when the task comes due. */
    val instructions: String,
    val schedule: Schedule,
    /**
     * Where an occurrence goes, in order, the first that lands: `conversation`, `sms`,
     * `whatsapp`, `imessage` or `phone`.
     */
    val deliverTo: List<String>,
    /** The part of the day an occurrence is held through, or null for no hold. */
    val quietHours: QuietHours?,
    val state: State,
    /** Null when there is none left. */
    val nextRunAt: Instant?,
    /** Up to the next five occurrences. */
    val upcoming: List<Instant>,
    /** When a run last finished, or null when none has. */
    val lastRunAt: Instant?,
    /** Failed occurrences in a row. Five fail the task. */
    val consecutiveFailures: Int,
    val createdBy: CreatedBy,
    /** The conversation the task follows up in. Empty until its first occurrence opens one. */
    val sessionId: String,
    val configId: String,
    /** Anything the backend that made the task kept on it. */
    val custom: JsonObject,
    val createdAt: Instant,
    val updatedAt: Instant,
) {
    /** Whether a task will come due. */
    public enum class State {
        Active,
        Paused,
        /** A one-shot that ran, or a repeat past its end. */
        Completed,
        /** Paused after five failed occurrences in a row. [ScheduledTasks.resume] clears it. */
        Failed,
        /** A state this SDK has never heard of. */
        Unknown,
    }

    /** Who made a task. */
    public enum class CreatedBy {
        /** The model, in a conversation. */
        Agent,
        /** The app's backend. */
        Api,
        /** A maker this SDK has never heard of. */
        Unknown,
    }

    /** When a task comes due. On a change, exactly one of [at] and [after] is given. */
    public data class Schedule(
        /**
         * When it first runs as the user's clock reads it, `2026-10-20T09:00:00`. With an offset
         * or `Z` it is that instant.
         */
        val at: String? = null,
        /** How long from now instead, an ISO 8601 duration: `PT20M`, `P1D`. Never read back. */
        val after: String? = null,
        /**
         * Null for once. Otherwise an RRULE without DTSTART or UNTIL: `FREQ=DAILY`,
         * `FREQ=WEEKLY;BYDAY=MO,WE,FR`.
         */
        val repeat: String? = null,
        /** An IANA zone, such as `Europe/Amsterdam`. Null keeps the task's own. */
        val timezone: String? = null,
        /** The last local date it repeats on, inclusive: `2026-12-31`. */
        val until: String? = null,
        /** What the schedule says in one sentence, written by the router and never sent. */
        val description: String = "",
    ) {
        internal val schema: ScheduledTaskSchedule
            get() = ScheduledTaskSchedule(at = at, `in` = after, repeat = repeat, timezone = timezone, until = until)

        internal companion object {
            fun of(schema: ScheduledTaskSchedule): Schedule = Schedule(
                at = schema.at,
                after = schema.`in`,
                repeat = schema.repeat,
                timezone = schema.timezone,
                until = schema.until,
                description = schema.description.orEmpty(),
            )
        }
    }

    /** Where the next occurrence moves to. */
    public sealed interface Next {
        /** As the user's clock reads it, in the task's zone: `2026-10-20T09:00:00`. */
        public data class At(val at: String) : Next

        /** How long from now, an ISO 8601 duration: `PT15M`. */
        public data class After(val duration: String) : Next
    }

    /**
     * The part of the user's day an occurrence that comes due in is held through, as their clock
     * reads it. It may run past midnight: `22:00` to `07:00` holds the night.
     */
    public data class QuietHours(val start: String, val end: String) {
        internal val schema: ScheduledTaskQuietHours
            get() = ScheduledTaskQuietHours(end = end, start = start)
    }

    internal companion object {
        fun of(schema: TaskSchema): ScheduledTask = ScheduledTask(
            id = schema.id,
            title = schema.title,
            instructions = schema.instructions,
            schedule = Schedule.of(schema.schedule),
            deliverTo = schema.deliverTo,
            quietHours = schema.quietHours?.let { QuietHours(it.start, it.end) },
            state = when (schema.state) {
                TaskSchema.State.active -> State.Active
                TaskSchema.State.paused -> State.Paused
                TaskSchema.State.completed -> State.Completed
                TaskSchema.State.failed -> State.Failed
                else -> State.Unknown
            },
            nextRunAt = instantOrNull(schema.nextRunAt),
            upcoming = schema.upcoming.map(::instant),
            lastRunAt = instantOrNull(schema.lastRunAt),
            consecutiveFailures = schema.consecutiveFailures.toInt(),
            createdBy = when (schema.createdBy) {
                TaskSchema.CreatedBy.ScheduledTaskCreatedByAgent -> CreatedBy.Agent
                TaskSchema.CreatedBy.ScheduledTaskCreatedByApi -> CreatedBy.Api
                else -> CreatedBy.Unknown
            },
            sessionId = schema.sessionId,
            configId = schema.configId,
            custom = JsonObject(schema.custom.orEmpty()),
            createdAt = instant(schema.createdAt),
            updatedAt = instant(schema.updatedAt),
        )
    }
}

private val ScheduledTask.Next.schema: ScheduledTaskNext
    get() = when (this) {
        is ScheduledTask.Next.At -> ScheduledTaskNext(at = at)
        is ScheduledTask.Next.After -> ScheduledTaskNext(`in` = duration)
    }

/** One time a task came due, or was run by hand. */
public data class ScheduledTaskRun(
    val id: String,
    val taskId: String,
    /** When the occurrence was due. */
    val scheduledFor: Instant,
    val trigger: Trigger,
    val state: State,
    /**
     * Where the follow-up landed: `conversation`, `sms`, `whatsapp`, `imessage` or `phone`.
     * Empty until it lands.
     */
    val deliveredTo: String,
    val sessionId: String,
    val attempts: Int,
    val startedAt: Instant?,
    val finishedAt: Instant?,
    /**
     * Why it failed or was missed, as a short code such as `delivery_failed` or `task_paused`.
     * Empty otherwise.
     */
    val error: String,
) {
    /** What set a run going. */
    public enum class Trigger {
        /** It came due. */
        Schedule,
        /** It was run by hand. */
        Manual,
        /** A trigger this SDK has never heard of. */
        Unknown,
    }

    /** How a run went. */
    public enum class State {
        /** Retries included. */
        Running,
        Delivered,
        Failed,
        /** Too late to be worth delivering, or due while the task was paused. */
        Missed,
        /** A state this SDK has never heard of. */
        Unknown,
    }

    internal companion object {
        fun of(schema: RunSchema): ScheduledTaskRun = ScheduledTaskRun(
            id = schema.id,
            taskId = schema.taskId,
            scheduledFor = instant(schema.scheduledFor),
            trigger = when (schema.trigger) {
                RunSchema.Trigger.schedule -> Trigger.Schedule
                RunSchema.Trigger.manual -> Trigger.Manual
                else -> Trigger.Unknown
            },
            state = when (schema.state) {
                RunSchema.State.running -> State.Running
                RunSchema.State.delivered -> State.Delivered
                RunSchema.State.failed -> State.Failed
                RunSchema.State.missed -> State.Missed
                else -> State.Unknown
            },
            deliveredTo = schema.deliveredTo,
            sessionId = schema.sessionId,
            attempts = schema.attempts.toInt(),
            startedAt = instantOrNull(schema.startedAt),
            finishedAt = instantOrNull(schema.finishedAt),
            error = schema.error.orEmpty(),
        )
    }
}
