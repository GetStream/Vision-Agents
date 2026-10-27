import Foundation
import StreamCore

/// The reminders and later work agents hold for the user `setUser` named.
///
/// The agent makes a task in a conversation, when the person asks for one, and the app's
/// backend can make one in their name. A device cannot: what the agent is told when a task
/// comes due is the backend's to write. What is here is what a person does with a reminder
/// they already have: find it, move it, pause it and cancel it.
///
///     let page = try await agents.scheduledTasks.list()
///     try await agents.scheduledTasks.update(page.items[0].id, next: .in("PT15M"))
public struct ScheduledTasks: Sendable {
    private let backend: Backend

    init(backend: Backend) {
        self.backend = backend
    }

    /// One page of this user's tasks, newest first, the ones their backend made included. Pass
    /// the page's `nextCursor` as `query.cursor` for the next one.
    public func list(_ query: ScheduledTaskQuery = ScheduledTaskQuery()) async throws -> Page<ScheduledTask> {
        let page = try await backend.call {
            try await $0.listScheduledTasks(
                configId: query.configID, cursor: query.cursor, limit: query.limit,
                sessionId: query.sessionID, state: query.state?.rawValue)
        }
        return Page(
            items: try page.items.map(ScheduledTask.init), hasMore: page.hasMore, nextCursor: page.nextCursor)
    }

    /// One task. Somebody else's is not found.
    public func get(_ id: String) async throws -> ScheduledTask {
        try ScheduledTask(try await backend.call { try await $0.getScheduledTask(id: id) })
    }

    /// Changes a task and returns it as it now is. Nil leaves a field as it is.
    ///
    /// `schedule` replaces the whole schedule, but for a timezone it leaves out. `next` moves
    /// only the next occurrence: a snooze, or a skip when it lands past the one after. Either
    /// lands 15 minutes from now at the soonest, unless it is no earlier than the current next
    /// occurrence. `deliverTo` empty is the conversation alone, and `quietHours` with both ends
    /// empty clears the hold.
    public func update(
        _ id: String, title: String? = nil, schedule: ScheduledTask.Schedule? = nil,
        next: ScheduledTask.Next? = nil, deliverTo: [String]? = nil,
        quietHours: ScheduledTask.QuietHours? = nil
    ) async throws -> ScheduledTask {
        let body = UpdateScheduledTaskRequest(
            deliverTo: deliverTo, next: next?.schema, quietHours: quietHours?.schema,
            schedule: schedule?.schema, title: title)
        return try ScheduledTask(
            try await backend.call { try await $0.updateScheduledTask(id: id, updateScheduledTaskRequest: body) })
    }

    /// Holds a task: nothing comes due while it is paused, and what it slept through is never
    /// replayed.
    public func pause(_ id: String) async throws -> ScheduledTask {
        try ScheduledTask(try await backend.call { try await $0.pauseScheduledTask(id: id) })
    }

    /// Starts a paused or failed task again, counting its next occurrence from now. A one-shot
    /// whose time came while it was paused is completed instead, and a task with no occurrence
    /// left needs a new schedule first.
    public func resume(_ id: String) async throws -> ScheduledTask {
        try ScheduledTask(try await backend.call { try await $0.resumeScheduledTask(id: id) })
    }

    /// Cancels a task: it and its runs are gone.
    public func delete(_ id: String) async throws {
        try await backend.call { try await $0.deleteScheduledTask(id: id) }
    }

    /// One page of the times a task came due, newest first. Snoozes and pauses write none.
    public func runs(_ id: String, limit: Int? = nil, cursor: String? = nil) async throws -> Page<ScheduledTaskRun> {
        let page = try await backend.call { try await $0.listScheduledTaskRuns(id: id, cursor: cursor, limit: limit) }
        return Page(
            items: try page.items.map(ScheduledTaskRun.init), hasMore: page.hasMore, nextCursor: page.nextCursor)
    }
}

/// Which of this user's tasks to list. `nil` leaves a filter out.
public struct ScheduledTaskQuery: Sendable, Hashable {
    public var state: ScheduledTask.State?
    /// Only the ones on this agent config.
    public var configID: String?
    /// Only the ones following up in this conversation.
    public var sessionID: String?
    /// Up to 200. Nil is 25.
    public var limit: Int?
    /// The `nextCursor` of the page before, sent with the same filters. Nil is the first page.
    public var cursor: String?

    public init(limit: Int? = nil, cursor: String? = nil) {
        self.limit = limit
        self.cursor = cursor
    }
}

/// Something an agent will do for the user later: a reminder, a check-in, a recurring report.
public struct ScheduledTask: Sendable, Hashable, Identifiable {
    public let id: String
    /// A short name the user will recognise.
    public let title: String
    /// What the agent is told when the task comes due.
    public let instructions: String
    public let schedule: Schedule
    /// Where an occurrence goes, in order, the first that lands: `conversation`, `sms`,
    /// `whatsapp`, `imessage` or `phone`.
    public let deliverTo: [String]
    /// The part of the day an occurrence is held through, or nil for no hold.
    public let quietHours: QuietHours?
    public let state: State
    /// Nil when there is none left.
    public let nextRunAt: Date?
    /// Up to the next five occurrences.
    public let upcoming: [Date]
    /// When a run last finished, or nil when none has.
    public let lastRunAt: Date?
    /// Failed occurrences in a row. Five fail the task.
    public let consecutiveFailures: Int
    public let createdBy: CreatedBy
    /// The conversation the task follows up in. Empty until its first occurrence opens one.
    public let sessionID: String
    public let configID: String
    public let custom: [String: RawJSON]
    public let createdAt: Date
    public let updatedAt: Date

    public enum State: String, Sendable {
        case active
        case paused
        /// A one-shot that ran, or a repeat past its end.
        case completed
        /// Paused after five failed occurrences in a row. `resume` clears it.
        case failed
        /// A state this SDK has never heard of.
        case unknown
    }

    public enum CreatedBy: String, Sendable {
        /// The model, in a conversation.
        case agent
        /// The app's backend.
        case api
        /// A maker this SDK has never heard of.
        case unknown
    }

    /// When a task comes due. On a change, exactly one of `at` and `in` is given.
    public struct Schedule: Sendable, Hashable {
        /// When it first runs as the user's clock reads it, `2026-10-20T09:00:00`. With an
        /// offset or `Z` it is that instant.
        public var at: String?
        /// How long from now instead, an ISO 8601 duration: `PT20M`, `P1D`. Never read back.
        public var `in`: String?
        /// Nil for once. Otherwise an RRULE without DTSTART or UNTIL: `FREQ=DAILY`,
        /// `FREQ=WEEKLY;BYDAY=MO,WE,FR`.
        public var `repeat`: String?
        /// An IANA zone, such as `Europe/Amsterdam`. Nil keeps the task's own.
        public var timezone: String?
        /// The last local date it repeats on, inclusive: `2026-12-31`.
        public var until: String?
        /// What the schedule says in one sentence, written by the router.
        public private(set) var description = ""

        public init(
            at: String? = nil, in: String? = nil, repeat: String? = nil, timezone: String? = nil,
            until: String? = nil
        ) {
            self.at = at
            self.in = `in`
            self.repeat = `repeat`
            self.timezone = timezone
            self.until = until
        }

        init(_ schema: ScheduledTaskSchedule) {
            at = schema.at
            self.in = schema.in
            self.repeat = schema.repeat
            timezone = schema.timezone
            until = schema.until
            description = schema.description ?? ""
        }

        var schema: ScheduledTaskSchedule {
            .init(in: `in`, repeat: `repeat`, at: at, timezone: timezone, until: until)
        }
    }

    /// Where the next occurrence moves to.
    public enum Next: Sendable, Hashable {
        /// As the user's clock reads it, in the task's zone: `2026-10-20T09:00:00`.
        case at(String)
        /// How long from now, an ISO 8601 duration: `PT15M`.
        case `in`(String)

        var schema: ScheduledTaskNext {
            switch self {
            case .at(let at): .init(at: at)
            case .in(let duration): .init(in: duration)
            }
        }
    }

    /// The part of the user's day an occurrence that comes due in is held through, as their
    /// clock reads it. It may run past midnight: `22:00` to `07:00` holds the night.
    public struct QuietHours: Sendable, Hashable {
        public var start: String
        public var end: String

        public init(start: String, end: String) {
            self.start = start
            self.end = end
        }

        var schema: ScheduledTaskQuietHours {
            .init(end: end, start: start)
        }
    }

    init(_ schema: ScheduledTaskPayload) throws {
        let dates = RouterDates()
        id = schema.id
        title = schema.title
        instructions = schema.instructions
        self.schedule = Schedule(schema.schedule)
        deliverTo = schema.deliverTo
        quietHours = schema.quietHours.map { QuietHours(start: $0.start, end: $0.end) }
        state = State(rawValue: schema.state.rawValue) ?? .unknown
        nextRunAt = try schema.nextRunAt.map(dates.decode)
        upcoming = try schema.upcoming.map(dates.decode)
        lastRunAt = try schema.lastRunAt.map(dates.decode)
        consecutiveFailures = schema.consecutiveFailures
        createdBy = CreatedBy(rawValue: schema.createdBy.rawValue) ?? .unknown
        sessionID = schema.sessionId
        configID = schema.configId
        custom = schema.custom ?? [:]
        createdAt = schema.createdAt
        updatedAt = schema.updatedAt
    }
}

/// One time a task came due, or was run by hand.
public struct ScheduledTaskRun: Sendable, Hashable, Identifiable {
    public let id: String
    public let taskID: String
    /// When the occurrence was due.
    public let scheduledFor: Date
    public let trigger: Trigger
    public let state: State
    /// Where the follow-up landed: `conversation`, `sms`, `whatsapp`, `imessage` or `phone`.
    /// Empty until it lands.
    public let deliveredTo: String
    public let sessionID: String
    public let attempts: Int
    public let startedAt: Date?
    public let finishedAt: Date?
    /// Why it failed or was missed, as a short code such as `delivery_failed` or `task_paused`.
    /// Empty otherwise.
    public let error: String

    public enum Trigger: String, Sendable {
        /// It came due.
        case schedule
        /// It was run by hand.
        case manual
        /// A trigger this SDK has never heard of.
        case unknown
    }

    public enum State: String, Sendable {
        /// Retries included.
        case running
        case delivered
        case failed
        /// Too late to be worth delivering, or due while the task was paused.
        case missed
        /// A state this SDK has never heard of.
        case unknown
    }

    init(_ schema: ScheduledTaskRunPayload) throws {
        id = schema.id
        taskID = schema.taskId
        scheduledFor = try RouterDates().decode(schema.scheduledFor)
        trigger = Trigger(rawValue: schema.trigger.rawValue) ?? .unknown
        state = State(rawValue: schema.state.rawValue) ?? .unknown
        deliveredTo = schema.deliveredTo
        sessionID = schema.sessionId
        attempts = schema.attempts
        startedAt = schema.startedAt
        finishedAt = schema.finishedAt
        error = schema.error ?? ""
    }
}
