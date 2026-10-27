import Foundation
import StreamCore

final class ScheduledTaskRunPayloadState: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let delivered = ScheduledTaskRunPayloadState(rawValue: "delivered")
    static let failed = ScheduledTaskRunPayloadState(rawValue: "failed")
    static let missed = ScheduledTaskRunPayloadState(rawValue: "missed")
    static let running = ScheduledTaskRunPayloadState(rawValue: "running")
}

final class ScheduledTaskRunPayloadTrigger: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let manual = ScheduledTaskRunPayloadTrigger(rawValue: "manual")
    static let schedule = ScheduledTaskRunPayloadTrigger(rawValue: "schedule")
}

final class ScheduledTaskRunPayload: Sendable, Codable, JSONEncodable {
    let attempts: Int
    /// Where the follow-up landed: conversation, sms, whatsapp, imessage or phone. Empty until it lands.
    let deliveredTo: String
    /// Why it failed, as a short code, when it did: delivery_failed, timed_out, schedule_unreadable, user_on_call, channel_unreachable, no_reply, reply_failed, no_phone_number, no_agent_number, not_answered, already_called (a call was placed for the occurrence on an earlier attempt, so the user is not rung again), outside_window (a WhatsApp thread task whose person last wrote over 24 hours ago, with no template to send), thread_gone (a thread task's thread no longer exists or is not its user's), connection_removed (a thread task's line was removed), or, refused for money, account_suspended, insufficient_credits, system_limit_reached or spend_limit_reached; on a missed run, task_paused (a one-shot's time came while it was paused). The full error is in the router's logs. Null otherwise.
    let error: String?
    let finishedAt: Date?
    let id: String
    /// When the occurrence was due, RFC 3339 with the task zone's offset.
    let scheduledFor: String
    let sessionId: String
    let startedAt: Date?
    /// running (retries included), delivered, failed, or missed (it fell outside the grace period, its call would have rung more than 15 minutes late, or it came while the task was paused).
    let state: ScheduledTaskRunPayloadState
    let taskId: String
    /// schedule when it came due, manual when it was run by hand.
    let trigger: ScheduledTaskRunPayloadTrigger

    init(attempts: Int, deliveredTo: String, error: String? = nil, finishedAt: Date? = nil, id: String, scheduledFor: String, sessionId: String, startedAt: Date? = nil, state: ScheduledTaskRunPayloadState, taskId: String, trigger: ScheduledTaskRunPayloadTrigger) {
        self.attempts = attempts
        self.deliveredTo = deliveredTo
        self.error = error
        self.finishedAt = finishedAt
        self.id = id
        self.scheduledFor = scheduledFor
        self.sessionId = sessionId
        self.startedAt = startedAt
        self.state = state
        self.taskId = taskId
        self.trigger = trigger
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.attempts = try container.decode(Int.self, forKey: .attempts)
        self.deliveredTo = try container.decode(String.self, forKey: .deliveredTo)
        self.error = try container.decodeIfPresent(String.self, forKey: .error)
        self.finishedAt = try container.decodeIfPresent(Date.self, forKey: .finishedAt)
        self.id = try container.decode(String.self, forKey: .id)
        self.scheduledFor = try container.decode(String.self, forKey: .scheduledFor)
        self.sessionId = try container.decode(String.self, forKey: .sessionId)
        self.startedAt = try container.decodeIfPresent(Date.self, forKey: .startedAt)
        self.state = try container.decode(ScheduledTaskRunPayloadState.self, forKey: .state)
        self.taskId = try container.decode(String.self, forKey: .taskId)
        self.trigger = try container.decode(ScheduledTaskRunPayloadTrigger.self, forKey: .trigger)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(attempts, forKey: .attempts)
        try container.encode(deliveredTo, forKey: .deliveredTo)
        try container.encodeIfPresent(error, forKey: .error)
        try container.encodeIfPresent(finishedAt, forKey: .finishedAt)
        try container.encode(id, forKey: .id)
        try container.encode(scheduledFor, forKey: .scheduledFor)
        try container.encode(sessionId, forKey: .sessionId)
        try container.encodeIfPresent(startedAt, forKey: .startedAt)
        try container.encode(state, forKey: .state)
        try container.encode(taskId, forKey: .taskId)
        try container.encode(trigger, forKey: .trigger)
    }

}

extension ScheduledTaskRunPayload: Hashable {
    static func == (lhs: ScheduledTaskRunPayload, rhs: ScheduledTaskRunPayload) -> Bool {
        lhs.attempts == rhs.attempts &&
        lhs.deliveredTo == rhs.deliveredTo &&
        lhs.error == rhs.error &&
        lhs.finishedAt == rhs.finishedAt &&
        lhs.id == rhs.id &&
        lhs.scheduledFor == rhs.scheduledFor &&
        lhs.sessionId == rhs.sessionId &&
        lhs.startedAt == rhs.startedAt &&
        lhs.state == rhs.state &&
        lhs.taskId == rhs.taskId &&
        lhs.trigger == rhs.trigger
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(attempts)
        hasher.combine(deliveredTo)
        hasher.combine(error)
        hasher.combine(finishedAt)
        hasher.combine(id)
        hasher.combine(scheduledFor)
        hasher.combine(sessionId)
        hasher.combine(startedAt)
        hasher.combine(state)
        hasher.combine(taskId)
        hasher.combine(trigger)
    }
}
