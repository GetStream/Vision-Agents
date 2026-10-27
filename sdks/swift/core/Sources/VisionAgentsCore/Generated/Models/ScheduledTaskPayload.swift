import Foundation
import StreamCore

final class ScheduledTaskPayloadCreatedBy: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let agent = ScheduledTaskPayloadCreatedBy(rawValue: "agent")
    static let api = ScheduledTaskPayloadCreatedBy(rawValue: "api")
}

final class ScheduledTaskPayloadState: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let active = ScheduledTaskPayloadState(rawValue: "active")
    static let completed = ScheduledTaskPayloadState(rawValue: "completed")
    static let failed = ScheduledTaskPayloadState(rawValue: "failed")
    static let paused = ScheduledTaskPayloadState(rawValue: "paused")
}

final class ScheduledTaskPayload: Sendable, Codable, JSONEncodable {
    let configId: String
    let consecutiveFailures: Int
    let createdAt: Date
    /// agent when the model made it in a conversation, api when a backend did.
    let createdBy: ScheduledTaskPayloadCreatedBy
    let custom: [String: RawJSON]?
    /// Where an occurrence is delivered, in order, the first that lands.
    let deliverTo: [String]
    let id: String
    let instructions: String
    /// When a run last finished, in UTC. Null when none has.
    let lastRunAt: String?
    /// The next occurrence, RFC 3339 with the offset of the task's own zone. Null when there is none.
    let nextRunAt: String?
    let quietHours: ScheduledTaskQuietHours?
    let schedule: ScheduledTaskSchedule
    /// The conversation the task follows up in. Empty until its first occurrence opens one.
    let sessionId: String
    /// active, paused, completed (a one-shot that ran, or a recurrence past its end), or failed (auto-paused after five failed occurrences in a row; resume clears it).
    let state: ScheduledTaskPayloadState
    let title: String
    /// Up to the next five occurrences, RFC 3339 with the task zone's offset.
    let upcoming: [String]
    let updatedAt: Date
    let userId: String

    init(configId: String, consecutiveFailures: Int, createdAt: Date, createdBy: ScheduledTaskPayloadCreatedBy, custom: [String: RawJSON]? = nil, deliverTo: [String], id: String, instructions: String, lastRunAt: String? = nil, nextRunAt: String? = nil, quietHours: ScheduledTaskQuietHours? = nil, schedule: ScheduledTaskSchedule, sessionId: String, state: ScheduledTaskPayloadState, title: String, upcoming: [String], updatedAt: Date, userId: String) {
        self.configId = configId
        self.consecutiveFailures = consecutiveFailures
        self.createdAt = createdAt
        self.createdBy = createdBy
        self.custom = custom
        self.deliverTo = deliverTo
        self.id = id
        self.instructions = instructions
        self.lastRunAt = lastRunAt
        self.nextRunAt = nextRunAt
        self.quietHours = quietHours
        self.schedule = schedule
        self.sessionId = sessionId
        self.state = state
        self.title = title
        self.upcoming = upcoming
        self.updatedAt = updatedAt
        self.userId = userId
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.configId = try container.decode(String.self, forKey: .configId)
        self.consecutiveFailures = try container.decode(Int.self, forKey: .consecutiveFailures)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
        self.createdBy = try container.decode(ScheduledTaskPayloadCreatedBy.self, forKey: .createdBy)
        self.custom = try container.decodeIfPresent([String: RawJSON].self, forKey: .custom)
        self.deliverTo = try container.decode([String].self, forKey: .deliverTo)
        self.id = try container.decode(String.self, forKey: .id)
        self.instructions = try container.decode(String.self, forKey: .instructions)
        self.lastRunAt = try container.decodeIfPresent(String.self, forKey: .lastRunAt)
        self.nextRunAt = try container.decodeIfPresent(String.self, forKey: .nextRunAt)
        self.quietHours = try container.decodeIfPresent(ScheduledTaskQuietHours.self, forKey: .quietHours)
        self.schedule = try container.decode(ScheduledTaskSchedule.self, forKey: .schedule)
        self.sessionId = try container.decode(String.self, forKey: .sessionId)
        self.state = try container.decode(ScheduledTaskPayloadState.self, forKey: .state)
        self.title = try container.decode(String.self, forKey: .title)
        self.upcoming = try container.decode([String].self, forKey: .upcoming)
        self.updatedAt = try container.decode(Date.self, forKey: .updatedAt)
        self.userId = try container.decode(String.self, forKey: .userId)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(configId, forKey: .configId)
        try container.encode(consecutiveFailures, forKey: .consecutiveFailures)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(createdBy, forKey: .createdBy)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encode(deliverTo, forKey: .deliverTo)
        try container.encode(id, forKey: .id)
        try container.encode(instructions, forKey: .instructions)
        try container.encodeIfPresent(lastRunAt, forKey: .lastRunAt)
        try container.encodeIfPresent(nextRunAt, forKey: .nextRunAt)
        try container.encodeIfPresent(quietHours, forKey: .quietHours)
        try container.encode(schedule, forKey: .schedule)
        try container.encode(sessionId, forKey: .sessionId)
        try container.encode(state, forKey: .state)
        try container.encode(title, forKey: .title)
        try container.encode(upcoming, forKey: .upcoming)
        try container.encode(updatedAt, forKey: .updatedAt)
        try container.encode(userId, forKey: .userId)
    }

}

extension ScheduledTaskPayload: Hashable {
    static func == (lhs: ScheduledTaskPayload, rhs: ScheduledTaskPayload) -> Bool {
        lhs.configId == rhs.configId &&
        lhs.consecutiveFailures == rhs.consecutiveFailures &&
        lhs.createdAt == rhs.createdAt &&
        lhs.createdBy == rhs.createdBy &&
        lhs.custom == rhs.custom &&
        lhs.deliverTo == rhs.deliverTo &&
        lhs.id == rhs.id &&
        lhs.instructions == rhs.instructions &&
        lhs.lastRunAt == rhs.lastRunAt &&
        lhs.nextRunAt == rhs.nextRunAt &&
        lhs.quietHours == rhs.quietHours &&
        lhs.schedule == rhs.schedule &&
        lhs.sessionId == rhs.sessionId &&
        lhs.state == rhs.state &&
        lhs.title == rhs.title &&
        lhs.upcoming == rhs.upcoming &&
        lhs.updatedAt == rhs.updatedAt &&
        lhs.userId == rhs.userId
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(configId)
        hasher.combine(consecutiveFailures)
        hasher.combine(createdAt)
        hasher.combine(createdBy)
        hasher.combine(custom)
        hasher.combine(deliverTo)
        hasher.combine(id)
        hasher.combine(instructions)
        hasher.combine(lastRunAt)
        hasher.combine(nextRunAt)
        hasher.combine(quietHours)
        hasher.combine(schedule)
        hasher.combine(sessionId)
        hasher.combine(state)
        hasher.combine(title)
        hasher.combine(upcoming)
        hasher.combine(updatedAt)
        hasher.combine(userId)
    }
}
