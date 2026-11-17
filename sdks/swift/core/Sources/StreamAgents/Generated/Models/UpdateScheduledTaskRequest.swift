import Foundation
import StreamCore

final class UpdateScheduledTaskRequest: Sendable, Codable, JSONEncodable {
    /// Replaces where an occurrence is delivered, in order. An empty list is the conversation alone.
    let deliverTo: [String]?
    /// What the agent is told when the task comes due. Server-side only: a device sending it is refused.
    let instructions: String?
    let next: ScheduledTaskNext?
    let quietHours: ScheduledTaskQuietHours?
    let schedule: ScheduledTaskSchedule?
    /// A short name the user will recognise.
    let title: String?

    init(deliverTo: [String]? = nil, instructions: String? = nil, next: ScheduledTaskNext? = nil, quietHours: ScheduledTaskQuietHours? = nil, schedule: ScheduledTaskSchedule? = nil, title: String? = nil) {
        self.deliverTo = deliverTo
        self.instructions = instructions
        self.next = next
        self.quietHours = quietHours
        self.schedule = schedule
        self.title = title
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.deliverTo = try container.decodeIfPresent([String].self, forKey: .deliverTo)
        self.instructions = try container.decodeIfPresent(String.self, forKey: .instructions)
        self.next = try container.decodeIfPresent(ScheduledTaskNext.self, forKey: .next)
        self.quietHours = try container.decodeIfPresent(ScheduledTaskQuietHours.self, forKey: .quietHours)
        self.schedule = try container.decodeIfPresent(ScheduledTaskSchedule.self, forKey: .schedule)
        self.title = try container.decodeIfPresent(String.self, forKey: .title)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(deliverTo, forKey: .deliverTo)
        try container.encodeIfPresent(instructions, forKey: .instructions)
        try container.encodeIfPresent(next, forKey: .next)
        try container.encodeIfPresent(quietHours, forKey: .quietHours)
        try container.encodeIfPresent(schedule, forKey: .schedule)
        try container.encodeIfPresent(title, forKey: .title)
    }

}

extension UpdateScheduledTaskRequest: Hashable {
    static func == (lhs: UpdateScheduledTaskRequest, rhs: UpdateScheduledTaskRequest) -> Bool {
        lhs.deliverTo == rhs.deliverTo &&
        lhs.instructions == rhs.instructions &&
        lhs.next == rhs.next &&
        lhs.quietHours == rhs.quietHours &&
        lhs.schedule == rhs.schedule &&
        lhs.title == rhs.title
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(deliverTo)
        hasher.combine(instructions)
        hasher.combine(next)
        hasher.combine(quietHours)
        hasher.combine(schedule)
        hasher.combine(title)
    }
}
