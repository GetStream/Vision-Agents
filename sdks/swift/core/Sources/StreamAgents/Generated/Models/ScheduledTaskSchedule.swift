import Foundation
import StreamCore

final class ScheduledTaskSchedule: Sendable, Codable, JSONEncodable {
    /// How long from now instead, an ISO 8601 duration: PT20M, P1D. Days are calendar days: P1D is the same time tomorrow. Resolved on write; a response never carries it.
    let `in`: String?
    /// Omit for once. Otherwise an RRULE subset without DTSTART or UNTIL: FREQ=DAILY, FREQ=WEEKLY;BYDAY=MO,WE,FR, FREQ=MONTHLY;BYMONTHDAY=1.
    let `repeat`: String?
    /// When it first runs, as the user's clock reads it, 2026-10-20T09:00:00; with an offset or Z it is that instant, read in the schedule's zone. Exactly one of at and in is given on a write.
    let at: String?
    /// What the schedule says, in one sentence. Read-only: written by the router, and the same sentence the model reads back.
    let description: String?
    /// An IANA zone, such as Europe/Amsterdam. On create, left out resolves to the user's stored zone, then the agent's default_timezone; on update, to the task's own zone.
    let timezone: String?
    /// The last local date it repeats on, inclusive, 2026-12-31.
    let until: String?

    init(`in`: String? = nil, `repeat`: String? = nil, at: String? = nil, description: String? = nil, timezone: String? = nil, until: String? = nil) {
        self.`in` = `in`
        self.`repeat` = `repeat`
        self.at = at
        self.description = description
        self.timezone = timezone
        self.until = until
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.`in` = try container.decodeIfPresent(String.self, forKey: .`in`)
        self.`repeat` = try container.decodeIfPresent(String.self, forKey: .`repeat`)
        self.at = try container.decodeIfPresent(String.self, forKey: .at)
        self.description = try container.decodeIfPresent(String.self, forKey: .description)
        self.timezone = try container.decodeIfPresent(String.self, forKey: .timezone)
        self.until = try container.decodeIfPresent(String.self, forKey: .until)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(`in`, forKey: .`in`)
        try container.encodeIfPresent(`repeat`, forKey: .`repeat`)
        try container.encodeIfPresent(at, forKey: .at)
        try container.encodeIfPresent(description, forKey: .description)
        try container.encodeIfPresent(timezone, forKey: .timezone)
        try container.encodeIfPresent(until, forKey: .until)
    }

}

extension ScheduledTaskSchedule: Hashable {
    static func == (lhs: ScheduledTaskSchedule, rhs: ScheduledTaskSchedule) -> Bool {
        lhs.`in` == rhs.`in` &&
        lhs.`repeat` == rhs.`repeat` &&
        lhs.at == rhs.at &&
        lhs.description == rhs.description &&
        lhs.timezone == rhs.timezone &&
        lhs.until == rhs.until
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(`in`)
        hasher.combine(`repeat`)
        hasher.combine(at)
        hasher.combine(description)
        hasher.combine(timezone)
        hasher.combine(until)
    }
}
