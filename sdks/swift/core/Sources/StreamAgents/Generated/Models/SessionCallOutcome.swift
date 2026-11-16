import Foundation
import StreamCore

/// How the call the router placed for a session ended.
final class SessionCallOutcome: Sendable, Codable, JSONEncodable {
    /// When the person picked up: their phone joined the call.
    let answeredAt: Date?
    /// How long they talked, from answered_at to when their phone left, rounded to the nearest second. Absent until it has.
    let durationSeconds: Int?
    /// How a placed call ended.
    /// completed: the person answered and the call ended.
    /// busy: the line was busy or the person declined.
    /// no_answer: it rang and nobody answered.
    /// failed: the vendor could not place it, for a bad number or a carrier fault.
    /// canceled: the vendor says the call was hung up before an answer.
    /// unknown: the vendor reports nothing about how it ended, and the phone never joined.
    let outcome: CallOutcome?

    init(answeredAt: Date? = nil, durationSeconds: Int? = nil, outcome: CallOutcome? = nil) {
        self.answeredAt = answeredAt
        self.durationSeconds = durationSeconds
        self.outcome = outcome
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.answeredAt = try container.decodeIfPresent(Date.self, forKey: .answeredAt)
        self.durationSeconds = try container.decodeIfPresent(Int.self, forKey: .durationSeconds)
        self.outcome = try container.decodeIfPresent(CallOutcome.self, forKey: .outcome)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(answeredAt, forKey: .answeredAt)
        try container.encodeIfPresent(durationSeconds, forKey: .durationSeconds)
        try container.encodeIfPresent(outcome, forKey: .outcome)
    }

}

extension SessionCallOutcome: Hashable {
    static func == (lhs: SessionCallOutcome, rhs: SessionCallOutcome) -> Bool {
        lhs.answeredAt == rhs.answeredAt &&
        lhs.durationSeconds == rhs.durationSeconds &&
        lhs.outcome == rhs.outcome
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(answeredAt)
        hasher.combine(durationSeconds)
        hasher.combine(outcome)
    }
}
