import Foundation
import StreamCore

final class ScheduledTaskNext: Sendable, Codable, JSONEncodable {
    /// How long from now instead, an ISO 8601 duration: PT15M.
    let `in`: String?
    /// The next occurrence as the user's clock reads it, in the task's zone.
    let at: String?

    init(`in`: String? = nil, at: String? = nil) {
        self.`in` = `in`
        self.at = at
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.`in` = try container.decodeIfPresent(String.self, forKey: .`in`)
        self.at = try container.decodeIfPresent(String.self, forKey: .at)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(`in`, forKey: .`in`)
        try container.encodeIfPresent(at, forKey: .at)
    }

}

extension ScheduledTaskNext: Hashable {
    static func == (lhs: ScheduledTaskNext, rhs: ScheduledTaskNext) -> Bool {
        lhs.`in` == rhs.`in` &&
        lhs.at == rhs.at
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(`in`)
        hasher.combine(at)
    }
}
