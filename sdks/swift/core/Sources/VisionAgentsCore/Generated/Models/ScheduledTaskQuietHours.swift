import Foundation
import StreamCore

final class ScheduledTaskQuietHours: Sendable, Codable, JSONEncodable {
    /// When the hold ends. It may read before the start: 22:00 to 07:00 holds the night.
    let end: String
    /// When the hold begins, as the user's clock reads it. Both boundaries empty is no hold, which is also how an update clears one.
    let start: String

    init(end: String, start: String) {
        self.end = end
        self.start = start
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.end = try container.decode(String.self, forKey: .end)
        self.start = try container.decode(String.self, forKey: .start)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(end, forKey: .end)
        try container.encode(start, forKey: .start)
    }

}

extension ScheduledTaskQuietHours: Hashable {
    static func == (lhs: ScheduledTaskQuietHours, rhs: ScheduledTaskQuietHours) -> Bool {
        lhs.end == rhs.end &&
        lhs.start == rhs.start
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(end)
        hasher.combine(start)
    }
}
