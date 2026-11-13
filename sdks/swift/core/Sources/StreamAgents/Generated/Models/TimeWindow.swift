import Foundation
import StreamCore

/// A window of time: $gte is its start, which it holds, and $lt its end, which it does not. Either may be left out, and $gte is before $lt.
final class TimeWindow: Sendable, Codable, JSONEncodable {
    let gte: Date?
    let lt: Date?

    init(gte: Date? = nil, lt: Date? = nil) {
        self.gte = gte
        self.lt = lt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.gte = try container.decodeIfPresent(Date.self, forKey: .gte)
        self.lt = try container.decodeIfPresent(Date.self, forKey: .lt)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(gte, forKey: .gte)
        try container.encodeIfPresent(lt, forKey: .lt)
    }

}

extension TimeWindow: Hashable {
    static func == (lhs: TimeWindow, rhs: TimeWindow) -> Bool {
        lhs.gte == rhs.gte &&
        lhs.lt == rhs.lt
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(gte)
        hasher.combine(lt)
    }
}
