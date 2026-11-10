import Foundation
import StreamCore

final class TimeRange: Sendable, Codable, JSONEncodable {
    /// At or after this RFC3339 time.
    let gte: Date?
    /// Strictly before this RFC3339 time.
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

extension TimeRange: Hashable {
    static func == (lhs: TimeRange, rhs: TimeRange) -> Bool {
        lhs.gte == rhs.gte &&
        lhs.lt == rhs.lt
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(gte)
        hasher.combine(lt)
    }
}
