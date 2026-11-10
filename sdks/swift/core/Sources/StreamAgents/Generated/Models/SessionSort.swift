import Foundation
import StreamCore

final class SessionSort: Sendable, Codable, JSONEncodable {
    /// -1, descending. Ascending is not offered.
    let direction: Int?
    /// updated_at is the most recently active first. relevance is the best match first, and only sorts a text search.
    let field: SessionSortField

    init(direction: Int? = nil, field: SessionSortField) {
        self.direction = direction
        self.field = field
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.direction = try container.decodeIfPresent(Int.self, forKey: .direction)
        self.field = try container.decode(SessionSortField.self, forKey: .field)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(direction, forKey: .direction)
        try container.encode(field, forKey: .field)
    }

}

extension SessionSort: Hashable {
    static func == (lhs: SessionSort, rhs: SessionSort) -> Bool {
        lhs.direction == rhs.direction &&
        lhs.field == rhs.field
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(direction)
        hasher.combine(field)
    }
}
