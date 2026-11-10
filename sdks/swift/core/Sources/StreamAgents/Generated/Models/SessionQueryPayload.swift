import Foundation
import StreamCore

final class SessionQueryPayload: Sendable, Codable, JSONEncodable {
    /// The next_cursor of the previous page, sent with the same filter and sort. Omitted is the first page.
    let cursor: String?
    /// Which sessions to list. A field not listed here is refused rather than ignored.
    let filter: SessionFilter?
    /// Up to 200. Omitted is 25.
    let limit: Int?
    /// Omitted is updated_at, or relevance for a text search.
    let sort: [SessionSort]?

    init(cursor: String? = nil, filter: SessionFilter? = nil, limit: Int? = nil, sort: [SessionSort]? = nil) {
        self.cursor = cursor
        self.filter = filter
        self.limit = limit
        self.sort = sort
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.cursor = try container.decodeIfPresent(String.self, forKey: .cursor)
        self.filter = try container.decodeIfPresent(SessionFilter.self, forKey: .filter)
        self.limit = try container.decodeIfPresent(Int.self, forKey: .limit)
        self.sort = try container.decodeIfPresent([SessionSort].self, forKey: .sort)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(cursor, forKey: .cursor)
        try container.encodeIfPresent(filter, forKey: .filter)
        try container.encodeIfPresent(limit, forKey: .limit)
        try container.encodeIfPresent(sort, forKey: .sort)
    }

}

extension SessionQueryPayload: Hashable {
    static func == (lhs: SessionQueryPayload, rhs: SessionQueryPayload) -> Bool {
        lhs.cursor == rhs.cursor &&
        lhs.filter == rhs.filter &&
        lhs.limit == rhs.limit &&
        lhs.sort == rhs.sort
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(cursor)
        hasher.combine(filter)
        hasher.combine(limit)
        hasher.combine(sort)
    }
}
