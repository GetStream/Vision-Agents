import Foundation
import StreamCore

final class SessionQueryPayload: Sendable, Codable, JSONEncodable {
    /// The next_cursor of the previous page, sent with the same filter and sort. Omitted is the first page.
    let cursor: String?
    /// Which sessions to list. The keys of one object are ANDed. $and and $or each take 1 to 10 filters and nest up to 3 deep; a whole filter holds at most 30 field conditions. An empty filter lists every session the caller may see. Any other key, and any other operator, is refused with a 400 that names it and lists the fields that can be filtered.
    let filter: SessionFilterPayload?
    /// Up to 200. Omitted is 25.
    let limit: Int?
    /// Omitted is last_message_at descending, or relevance for a text search. One of: last_message_at; pinned, then last_message_at, which lists the pinned sessions first; review_score, for a server-side caller; relevance, for a text search and nothing else.
    let sort: [SessionSortPayload]?

    init(cursor: String? = nil, filter: SessionFilterPayload? = nil, limit: Int? = nil, sort: [SessionSortPayload]? = nil) {
        self.cursor = cursor
        self.filter = filter
        self.limit = limit
        self.sort = sort
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.cursor = try container.decodeIfPresent(String.self, forKey: .cursor)
        self.filter = try container.decodeIfPresent(SessionFilterPayload.self, forKey: .filter)
        self.limit = try container.decodeIfPresent(Int.self, forKey: .limit)
        self.sort = try container.decodeIfPresent([SessionSortPayload].self, forKey: .sort)
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
