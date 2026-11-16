import Foundation
import StreamCore

final class SessionPage: Sendable, Codable, JSONEncodable {
    let hasMore: Bool
    let items: [SessionPayload]
    /// Pass as `cursor` for the next page. Absent on the last one.
    let nextCursor: String?

    init(hasMore: Bool, items: [SessionPayload], nextCursor: String? = nil) {
        self.hasMore = hasMore
        self.items = items
        self.nextCursor = nextCursor
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.hasMore = try container.decode(Bool.self, forKey: .hasMore)
        self.items = try container.decode([SessionPayload].self, forKey: .items)
        self.nextCursor = try container.decodeIfPresent(String.self, forKey: .nextCursor)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(hasMore, forKey: .hasMore)
        try container.encode(items, forKey: .items)
        try container.encodeIfPresent(nextCursor, forKey: .nextCursor)
    }

}

extension SessionPage: Hashable {
    static func == (lhs: SessionPage, rhs: SessionPage) -> Bool {
        lhs.hasMore == rhs.hasMore &&
        lhs.items == rhs.items &&
        lhs.nextCursor == rhs.nextCursor
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(hasMore)
        hasher.combine(items)
        hasher.combine(nextCursor)
    }
}
