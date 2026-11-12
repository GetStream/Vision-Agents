import Foundation
import StreamCore

final class SessionSortPayload: Sendable, Codable, JSONEncodable {
    /// -1 is descending and 1 ascending. pinned and relevance only sort descending.
    let direction: Int?
    /// last_message_at is when the agent last answered, or when the session began if it has not. pinned lists the pinned sessions first: it sorts descending only, comes first, and is followed by last_message_at, descending when left out. review_score is the review's score, sessions without one last either way, and is a server-side caller's only. relevance is the best match first, and is the one sort of a text search.
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

extension SessionSortPayload: Hashable {
    static func == (lhs: SessionSortPayload, rhs: SessionSortPayload) -> Bool {
        lhs.direction == rhs.direction &&
        lhs.field == rhs.field
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(direction)
        hasher.combine(field)
    }
}
