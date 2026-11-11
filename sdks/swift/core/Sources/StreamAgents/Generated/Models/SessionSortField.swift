import Foundation
import StreamCore

/// last_message_at is when the agent last answered, or when the session began if it has not. pinned lists the pinned sessions first: it sorts descending only, comes first, and is followed by last_message_at, descending when left out. review_score is the review's score, sessions without one last either way, and is a server-side caller's only. relevance is the best match first, and is the one sort of a text search.
final class SessionSortField: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let lastMessageAt = SessionSortField(rawValue: "last_message_at")
    static let pinned = SessionSortField(rawValue: "pinned")
    static let relevance = SessionSortField(rawValue: "relevance")
    static let reviewScore = SessionSortField(rawValue: "review_score")
}