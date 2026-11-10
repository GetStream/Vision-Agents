import Foundation
import StreamCore

/// updated_at is the most recently active first. relevance is the best match first, and only sorts a text search.
final class SessionSortField: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let relevance = SessionSortField(rawValue: "relevance")
    static let updatedAt = SessionSortField(rawValue: "updated_at")
}