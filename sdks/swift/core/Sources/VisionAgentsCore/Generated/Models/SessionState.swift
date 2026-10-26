import Foundation
import StreamCore

/// Whether the agent is still in the call.
final class SessionState: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let ended = SessionState(rawValue: "ended")
    static let live = SessionState(rawValue: "live")
}