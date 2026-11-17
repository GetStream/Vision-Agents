import Foundation
import StreamCore

/// How far a session had got when it ended: ringing, before a person was on the line of a phone call; conversation, once the session was open.
final class SessionEndStage: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let conversation = SessionEndStage(rawValue: "conversation")
    static let ringing = SessionEndStage(rawValue: "ringing")
}