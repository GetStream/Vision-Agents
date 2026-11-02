import Foundation
import StreamCore

/// Why the session ended: agent_hangup, the agent hung up; transferred, the agent handed the caller to a person; hangup, the caller left; start, nobody joined the call in time; unanswered, the call placed was never answered.
final class SessionEndReason: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let agentHangup = SessionEndReason(rawValue: "agent_hangup")
    static let hangup = SessionEndReason(rawValue: "hangup")
    static let start = SessionEndReason(rawValue: "start")
    static let transferred = SessionEndReason(rawValue: "transferred")
    static let unanswered = SessionEndReason(rawValue: "unanswered")
}