import Foundation
import StreamCore

/// Why the session ended, with whose fault it was and the stage it came at: agent_hangup, the agent hung up (fault none, stage conversation); transferred, the agent handed the caller to a person (fault none, stage conversation); hangup, the caller left (fault caller, stage conversation); start, nobody joined the phone call in time (fault carrier, stage ringing); unanswered, the call placed was never answered (fault caller, stage ringing); voicemail, an answering machine picked up the call placed, and the agent left its message or hung up (fault none, stage ringing); closed, the session was closed through the API, or replaced by a newer one (fault none, stage conversation); shutdown, the router ended the session as it shut down (fault router, stage conversation); agent_failed, the agent left the call or answered no turn for a failure of the router's own (fault router, stage conversation); model_failed, a model vendor failed: the agent left the call or answered no turn (fault vendor, stage conversation); model_config_refused, the model failed on the customer's own key or model, or refused a setting, and the agent answered no turn (fault customer_config, stage conversation); model_no_credit, the credit or spend limits refused the model's work, and the agent answered no turn (fault customer_credit, stage conversation).
final class SessionEndReason: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let agentFailed = SessionEndReason(rawValue: "agent_failed")
    static let agentHangup = SessionEndReason(rawValue: "agent_hangup")
    static let closed = SessionEndReason(rawValue: "closed")
    static let hangup = SessionEndReason(rawValue: "hangup")
    static let modelConfigRefused = SessionEndReason(rawValue: "model_config_refused")
    static let modelFailed = SessionEndReason(rawValue: "model_failed")
    static let modelNoCredit = SessionEndReason(rawValue: "model_no_credit")
    static let shutdown = SessionEndReason(rawValue: "shutdown")
    static let start = SessionEndReason(rawValue: "start")
    static let transferred = SessionEndReason(rawValue: "transferred")
    static let unanswered = SessionEndReason(rawValue: "unanswered")
    static let voicemail = SessionEndReason(rawValue: "voicemail")
}