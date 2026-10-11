import Foundation

/// One thing a client can do to a running conversation over its socket.
///
/// The commands `readCommands` in the router accepts, but for `respond`: asking is
/// `responses.create`, which names the turn it starts. Everything else a caller might want is
/// a request rather than a frame.
public enum Command: Sendable, Hashable {
    /// Speak this without going through the model.
    case say(String)
    /// Abandon the reply in flight.
    case interrupt
    /// Answer a tool call. One of `output` or `error` says how it went. A call made for a
    /// request is only accepted back with its `requestID` and `turnID`.
    case toolResult(
        id: String, output: String?, error: String?, requestID: String = "", turnID: String = "")
    /// A person's answer to a tool call that waited for them. A declined call's step shows
    /// `summary`. The call is still answered with `toolResult` either way.
    case toolApproval(
        id: String, allowed: Bool, summary: String = "", requestID: String = "", turnID: String = "")
    /// End the session.
    case close
}

extension Command: Encodable {
    private enum CodingKeys: String, CodingKey {
        case type, text, toolCallID = "tool_call_id", output, error
        case requestID = "request_id", turnID = "turn_id", allowed, summary
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .say(let text):
            try container.encode("say", forKey: .type)
            try container.encode(text, forKey: .text)
        case .interrupt:
            try container.encode("interrupt", forKey: .type)
        case .toolResult(let id, let output, let error, let requestID, let turnID):
            try container.encode("tool_result", forKey: .type)
            try container.encode(id, forKey: .toolCallID)
            // The router reads both fields off one struct and treats the empty string as
            // absent, so sending the empty string and sending nothing are the same thing.
            try container.encode(output ?? "", forKey: .output)
            try container.encode(error ?? "", forKey: .error)
            if !requestID.isEmpty {
                try container.encode(requestID, forKey: .requestID)
            }
            if !turnID.isEmpty {
                try container.encode(turnID, forKey: .turnID)
            }
        case .toolApproval(let id, let allowed, let summary, let requestID, let turnID):
            try container.encode("tool_approval", forKey: .type)
            try container.encode(id, forKey: .toolCallID)
            try container.encode(allowed, forKey: .allowed)
            if !summary.isEmpty {
                try container.encode(summary, forKey: .summary)
            }
            if !requestID.isEmpty {
                try container.encode(requestID, forKey: .requestID)
            }
            if !turnID.isEmpty {
                try container.encode(turnID, forKey: .turnID)
            }
        case .close:
            try container.encode("close", forKey: .type)
        }
    }
}
