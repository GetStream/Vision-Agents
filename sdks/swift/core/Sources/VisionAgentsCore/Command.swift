import Foundation

/// One thing a client can do to a running conversation over its socket.
///
/// These are the six commands `readCommands` in the router accepts. Everything else a caller
/// might want is a request rather than a frame.
public enum Command: Sendable, Hashable {
    /// Speak this without going through the model.
    case say(String)
    /// Answer this as though it had been heard.
    case respond(String)
    /// Submit durable text. Retain this ID and exact text when retrying.
    case respondCommand(id: String, text: String)
    /// Abandon the reply in flight.
    case interrupt
    /// Stop one durable command wherever it got to. A stop arriving after that command
    /// finished replays its receipt and leaves the command running now alone.
    case interruptCommand(id: String)
    /// Replace the system prompt, from the next turn on.
    case instructions(String)
    /// Answer a tool call. One of `output` or `error` says how it went.
    case toolResult(id: String, output: String?, error: String?)
    /// End the session.
    case close
}

extension Command: Encodable {
    private enum CodingKeys: String, CodingKey {
        case type, text, instructions, toolCallID = "tool_call_id", commandID = "command_id", output, error
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .say(let text):
            try container.encode("say", forKey: .type)
            try container.encode(text, forKey: .text)
        case .respond(let text):
            try container.encode("respond", forKey: .type)
            try container.encode(text, forKey: .text)
        case .respondCommand(let id, let text):
            try container.encode("respond", forKey: .type)
            try container.encode(text, forKey: .text)
            try container.encode(id, forKey: .commandID)
        case .interrupt:
            try container.encode("interrupt", forKey: .type)
        case .interruptCommand(let id):
            try container.encode("interrupt", forKey: .type)
            try container.encode(id, forKey: .commandID)
        case .instructions(let instructions):
            try container.encode("instructions", forKey: .type)
            try container.encode(instructions, forKey: .instructions)
        case .toolResult(let id, let output, let error):
            try container.encode("tool_result", forKey: .type)
            try container.encode(id, forKey: .toolCallID)
            // The router reads both fields off one struct and treats the empty string as
            // absent, so sending the empty string and sending nothing are the same thing.
            try container.encode(output ?? "", forKey: .output)
            try container.encode(error ?? "", forKey: .error)
        case .close:
            try container.encode("close", forKey: .type)
        }
    }
}
