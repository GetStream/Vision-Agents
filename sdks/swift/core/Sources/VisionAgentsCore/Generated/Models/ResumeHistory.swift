import Foundation
import StreamCore

/// How a conversation is handed back to the model when a session carries it on.
final class ResumeHistory: Sendable, Codable, JSONEncodable {
    /// Hand each reply back with the tool calls it made and what they returned, as they were recorded, so a resumed agent knows what it already looked up and does not run its tools again. Over max_characters the oldest results are replaced with [result trimmed] before any turn is left out, and the conversation's first message is kept. A reply whose calls were not recorded, have been removed with their session or redacted under a retention setting is handed back as text. Results are replayed as recorded: credentials redacted and any result over 8 KB cut to its first 2048 characters. Off by default.
    let includeTools: Bool?
    /// How many characters are handed back, tool calls and results included: about four to a token. The whole of it is sent, and billed, with the first reply after the conversation is resumed, and is cached from then on by models that cache, so keep it well inside the model's context window. Omitted is 60000.
    let maxCharacters: Int?
    /// How many turns are handed back, newest first. Tool calls and their results are not turns. Omitted is 100.
    let maxMessages: Int?

    init(includeTools: Bool? = nil, maxCharacters: Int? = nil, maxMessages: Int? = nil) {
        self.includeTools = includeTools
        self.maxCharacters = maxCharacters
        self.maxMessages = maxMessages
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.includeTools = try container.decodeIfPresent(Bool.self, forKey: .includeTools)
        self.maxCharacters = try container.decodeIfPresent(Int.self, forKey: .maxCharacters)
        self.maxMessages = try container.decodeIfPresent(Int.self, forKey: .maxMessages)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(includeTools, forKey: .includeTools)
        try container.encodeIfPresent(maxCharacters, forKey: .maxCharacters)
        try container.encodeIfPresent(maxMessages, forKey: .maxMessages)
    }

}

extension ResumeHistory: Hashable {
    static func == (lhs: ResumeHistory, rhs: ResumeHistory) -> Bool {
        lhs.includeTools == rhs.includeTools &&
        lhs.maxCharacters == rhs.maxCharacters &&
        lhs.maxMessages == rhs.maxMessages
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(includeTools)
        hasher.combine(maxCharacters)
        hasher.combine(maxMessages)
    }
}
