import Foundation
import StreamCore

final class AgentResponseItemKind: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let answer = AgentResponseItemKind(rawValue: "answer")
    static let blocked = AgentResponseItemKind(rawValue: "blocked")
    static let error = AgentResponseItemKind(rawValue: "error")
    static let said = AgentResponseItemKind(rawValue: "said")
    static let thought = AgentResponseItemKind(rawValue: "thought")
    static let toolCall = AgentResponseItemKind(rawValue: "tool_call")
    static let toolResult = AgentResponseItemKind(rawValue: "tool_result")
}

final class AgentResponseItem: Sendable, Codable, JSONEncodable {
    let at: Date
    let kind: AgentResponseItemKind
    /// The position within the response, assigned by the writer rather than by the database, so items keep the order they happened in.
    let ordinal: Int
    /// Whatever the kind carries that text cannot: a tool's arguments, a guardrail's reason, the id that ties a call to its result. A tool item whose output a tool output retention took out has redacted: true, no arguments and no text, and error: redacted in place of an error's wording.
    let payload: [String: RawJSON]?
    let responseId: String
    let sessionId: String?
    let text: String?
    let toolName: String?

    init(at: Date, kind: AgentResponseItemKind, ordinal: Int, payload: [String: RawJSON]? = nil, responseId: String, sessionId: String? = nil, text: String? = nil, toolName: String? = nil) {
        self.at = at
        self.kind = kind
        self.ordinal = ordinal
        self.payload = payload
        self.responseId = responseId
        self.sessionId = sessionId
        self.text = text
        self.toolName = toolName
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.at = try container.decode(Date.self, forKey: .at)
        self.kind = try container.decode(AgentResponseItemKind.self, forKey: .kind)
        self.ordinal = try container.decode(Int.self, forKey: .ordinal)
        self.payload = try container.decodeIfPresent([String: RawJSON].self, forKey: .payload)
        self.responseId = try container.decode(String.self, forKey: .responseId)
        self.sessionId = try container.decodeIfPresent(String.self, forKey: .sessionId)
        self.text = try container.decodeIfPresent(String.self, forKey: .text)
        self.toolName = try container.decodeIfPresent(String.self, forKey: .toolName)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(at, forKey: .at)
        try container.encode(kind, forKey: .kind)
        try container.encode(ordinal, forKey: .ordinal)
        try container.encodeIfPresent(payload, forKey: .payload)
        try container.encode(responseId, forKey: .responseId)
        try container.encodeIfPresent(sessionId, forKey: .sessionId)
        try container.encodeIfPresent(text, forKey: .text)
        try container.encodeIfPresent(toolName, forKey: .toolName)
    }

}

extension AgentResponseItem: Hashable {
    static func == (lhs: AgentResponseItem, rhs: AgentResponseItem) -> Bool {
        lhs.at == rhs.at &&
        lhs.kind == rhs.kind &&
        lhs.ordinal == rhs.ordinal &&
        lhs.payload == rhs.payload &&
        lhs.responseId == rhs.responseId &&
        lhs.sessionId == rhs.sessionId &&
        lhs.text == rhs.text &&
        lhs.toolName == rhs.toolName
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(at)
        hasher.combine(kind)
        hasher.combine(ordinal)
        hasher.combine(payload)
        hasher.combine(responseId)
        hasher.combine(sessionId)
        hasher.combine(text)
        hasher.combine(toolName)
    }
}
