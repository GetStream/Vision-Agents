import Foundation
import StreamCore

final class SessionToolExecutor: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let client = SessionToolExecutor(rawValue: "client")
    static let server = SessionToolExecutor(rawValue: "server")
}

/// One of the caller's own functions. The model is offered it by name and description; running it is the caller's business, over the events socket.
final class SessionTool: Sendable, Codable, JSONEncodable {
    /// Says a person must allow each call before it runs. In a persistent conversation the call's ai_tool_call attachment opens as awaiting_approval, addressed to the person whose command it answers (and, for a client tool, their install), and carries this question for their client to ask. The caller collects the answer and reports it over the events socket with tool_approval: allowed, the call goes on as it would have (awaiting_client for a client tool, running otherwise); declined, it is cancelled. The caller still answers the call with tool_result either way. Every channel member can read the question. On a voice call the router asks instead: it does not send the call, answers the model at once with the title, each {argument} in it filled in from the call, and offers the model confirm_tool_call. A cascade says the question as written; a speech-to-speech model is told to ask it in those words. Only a yes within 15 seconds of the question finishing sends the call, under its own id and with approved true, and the tool's timeout starts then. A no, or silence, declines it, and the call is never sent.
    let approval: SessionToolApproval?
    /// What the model is told the tool does, which is the whole of how it decides when to reach for one.
    let description: String
    /// What a call is doing, in words for the people in the conversation, such as "Checking your location". Shown on the reply's ai_tool_call attachment.
    let displayTitle: String?
    /// Who runs it. A client tool runs on a person's device: in a persistent conversation its call is shown as awaiting the device of the person whose command it answers (their user and the command's client_id), with its arguments, which every channel member can read. The caller still answers it over the events socket, once the device has reported. Defaults to server.
    let executor: SessionToolExecutor?
    /// Whether the caller interrupting the agent cancels a call in flight: the events socket is sent tool_cancel for it, and the model answers the new turn without it. Off, the default, the call goes on and its result is told when it comes.
    let interruptible: Bool?
    let name: String
    /// A JSON Schema object describing the arguments.
    let parameters: [String: RawJSON]?
    /// How long one call may take before the model is told it failed and tool_cancel is sent, in place of the session's tool_timeout_ms. Zero or omitted takes the session's.
    let timeoutMs: Int?

    init(approval: SessionToolApproval? = nil, description: String, displayTitle: String? = nil, executor: SessionToolExecutor? = nil, interruptible: Bool? = nil, name: String, parameters: [String: RawJSON]? = nil, timeoutMs: Int? = nil) {
        self.approval = approval
        self.description = description
        self.displayTitle = displayTitle
        self.executor = executor
        self.interruptible = interruptible
        self.name = name
        self.parameters = parameters
        self.timeoutMs = timeoutMs
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.approval = try container.decodeIfPresent(SessionToolApproval.self, forKey: .approval)
        self.description = try container.decode(String.self, forKey: .description)
        self.displayTitle = try container.decodeIfPresent(String.self, forKey: .displayTitle)
        self.executor = try container.decodeIfPresent(SessionToolExecutor.self, forKey: .executor)
        self.interruptible = try container.decodeIfPresent(Bool.self, forKey: .interruptible)
        self.name = try container.decode(String.self, forKey: .name)
        self.parameters = try container.decodeIfPresent([String: RawJSON].self, forKey: .parameters)
        self.timeoutMs = try container.decodeIfPresent(Int.self, forKey: .timeoutMs)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(approval, forKey: .approval)
        try container.encode(description, forKey: .description)
        try container.encodeIfPresent(displayTitle, forKey: .displayTitle)
        try container.encodeIfPresent(executor, forKey: .executor)
        try container.encodeIfPresent(interruptible, forKey: .interruptible)
        try container.encode(name, forKey: .name)
        try container.encodeIfPresent(parameters, forKey: .parameters)
        try container.encodeIfPresent(timeoutMs, forKey: .timeoutMs)
    }

}

extension SessionTool: Hashable {
    static func == (lhs: SessionTool, rhs: SessionTool) -> Bool {
        lhs.approval == rhs.approval &&
        lhs.description == rhs.description &&
        lhs.displayTitle == rhs.displayTitle &&
        lhs.executor == rhs.executor &&
        lhs.interruptible == rhs.interruptible &&
        lhs.name == rhs.name &&
        lhs.parameters == rhs.parameters &&
        lhs.timeoutMs == rhs.timeoutMs
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(approval)
        hasher.combine(description)
        hasher.combine(displayTitle)
        hasher.combine(executor)
        hasher.combine(interruptible)
        hasher.combine(name)
        hasher.combine(parameters)
        hasher.combine(timeoutMs)
    }
}
