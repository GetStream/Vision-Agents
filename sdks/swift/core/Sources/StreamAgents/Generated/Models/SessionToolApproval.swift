import Foundation
import StreamCore

/// Says a person must allow each call before it runs. In a persistent conversation the call's ai_tool_call attachment opens as awaiting_approval, addressed to the person whose command it answers (and, for a client tool, their install), and carries this question for their client to ask. The caller collects the answer and reports it over the events socket with tool_approval: allowed, the call goes on as it would have (awaiting_client for a client tool, running otherwise); declined, it is cancelled. The caller still answers the call with tool_result either way. Every channel member can read the question. On a voice call the router asks instead: it does not send the call, answers the model at once with the title, each {argument} in it filled in from the call, and offers the model confirm_tool_call. A cascade says the question as written; a speech-to-speech model is told to ask it in those words. Only a yes within 15 seconds of the question finishing sends the call, under its own id and with approved true, and the tool's timeout starts then. A no, or silence, declines it, and the call is never sent.
final class SessionToolApproval: Sendable, Codable, JSONEncodable {
    /// The label of the button that allows the call.
    let allowTitle: String?
    /// The label of the button that declines it.
    let declineTitle: String?
    /// What allowing it shares or does, such as "Only your city is shared."
    let message: String?
    /// The argument, a string, in which the model says why it wants this call. Its text (at most 160 characters) is shown as the approval's reason, so it is visible to every channel member even for a server tool.
    let reasonArgument: String?
    /// The question, such as "Share your location?".
    let title: String

    init(allowTitle: String? = nil, declineTitle: String? = nil, message: String? = nil, reasonArgument: String? = nil, title: String) {
        self.allowTitle = allowTitle
        self.declineTitle = declineTitle
        self.message = message
        self.reasonArgument = reasonArgument
        self.title = title
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.allowTitle = try container.decodeIfPresent(String.self, forKey: .allowTitle)
        self.declineTitle = try container.decodeIfPresent(String.self, forKey: .declineTitle)
        self.message = try container.decodeIfPresent(String.self, forKey: .message)
        self.reasonArgument = try container.decodeIfPresent(String.self, forKey: .reasonArgument)
        self.title = try container.decode(String.self, forKey: .title)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(allowTitle, forKey: .allowTitle)
        try container.encodeIfPresent(declineTitle, forKey: .declineTitle)
        try container.encodeIfPresent(message, forKey: .message)
        try container.encodeIfPresent(reasonArgument, forKey: .reasonArgument)
        try container.encode(title, forKey: .title)
    }

}

extension SessionToolApproval: Hashable {
    static func == (lhs: SessionToolApproval, rhs: SessionToolApproval) -> Bool {
        lhs.allowTitle == rhs.allowTitle &&
        lhs.declineTitle == rhs.declineTitle &&
        lhs.message == rhs.message &&
        lhs.reasonArgument == rhs.reasonArgument &&
        lhs.title == rhs.title
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(allowTitle)
        hasher.combine(declineTitle)
        hasher.combine(message)
        hasher.combine(reasonArgument)
        hasher.combine(title)
    }
}
