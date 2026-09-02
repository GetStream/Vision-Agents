import Foundation

/// Who said something.
public struct Participant: Sendable, Hashable {
    public let id: String
    public let userID: String
    public let name: String

    /// What to show for this participant: their name, or their user id when they have none.
    public var display: String { name.isEmpty ? userID : name }

    init(_ fields: [String: JSONValue]) {
        id = fields["id"]?.stringValue ?? ""
        userID = fields["user_id"]?.stringValue ?? ""
        name = fields["name"]?.stringValue ?? ""
    }
}

/// One event on a session's socket.
///
/// The fields are kept as they arrived so that an event added to the router after this SDK
/// shipped still reaches the caller, with `kind` nil and `type` naming it. Switching on `kind`
/// covers what is known; reading `self["whatever"]` covers the rest.
public struct AgentEvent: Sendable, Hashable {
    /// The router's own name for this event, always present.
    public let type: String

    /// The event's fields, flattened as the router sends them.
    public let fields: [String: JSONValue]

    /// This event as one of the kinds the SDK knows, or nil for one it does not.
    public var kind: Kind? { Kind(rawValue: type) }

    public init(type: String, fields: [String: JSONValue] = [:]) {
        self.type = type
        self.fields = fields
    }

    /// The events the router documents. A frame outside this set is still delivered.
    public enum Kind: String, Sendable, CaseIterable {
        case joined
        case participantJoined = "participant_joined"
        case participantLeft = "participant_left"
        case hearing
        case heard
        case decision
        case responding
        case responseDelta = "response_delta"
        case responded
        case spoke
        case turn
        case delegated
        case taskSettled = "task_settled"
        case taskCancelled = "task_cancelled"
        case toolCall = "tool_call"
        case toolRan = "tool_ran"
        case transferred
        case pressed
        case lookedUp = "looked_up"
        case backchannel
        case interrupted
        case overlapDecided = "overlap_decided"
        case conversationCompacted = "conversation_compacted"
        case error
        case left
    }

    public subscript(key: String) -> JSONValue {
        fields[key] ?? .null
    }
}

extension AgentEvent {
    /// What was said, transcribed or generated, depending on the event.
    public var text: String { self["text"].stringValue }

    /// The turn this belongs to, or the empty string for events outside a turn.
    public var turnID: String { self["turn_id"].stringValue }

    /// Who this is about, or nil for the events that are about nobody.
    public var participant: Participant? {
        guard case .object(let fields) = self["participant"] else { return nil }
        return Participant(fields)
    }

    /// What went wrong, for `error` and for the events that carry a failure of their own.
    public var errorText: String { self["error"].stringValue }
}

extension AgentEvent {
    /// A tool the model wants run, or nil when this event is not a tool call.
    public var toolCall: ToolCall? {
        guard kind == .toolCall else { return nil }
        return ToolCall(
            id: self["id"].stringValue,
            name: self["name"].stringValue,
            arguments: self["arguments"].stringValue)
    }

    /// A request from the model to run one of the caller's functions.
    public struct ToolCall: Sendable, Hashable {
        public let id: String
        public let name: String
        /// The arguments as the model wrote them, which is a JSON object encoded as a string.
        public let arguments: String

        /// The arguments decoded, or an empty dictionary if the model wrote something else.
        public var argumentValues: [String: JSONValue] {
            guard let data = arguments.data(using: .utf8),
                let decoded = try? JSONDecoder().decode(JSONValue.self, from: data)
            else { return [:] }
            return decoded.objectValue
        }
    }
}

extension AgentEvent: Decodable {
    public init(from decoder: Decoder) throws {
        let fields = try decoder.singleValueContainer().decode([String: JSONValue].self)
        guard case .string(let type)? = fields["type"] else {
            throw DecodingError.dataCorruptedError(
                in: try decoder.singleValueContainer(),
                debugDescription: "a session frame needs a type")
        }
        self.init(type: type, fields: fields.filter { $0.key != "type" })
    }
}
