import Foundation
import StreamCore

/// Who said something.
public struct Participant: Sendable, Hashable {
    public let id: String
    public let userID: String
    public let name: String

    /// What to show for this participant: their name, or their user id when they have none.
    public var display: String { name.isEmpty ? userID : name }

    init(_ fields: [String: RawJSON]) {
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
    public let fields: [String: RawJSON]

    /// This event as one of the kinds the SDK knows, or nil for one it does not.
    public var kind: Kind? { Kind(rawValue: type) }

    public init(type: String, fields: [String: RawJSON] = [:]) {
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
        case reasoningDelta = "reasoning_delta"
        case uiDelta = "ui_delta"
        case uiCompleted = "ui_completed"
        case responded
        case spoke
        case turn
        case delegated
        case taskSettled = "task_settled"
        case taskCancelled = "task_cancelled"
        case toolCall = "tool_call"
        case toolCancel = "tool_cancel"
        case toolHeld = "tool_held"
        case toolStarted = "tool_started"
        case toolUpdated = "tool_updated"
        case toolRan = "tool_ran"
        case toolCallStarted = "tool_call.started"
        case toolCallProgress = "tool_call.progress"
        case toolCallFinished = "tool_call.finished"
        case transferred
        case pressed
        case lookedUp = "looked_up"
        case backchannel
        case interrupted
        case overlapDecided = "overlap_decided"
        case conversationCompacted = "conversation_compacted"
        case connectorScopeRequired = "connector_scope_required"
        case error
        case left
    }

    public subscript(key: String) -> RawJSON {
        fields[key] ?? .nil
    }

    /// The string under `key`, or the empty string for anything else.
    ///
    /// Absent and empty are one case on purpose: the router writes `""` for a field that does
    /// not apply, so a caller that told them apart would be distinguishing nothing.
    func string(_ key: String) -> String {
        self[key].stringValue ?? ""
    }
}

extension AgentEvent {
    /// What was said, transcribed or generated, depending on the event.
    public var text: String { string("text") }

    /// The turn this belongs to, or the empty string for events outside a turn.
    public var turnID: String { string("turn_id") }

    /// Who this is about, or nil for the events that are about nobody.
    public var participant: Participant? {
        guard case .dictionary(let fields) = self["participant"] else { return nil }
        return Participant(fields)
    }

    /// What went wrong, for `error` and for the events that carry a failure of their own.
    public var errorText: String { string("error") }
}

extension AgentEvent {
    /// A tool the model wants run, or nil when this event is not a tool call.
    public var toolCall: ToolCall? {
        guard kind == .toolCall else { return nil }
        return ToolCall(
            id: string("id"),
            name: string("name"),
            arguments: string("arguments"),
            requestID: string("request_id"),
            turnID: string("turn_id"),
            run: self["run"].boolValue ?? true,
            approved: self["approved"].boolValue ?? false)
    }

    /// A request from the model to run one of the caller's functions.
    public struct ToolCall: Sendable, Hashable {
        public let id: String
        public let name: String
        /// The arguments as the model wrote them, which is a JSON object encoded as a string.
        public let arguments: String
        /// The request and turn that asked, which its result must name again.
        public var requestID = ""
        public var turnID = ""
        /// Whether this device is the one to run it. Every watcher is sent every call, and
        /// only the one the router names answers it.
        public var run = true
        /// Whether the person already said yes to it, on a voice call, so nobody asks again.
        public var approved = false

        /// The arguments decoded, or an empty dictionary if the model wrote something else.
        public var argumentValues: [String: RawJSON] {
            guard let data = arguments.data(using: .utf8),
                let decoded = try? JSONDecoder().decode([String: RawJSON].self, from: data)
            else { return [:] }
            return decoded
        }
    }
}

extension AgentEvent {
    /// The call's whole state on a `tool_call.started`, `tool_call.progress` or
    /// `tool_call.finished`, or nil for any other event.
    public var toolCallState: ToolCallState? {
        guard kind == .toolCallStarted || kind == .toolCallProgress || kind == .toolCallFinished,
            !string("tool_call_id").isEmpty
        else { return nil }
        let dates = RouterDates()
        let date = { (key: String) in try? dates.decode(self.string(key)) }
        return ToolCallState(
            id: string("tool_call_id"),
            tool: string("tool"),
            turnID: string("turn_id"),
            status: ToolCallState.Status(rawValue: string("status")),
            progress: string("progress"),
            startedAt: date("started_at"),
            durationMS: Int(self["duration_ms"].numberValue ?? 0),
            approvedAt: date("approved_at"),
            rejectedAt: date("rejected_at"),
            error: string("error"))
    }
}

/// One tool call as the session last reported it, whichever process runs it.
public struct ToolCallState: Sendable, Hashable, Identifiable {
    /// The call's `tool_call_id`.
    public let id: String
    public let tool: String
    public let turnID: String
    public let status: Status
    /// The last thing said about the call while it runs, such as "Searching for flights", or
    /// the question it waits on. Empty for none.
    public let progress: String
    public let startedAt: Date?
    /// From `startedAt` to this report; once finished, how long the call took.
    public let durationMS: Int
    /// When a person allowed the call.
    public let approvedAt: Date?
    /// When a person declined the call, or its wait for them passed.
    public let rejectedAt: Date?
    /// Why a failed call failed, or what a declined one shows. Empty for none.
    public let error: String

    /// Where a call stands. A value the router added after this SDK shipped is kept as it came.
    public struct Status: RawRepresentable, Sendable, Hashable {
        public let rawValue: String
        public init(rawValue: String) { self.rawValue = rawValue }

        public static let inProgress = Status(rawValue: "in_progress")
        /// Held for a person's approval.
        public static let waiting = Status(rawValue: "waiting")
        public static let success = Status(rawValue: "success")
        public static let failed = Status(rawValue: "failed")
        public static let cancelled = Status(rawValue: "cancelled")

        /// Whether the call is over.
        public var isFinished: Bool { self == .success || self == .failed || self == .cancelled }
    }
}

extension AgentEvent {
    /// A connector tool call the provider refused because the person's own connection lacks
    /// access it asked for, and the step-up consent the router began for it, or nil when this
    /// event is not one. The old grant keeps working until the step-up succeeds, and the same
    /// call works afterwards in the same session.
    ///
    /// An app cannot finish the step-up yet: `launchURL` takes `handoffToken` only from a page
    /// on the dashboard's origin, as for a `connector_authorization` attachment.
    public var connectorScopeRequired: ConnectorScopeRequired? {
        guard kind == .connectorScopeRequired else { return nil }
        return ConnectorScopeRequired(
            name: string("name"),
            connectorID: string("connector_id"),
            connectionID: string("connection_id"),
            scopes: self["scopes"].stringArrayValue ?? [],
            authorizationID: string("authorization_id"),
            launchURL: string("launch_url"),
            handoffToken: string("handoff_token"),
            expiresAt: try? RouterDates().decode(string("expires_at")))
    }

    /// The access a connector tool call needs, and where the person grants it.
    public struct ConnectorScopeRequired: Sendable, Hashable {
        /// The binding's alias.
        public let name: String
        public let connectorID: String
        public let connectionID: String
        /// What the provider asked for, empty for a claims challenge.
        public let scopes: [String]
        public let authorizationID: String
        public let launchURL: String
        public let handoffToken: String
        public let expiresAt: Date?
    }
}

extension AgentEvent: Decodable {
    public init(from decoder: Decoder) throws {
        let fields = try decoder.singleValueContainer().decode([String: RawJSON].self)
        guard case .string(let type)? = fields["type"] else {
            throw DecodingError.dataCorruptedError(
                in: try decoder.singleValueContainer(),
                debugDescription: "a session frame needs a type")
        }
        self.init(type: type, fields: fields.filter { $0.key != "type" })
    }
}
