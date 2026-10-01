import Foundation

/// Selects a user-owned connector account for one session.
public struct SessionConnectorSelection: Sendable, Hashable {
    public let name: String
    public let connectionID: String

    public init(name: String, connectionID: String) {
        self.name = name
        self.connectionID = connectionID
    }
}

/// A running or finished conversation.
public struct Session: Sendable, Hashable, Identifiable {
    /// What the router holds this session by. This addresses the session and its socket, and
    /// it is not the call id.
    public let id: String
    /// The Stream call the agent joined, which is what a video SDK joins. Empty for a text
    /// session.
    public let callID: String
    public let callType: String
    /// Keys the transcript, and names the chat channel it is written to.
    public let agentID: String
    public let isText: Bool
    public let state: State
    public let instructions: String
    public let llm: String
    public let createdAt: Date

    public enum State: String, Sendable {
        case live
        case ended
    }

    init(_ schema: Components.Schemas.Session) {
        id = schema.id
        callID = schema.callId
        callType = schema.callType
        agentID = schema.agentId
        isText = schema.text ?? false
        state = State(rawValue: schema.state.rawValue) ?? .ended
        instructions = schema.instructions ?? ""
        llm = schema.llm ?? ""
        createdAt = schema.createdAt
    }
}

/// One turn of a session as the router wrote it down: what was asked, and how answering it
/// ended. This is what a rewind or a fork names.
public struct Response: Sendable, Hashable, Identifiable {
    public let id: String
    public let sessionID: String
    /// What the person asked. Empty for a turn the agent started on its own, like a greeting.
    public let said: String
    public let status: Status
    /// What went wrong, for a failed turn.
    public let error: String
    public let createdAt: Date
    public let finishedAt: Date?

    public enum Status: String, Sendable {
        case running
        case completed
        case failed
        /// Interrupted by the caller, which is not a failure: what was said still counts.
        case cancelled
        /// A status this SDK has never heard of.
        case unknown
    }

    init(_ schema: Components.Schemas.AgentResponse) {
        id = schema.id
        sessionID = schema.sessionId
        said = schema.said ?? ""
        status = Status(rawValue: schema.status.rawValue) ?? .unknown
        error = schema.error ?? ""
        createdAt = schema.createdAt
        finishedAt = schema.finishedAt
    }
}

/// One thing the agent did while answering a response: what was said, thought, answered,
/// blocked or failed, in the order it happened.
public struct ResponseItem: Sendable, Hashable {
    public let responseID: String
    /// Where this falls within its response.
    public let ordinal: Int
    public let kind: Kind
    public let text: String
    /// The tool called, for an item that records a tool call.
    public let toolName: String

    public enum Kind: String, Sendable {
        case said
        case thought
        case answer
        case blocked
        case error
        /// A kind this SDK has never heard of.
        case unknown
    }

    init(_ schema: Components.Schemas.AgentResponseItem) {
        responseID = schema.responseId
        ordinal = schema.ordinal
        kind = Kind(rawValue: schema.kind.rawValue) ?? .unknown
        text = schema.text ?? ""
        toolName = schema.toolName ?? ""
    }
}

/// An image to show the agent with what is asked, by URL or as a data URI.
public struct ImageSource: Sendable, Hashable {
    public var url: String
    /// How closely the model looks. Nil lets the model decide.
    public var detail: Detail?

    public enum Detail: String, Sendable {
        case auto
        case low
        case high
    }

    public init(url: String, detail: Detail? = nil) {
        self.url = url
        self.detail = detail
    }

    var schema: Components.Schemas.ImageSource {
        .init(url: url, detail: detail.flatMap { .init(rawValue: $0.rawValue) })
    }
}

/// What to change about a conversation while continuing it as a new one.
///
/// `nil` means the fork keeps what the parent had.
public struct ForkOptions: Sendable {
    /// Carry the history only up to the end of this response, and branch from there.
    public var responseID: String?
    /// Another agent config to continue as, by id.
    public var agent: String?
    public var title: String?
    public var instructions: String?
    /// Start the fork with none of the parent's history. Cannot be combined with
    /// `responseID`, which is a point in that history.
    public var withoutHistory = false
    /// The call the fork joins, which a voice session needs and a text session refuses.
    public var callID: String?

    public init(responseID: String? = nil) {
        self.responseID = responseID
    }
}
