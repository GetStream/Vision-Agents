import Foundation

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
