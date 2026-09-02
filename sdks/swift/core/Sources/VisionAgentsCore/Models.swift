import Foundation

/// A stored agent configuration: what an agent is, before a session starts one.
public struct AgentConfig: Sendable, Hashable, Identifiable {
    public let id: String
    public let name: String
    /// Whether this agent talks or writes.
    public let mode: Mode
    public let instructions: String
    public let greeting: String
    public let llm: String
    public let skills: [String]
    public let updatedAt: Date

    public enum Mode: String, Sendable {
        case voice
        case text
    }

    init(_ schema: Components.Schemas.AgentConfig) {
        id = schema.id
        name = schema.name
        mode = Mode(rawValue: schema.mode.rawValue) ?? .voice
        instructions = schema.instructions ?? ""
        greeting = schema.greeting ?? ""
        llm = schema.llm ?? ""
        skills = schema.skills ?? []
        updatedAt = schema.updatedAt
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

/// Credentials for joining the Stream call an agent is on.
public struct CallToken: Sendable, Hashable {
    public let apiKey: String
    public let token: String
    public let userID: String
    public let userName: String
    /// The Stream call to join, which is not the id the router holds the call by.
    public let callID: String
    public let callType: String
    public let expiresAt: Date

    init(_ schema: Components.Schemas.CallToken) {
        apiKey = schema.apiKey
        token = schema.token
        userID = schema.userId
        userName = schema.userName
        callID = schema.callId
        callType = schema.callType
        expiresAt = schema.expiresAt
    }
}

/// Credentials for reading and writing an agent's conversation channel.
public struct ChatToken: Sendable, Hashable {
    public let apiKey: String
    public let token: String
    public let userID: String
    public let userName: String
    public let channelType: String
    public let channelID: String
    public let expiresAt: Date

    init(_ schema: Components.Schemas.ChatToken) {
        apiKey = schema.apiKey
        token = schema.token
        userID = schema.userId
        userName = schema.userName
        channelType = schema.channelType
        channelID = schema.channelId
        expiresAt = schema.expiresAt
    }
}

/// One thing said on a call, from the stored transcript.
public struct TranscriptMessage: Sendable, Hashable {
    /// Who said it, the agent under its own user id.
    public let speaker: String
    public let text: String
    public let createdAt: Date

    init(_ schema: Components.Schemas.TranscriptMessage) {
        speaker = schema.speaker
        text = schema.text
        createdAt = schema.createdAt
    }
}

/// The router's record of a call.
///
/// Named for the record rather than the call because Stream's Video SDK has a `Call` of its
/// own, and an app that holds a conversation has both in scope.
public struct CallRecord: Sendable, Hashable, Identifiable {
    public let id: String
    public let callID: String
    public let agentID: String
    public let direction: String
    public let startedAt: Date
    public let endedAt: Date?
    public let summary: String

    /// Whether the call is still going.
    public var isRunning: Bool { endedAt == nil }

    init(_ schema: Components.Schemas.Call) {
        id = schema.id
        callID = schema.callId
        agentID = schema.agentId
        direction = schema.direction.rawValue
        startedAt = schema.startedAt
        endedAt = schema.endedAt
        summary = schema.summary ?? ""
    }
}
