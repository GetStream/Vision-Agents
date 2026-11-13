import Foundation
import StreamCore

final class AgentResponseStatus: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let cancelled = AgentResponseStatus(rawValue: "cancelled")
    static let completed = AgentResponseStatus(rawValue: "completed")
    static let failed = AgentResponseStatus(rawValue: "failed")
    static let refused = AgentResponseStatus(rawValue: "refused")
    static let running = AgentResponseStatus(rawValue: "running")
}

final class AgentResponse: Sendable, Codable, JSONEncodable {
    let branch: ResponseBranch?
    let createdAt: Date
    let error: String?
    let finishedAt: Date?
    let id: String
    /// The response this one was answered after, absent for a session's first and for one answered before responses were kept as a tree.
    let parentResponseId: String?
    let rewoundTo: RewoundTo?
    /// What the person asked, which is the first item of every response.
    let said: String?
    let sessionId: String
    /// cancelled is a turn the caller interrupted, which is a different thing from one that failed: nothing went wrong, and what had already been said still counts. refused is a turn the prompt injection screen refused: a rewind or a fork carries what it asked forward as withheld.
    let status: AgentResponseStatus

    init(branch: ResponseBranch? = nil, createdAt: Date, error: String? = nil, finishedAt: Date? = nil, id: String, parentResponseId: String? = nil, rewoundTo: RewoundTo? = nil, said: String? = nil, sessionId: String, status: AgentResponseStatus) {
        self.branch = branch
        self.createdAt = createdAt
        self.error = error
        self.finishedAt = finishedAt
        self.id = id
        self.parentResponseId = parentResponseId
        self.rewoundTo = rewoundTo
        self.said = said
        self.sessionId = sessionId
        self.status = status
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.branch = try container.decodeIfPresent(ResponseBranch.self, forKey: .branch)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
        self.error = try container.decodeIfPresent(String.self, forKey: .error)
        self.finishedAt = try container.decodeIfPresent(Date.self, forKey: .finishedAt)
        self.id = try container.decode(String.self, forKey: .id)
        self.parentResponseId = try container.decodeIfPresent(String.self, forKey: .parentResponseId)
        self.rewoundTo = try container.decodeIfPresent(RewoundTo.self, forKey: .rewoundTo)
        self.said = try container.decodeIfPresent(String.self, forKey: .said)
        self.sessionId = try container.decode(String.self, forKey: .sessionId)
        self.status = try container.decode(AgentResponseStatus.self, forKey: .status)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(branch, forKey: .branch)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encodeIfPresent(error, forKey: .error)
        try container.encodeIfPresent(finishedAt, forKey: .finishedAt)
        try container.encode(id, forKey: .id)
        try container.encodeIfPresent(parentResponseId, forKey: .parentResponseId)
        try container.encodeIfPresent(rewoundTo, forKey: .rewoundTo)
        try container.encodeIfPresent(said, forKey: .said)
        try container.encode(sessionId, forKey: .sessionId)
        try container.encode(status, forKey: .status)
    }

}

extension AgentResponse: Hashable {
    static func == (lhs: AgentResponse, rhs: AgentResponse) -> Bool {
        lhs.branch == rhs.branch &&
        lhs.createdAt == rhs.createdAt &&
        lhs.error == rhs.error &&
        lhs.finishedAt == rhs.finishedAt &&
        lhs.id == rhs.id &&
        lhs.parentResponseId == rhs.parentResponseId &&
        lhs.rewoundTo == rhs.rewoundTo &&
        lhs.said == rhs.said &&
        lhs.sessionId == rhs.sessionId &&
        lhs.status == rhs.status
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(branch)
        hasher.combine(createdAt)
        hasher.combine(error)
        hasher.combine(finishedAt)
        hasher.combine(id)
        hasher.combine(parentResponseId)
        hasher.combine(rewoundTo)
        hasher.combine(said)
        hasher.combine(sessionId)
        hasher.combine(status)
    }
}
