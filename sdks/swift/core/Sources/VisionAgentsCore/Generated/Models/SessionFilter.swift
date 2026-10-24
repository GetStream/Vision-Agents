import Foundation
import StreamCore

/// Which sessions to list. A field not listed here is refused rather than ignored.
final class SessionFilter: Sendable, Codable, JSONEncodable {
    /// The agent name the session was opened against.
    let agent: String?
    /// The agent id the session was created with, which names its transcript channel.
    let agentId: String?
    /// The agent config the session ran under. Empty for a session that spelled itself out.
    let configId: String?
    let createdAt: TimeRange?
    /// The session's custom object holds every one of these pairs, which is how a caller finds again what it labelled.
    let custom: [String: String]?
    /// text, voice or video: how the user took part.
    let modality: String?
    let projectId: String?
    /// live or ended, as each session reports its state.
    let state: String?
    let text: TextMatch?
    /// Whose sessions to list. Only a server-side caller may set it: an end user is narrowed to their own whatever they ask for.
    let userId: String?

    init(agent: String? = nil, agentId: String? = nil, configId: String? = nil, createdAt: TimeRange? = nil, custom: [String: String]? = nil, modality: String? = nil, projectId: String? = nil, state: String? = nil, text: TextMatch? = nil, userId: String? = nil) {
        self.agent = agent
        self.agentId = agentId
        self.configId = configId
        self.createdAt = createdAt
        self.custom = custom
        self.modality = modality
        self.projectId = projectId
        self.state = state
        self.text = text
        self.userId = userId
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.agent = try container.decodeIfPresent(String.self, forKey: .agent)
        self.agentId = try container.decodeIfPresent(String.self, forKey: .agentId)
        self.configId = try container.decodeIfPresent(String.self, forKey: .configId)
        self.createdAt = try container.decodeIfPresent(TimeRange.self, forKey: .createdAt)
        self.custom = try container.decodeIfPresent([String: String].self, forKey: .custom)
        self.modality = try container.decodeIfPresent(String.self, forKey: .modality)
        self.projectId = try container.decodeIfPresent(String.self, forKey: .projectId)
        self.state = try container.decodeIfPresent(String.self, forKey: .state)
        self.text = try container.decodeIfPresent(TextMatch.self, forKey: .text)
        self.userId = try container.decodeIfPresent(String.self, forKey: .userId)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(agent, forKey: .agent)
        try container.encodeIfPresent(agentId, forKey: .agentId)
        try container.encodeIfPresent(configId, forKey: .configId)
        try container.encodeIfPresent(createdAt, forKey: .createdAt)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encodeIfPresent(modality, forKey: .modality)
        try container.encodeIfPresent(projectId, forKey: .projectId)
        try container.encodeIfPresent(state, forKey: .state)
        try container.encodeIfPresent(text, forKey: .text)
        try container.encodeIfPresent(userId, forKey: .userId)
    }

}

extension SessionFilter: Hashable {
    static func == (lhs: SessionFilter, rhs: SessionFilter) -> Bool {
        lhs.agent == rhs.agent &&
        lhs.agentId == rhs.agentId &&
        lhs.configId == rhs.configId &&
        lhs.createdAt == rhs.createdAt &&
        lhs.custom == rhs.custom &&
        lhs.modality == rhs.modality &&
        lhs.projectId == rhs.projectId &&
        lhs.state == rhs.state &&
        lhs.text == rhs.text &&
        lhs.userId == rhs.userId
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(agent)
        hasher.combine(agentId)
        hasher.combine(configId)
        hasher.combine(createdAt)
        hasher.combine(custom)
        hasher.combine(modality)
        hasher.combine(projectId)
        hasher.combine(state)
        hasher.combine(text)
        hasher.combine(userId)
    }
}
