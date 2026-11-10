import Foundation
import StreamCore

/// Continue a conversation as a new one. Everything the parent was opened with is inherited; anything named here is written over it, which is what makes a fork useful rather than a copy -- the usual reason to fork is to ask the same question of a different model.
final class ForkSessionRequest: Sendable, Codable, JSONEncodable {
    let agent: String?
    let configId: String?
    let custom: [String: RawJSON]?
    let description: String?
    /// Hold the fork off the record. The parent still exists; this conversation onwards is simply not kept.
    let incognito: Bool?
    /// Carry the parent's history into the fork, so the new conversation continues from what was already said. False starts the same configuration over from nothing, which is what comparing two answers to the same opening question wants.
    let messages: Bool?
    /// What to change about the models for one session, over whatever its agent config decided.
    /// It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides the config, and a caller reading a session back wants to see what they changed in one place rather than diffed against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because a caller able to rewrite those could make a session impersonate a different agent. A backend sets one session's instructions with the instructions of CreateSessionRequest instead, which a device may not send.
    let modelOverwrites: ModelOverwritesPayload?
    let projectId: String?
    /// Carry the parent's history only up to the end of this response, so the fork continues from that point rather than from where the parent is now. The history is read from what the parent recorded, which also lets a parent that kept no Chat transcript be forked with its history. Cannot be combined with messages false.
    let responseId: String?
    let title: String?

    init(agent: String? = nil, configId: String? = nil, custom: [String: RawJSON]? = nil, description: String? = nil, incognito: Bool? = nil, messages: Bool? = nil, modelOverwrites: ModelOverwritesPayload? = nil, projectId: String? = nil, responseId: String? = nil, title: String? = nil) {
        self.agent = agent
        self.configId = configId
        self.custom = custom
        self.description = description
        self.incognito = incognito
        self.messages = messages
        self.modelOverwrites = modelOverwrites
        self.projectId = projectId
        self.responseId = responseId
        self.title = title
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.agent = try container.decodeIfPresent(String.self, forKey: .agent)
        self.configId = try container.decodeIfPresent(String.self, forKey: .configId)
        self.custom = try container.decodeIfPresent([String: RawJSON].self, forKey: .custom)
        self.description = try container.decodeIfPresent(String.self, forKey: .description)
        self.incognito = try container.decodeIfPresent(Bool.self, forKey: .incognito)
        self.messages = try container.decodeIfPresent(Bool.self, forKey: .messages)
        self.modelOverwrites = try container.decodeIfPresent(ModelOverwritesPayload.self, forKey: .modelOverwrites)
        self.projectId = try container.decodeIfPresent(String.self, forKey: .projectId)
        self.responseId = try container.decodeIfPresent(String.self, forKey: .responseId)
        self.title = try container.decodeIfPresent(String.self, forKey: .title)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(agent, forKey: .agent)
        try container.encodeIfPresent(configId, forKey: .configId)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encodeIfPresent(description, forKey: .description)
        try container.encodeIfPresent(incognito, forKey: .incognito)
        try container.encodeIfPresent(messages, forKey: .messages)
        try container.encodeIfPresent(modelOverwrites, forKey: .modelOverwrites)
        try container.encodeIfPresent(projectId, forKey: .projectId)
        try container.encodeIfPresent(responseId, forKey: .responseId)
        try container.encodeIfPresent(title, forKey: .title)
    }

}

extension ForkSessionRequest: Hashable {
    static func == (lhs: ForkSessionRequest, rhs: ForkSessionRequest) -> Bool {
        lhs.agent == rhs.agent &&
        lhs.configId == rhs.configId &&
        lhs.custom == rhs.custom &&
        lhs.description == rhs.description &&
        lhs.incognito == rhs.incognito &&
        lhs.messages == rhs.messages &&
        lhs.modelOverwrites == rhs.modelOverwrites &&
        lhs.projectId == rhs.projectId &&
        lhs.responseId == rhs.responseId &&
        lhs.title == rhs.title
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(agent)
        hasher.combine(configId)
        hasher.combine(custom)
        hasher.combine(description)
        hasher.combine(incognito)
        hasher.combine(messages)
        hasher.combine(modelOverwrites)
        hasher.combine(projectId)
        hasher.combine(responseId)
        hasher.combine(title)
    }
}
