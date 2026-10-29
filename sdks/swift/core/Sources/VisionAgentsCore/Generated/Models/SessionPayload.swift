import Foundation
import StreamCore

final class SessionPayload: Sendable, Codable, JSONEncodable {
    /// The name the agent was addressed as. Recorded on the session as well as the config id, so renaming a config does not rewrite what older sessions were opened against.
    let agent: String?
    let agentId: String
    /// The call the agent is on while voice is started: the session's id unless a call was named. Empty while the conversation is held in writing.
    let callId: String
    /// How the call the router placed for a session ended.
    let callOutcome: SessionCallOutcome?
    let callType: String
    /// When the session ended. Absent while it is still running.
    let closedAt: Date?
    /// The agent config the session ran under, empty for one that spelled itself out.
    let configId: String?
    /// Older history was omitted from the model context.
    let contextTruncated: Bool?
    /// The Stream Chat channel the conversation is kept in, typed and spoken. Absent for an incognito session.
    let conversationId: String?
    let createdAt: Date
    let custom: [String: RawJSON]?
    let description: String?
    /// The session this one continued from, empty for one opened fresh.
    let forkedFrom: String?
    let id: String
    /// Nothing about this session was recorded. It is reported so a caller can see that what they asked for is what they got, but it is never read back from storage: an incognito session has no row to read it from.
    let incognito: Bool?
    /// What the agent was told, the session's own or its config's. Read by a backend only: a device is never given it.
    let instructions: String?
    /// When the agent last answered. This is what a most-recently-used ordering of conversations reads, since a session renamed long after it ended has not become more recent.
    let lastResponseAt: Date?
    /// The provider and model answering, once routing has picked one.
    let llm: String?
    /// How the user took part: text for a conversation held in writing, voice for a call, and video once the agent has seen the user's video. It only moves up, from text or voice to video.
    let modality: SessionModality
    /// How the session hears and speaks: a transcriber, a conversation model and a voice; one speech-to-speech model; or in writing.
    let mode: SessionMode?
    /// What to change about the models for one session, over whatever its agent config decided.
    /// It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides the config, and a caller reading a session back wants to see what they changed in one place rather than diffed against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because a caller able to rewrite those could make a session impersonate a different agent. A backend sets one session's instructions with the instructions of CreateSessionRequest instead, which a device may not send.
    let modelOverwrites: ModelOverwritesPayload?
    let projectId: String?
    /// Why the review scored the conversation as it did. Read by a backend only.
    let reviewNotes: String?
    /// How well the agent handled the conversation, from 1 to 5, written a few seconds after it ended. Read by a backend only.
    let reviewScore: Int?
    /// Whether the agent is still in the call.
    let state: SessionState
    /// The provider and model holding a native conversation, once routing has picked one.
    let sts: String?
    /// The provider and model transcribing, once somebody has been heard.
    let stt: String?
    /// The provider and model delegated work runs on.
    let subagent: String?
    /// What a model made of the conversation, written a few seconds after it ended. A session nobody spoke in has none. Read by a backend only.
    let summary: String?
    /// The conversation is held in writing rather than on a call.
    let text: Bool?
    let title: String?
    /// The provider and model speaking.
    let tts: String?
    /// What the call spent, summed over every request it made. Counted once the call is over, so it is absent while one is still running. Requests that failed are included: a model that read the prompt and then fell over is still billed for it.
    let usage: CallUsage?
    let userId: String
    let video: SessionVideo?
    /// The voice speaking, in the provider's own terms. It is the provider's default when the session asked for none.
    let voice: String?

    init(agent: String? = nil, agentId: String, callId: String, callOutcome: SessionCallOutcome? = nil, callType: String, closedAt: Date? = nil, configId: String? = nil, contextTruncated: Bool? = nil, conversationId: String? = nil, createdAt: Date, custom: [String: RawJSON]? = nil, description: String? = nil, forkedFrom: String? = nil, id: String, incognito: Bool? = nil, instructions: String? = nil, lastResponseAt: Date? = nil, llm: String? = nil, modality: SessionModality, mode: SessionMode? = nil, modelOverwrites: ModelOverwritesPayload? = nil, projectId: String? = nil, reviewNotes: String? = nil, reviewScore: Int? = nil, state: SessionState, sts: String? = nil, stt: String? = nil, subagent: String? = nil, summary: String? = nil, text: Bool? = nil, title: String? = nil, tts: String? = nil, usage: CallUsage? = nil, userId: String, video: SessionVideo? = nil, voice: String? = nil) {
        self.agent = agent
        self.agentId = agentId
        self.callId = callId
        self.callOutcome = callOutcome
        self.callType = callType
        self.closedAt = closedAt
        self.configId = configId
        self.contextTruncated = contextTruncated
        self.conversationId = conversationId
        self.createdAt = createdAt
        self.custom = custom
        self.description = description
        self.forkedFrom = forkedFrom
        self.id = id
        self.incognito = incognito
        self.instructions = instructions
        self.lastResponseAt = lastResponseAt
        self.llm = llm
        self.modality = modality
        self.mode = mode
        self.modelOverwrites = modelOverwrites
        self.projectId = projectId
        self.reviewNotes = reviewNotes
        self.reviewScore = reviewScore
        self.state = state
        self.sts = sts
        self.stt = stt
        self.subagent = subagent
        self.summary = summary
        self.text = text
        self.title = title
        self.tts = tts
        self.usage = usage
        self.userId = userId
        self.video = video
        self.voice = voice
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.agent = try container.decodeIfPresent(String.self, forKey: .agent)
        self.agentId = try container.decode(String.self, forKey: .agentId)
        self.callId = try container.decode(String.self, forKey: .callId)
        self.callOutcome = try container.decodeIfPresent(SessionCallOutcome.self, forKey: .callOutcome)
        self.callType = try container.decode(String.self, forKey: .callType)
        self.closedAt = try container.decodeIfPresent(Date.self, forKey: .closedAt)
        self.configId = try container.decodeIfPresent(String.self, forKey: .configId)
        self.contextTruncated = try container.decodeIfPresent(Bool.self, forKey: .contextTruncated)
        self.conversationId = try container.decodeIfPresent(String.self, forKey: .conversationId)
        self.createdAt = try container.decode(Date.self, forKey: .createdAt)
        self.custom = try container.decodeIfPresent([String: RawJSON].self, forKey: .custom)
        self.description = try container.decodeIfPresent(String.self, forKey: .description)
        self.forkedFrom = try container.decodeIfPresent(String.self, forKey: .forkedFrom)
        self.id = try container.decode(String.self, forKey: .id)
        self.incognito = try container.decodeIfPresent(Bool.self, forKey: .incognito)
        self.instructions = try container.decodeIfPresent(String.self, forKey: .instructions)
        self.lastResponseAt = try container.decodeIfPresent(Date.self, forKey: .lastResponseAt)
        self.llm = try container.decodeIfPresent(String.self, forKey: .llm)
        self.modality = try container.decode(SessionModality.self, forKey: .modality)
        self.mode = try container.decodeIfPresent(SessionMode.self, forKey: .mode)
        self.modelOverwrites = try container.decodeIfPresent(ModelOverwritesPayload.self, forKey: .modelOverwrites)
        self.projectId = try container.decodeIfPresent(String.self, forKey: .projectId)
        self.reviewNotes = try container.decodeIfPresent(String.self, forKey: .reviewNotes)
        self.reviewScore = try container.decodeIfPresent(Int.self, forKey: .reviewScore)
        self.state = try container.decode(SessionState.self, forKey: .state)
        self.sts = try container.decodeIfPresent(String.self, forKey: .sts)
        self.stt = try container.decodeIfPresent(String.self, forKey: .stt)
        self.subagent = try container.decodeIfPresent(String.self, forKey: .subagent)
        self.summary = try container.decodeIfPresent(String.self, forKey: .summary)
        self.text = try container.decodeIfPresent(Bool.self, forKey: .text)
        self.title = try container.decodeIfPresent(String.self, forKey: .title)
        self.tts = try container.decodeIfPresent(String.self, forKey: .tts)
        self.usage = try container.decodeIfPresent(CallUsage.self, forKey: .usage)
        self.userId = try container.decode(String.self, forKey: .userId)
        self.video = try container.decodeIfPresent(SessionVideo.self, forKey: .video)
        self.voice = try container.decodeIfPresent(String.self, forKey: .voice)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(agent, forKey: .agent)
        try container.encode(agentId, forKey: .agentId)
        try container.encode(callId, forKey: .callId)
        try container.encodeIfPresent(callOutcome, forKey: .callOutcome)
        try container.encode(callType, forKey: .callType)
        try container.encodeIfPresent(closedAt, forKey: .closedAt)
        try container.encodeIfPresent(configId, forKey: .configId)
        try container.encodeIfPresent(contextTruncated, forKey: .contextTruncated)
        try container.encodeIfPresent(conversationId, forKey: .conversationId)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encodeIfPresent(description, forKey: .description)
        try container.encodeIfPresent(forkedFrom, forKey: .forkedFrom)
        try container.encode(id, forKey: .id)
        try container.encodeIfPresent(incognito, forKey: .incognito)
        try container.encodeIfPresent(instructions, forKey: .instructions)
        try container.encodeIfPresent(lastResponseAt, forKey: .lastResponseAt)
        try container.encodeIfPresent(llm, forKey: .llm)
        try container.encode(modality, forKey: .modality)
        try container.encodeIfPresent(mode, forKey: .mode)
        try container.encodeIfPresent(modelOverwrites, forKey: .modelOverwrites)
        try container.encodeIfPresent(projectId, forKey: .projectId)
        try container.encodeIfPresent(reviewNotes, forKey: .reviewNotes)
        try container.encodeIfPresent(reviewScore, forKey: .reviewScore)
        try container.encode(state, forKey: .state)
        try container.encodeIfPresent(sts, forKey: .sts)
        try container.encodeIfPresent(stt, forKey: .stt)
        try container.encodeIfPresent(subagent, forKey: .subagent)
        try container.encodeIfPresent(summary, forKey: .summary)
        try container.encodeIfPresent(text, forKey: .text)
        try container.encodeIfPresent(title, forKey: .title)
        try container.encodeIfPresent(tts, forKey: .tts)
        try container.encodeIfPresent(usage, forKey: .usage)
        try container.encode(userId, forKey: .userId)
        try container.encodeIfPresent(video, forKey: .video)
        try container.encodeIfPresent(voice, forKey: .voice)
    }

}

extension SessionPayload: Hashable {
    static func == (lhs: SessionPayload, rhs: SessionPayload) -> Bool {
        lhs.agent == rhs.agent &&
        lhs.agentId == rhs.agentId &&
        lhs.callId == rhs.callId &&
        lhs.callOutcome == rhs.callOutcome &&
        lhs.callType == rhs.callType &&
        lhs.closedAt == rhs.closedAt &&
        lhs.configId == rhs.configId &&
        lhs.contextTruncated == rhs.contextTruncated &&
        lhs.conversationId == rhs.conversationId &&
        lhs.createdAt == rhs.createdAt &&
        lhs.custom == rhs.custom &&
        lhs.description == rhs.description &&
        lhs.forkedFrom == rhs.forkedFrom &&
        lhs.id == rhs.id &&
        lhs.incognito == rhs.incognito &&
        lhs.instructions == rhs.instructions &&
        lhs.lastResponseAt == rhs.lastResponseAt &&
        lhs.llm == rhs.llm &&
        lhs.modality == rhs.modality &&
        lhs.mode == rhs.mode &&
        lhs.modelOverwrites == rhs.modelOverwrites &&
        lhs.projectId == rhs.projectId &&
        lhs.reviewNotes == rhs.reviewNotes &&
        lhs.reviewScore == rhs.reviewScore &&
        lhs.state == rhs.state &&
        lhs.sts == rhs.sts &&
        lhs.stt == rhs.stt &&
        lhs.subagent == rhs.subagent &&
        lhs.summary == rhs.summary &&
        lhs.text == rhs.text &&
        lhs.title == rhs.title &&
        lhs.tts == rhs.tts &&
        lhs.usage == rhs.usage &&
        lhs.userId == rhs.userId &&
        lhs.video == rhs.video &&
        lhs.voice == rhs.voice
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(agent)
        hasher.combine(agentId)
        hasher.combine(callId)
        hasher.combine(callOutcome)
        hasher.combine(callType)
        hasher.combine(closedAt)
        hasher.combine(configId)
        hasher.combine(contextTruncated)
        hasher.combine(conversationId)
        hasher.combine(createdAt)
        hasher.combine(custom)
        hasher.combine(description)
        hasher.combine(forkedFrom)
        hasher.combine(id)
        hasher.combine(incognito)
        hasher.combine(instructions)
        hasher.combine(lastResponseAt)
        hasher.combine(llm)
        hasher.combine(modality)
        hasher.combine(mode)
        hasher.combine(modelOverwrites)
        hasher.combine(projectId)
        hasher.combine(reviewNotes)
        hasher.combine(reviewScore)
        hasher.combine(state)
        hasher.combine(sts)
        hasher.combine(stt)
        hasher.combine(subagent)
        hasher.combine(summary)
        hasher.combine(text)
        hasher.combine(title)
        hasher.combine(tts)
        hasher.combine(usage)
        hasher.combine(userId)
        hasher.combine(video)
        hasher.combine(voice)
    }
}
