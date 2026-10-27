import Foundation
import StreamCore

final class CreateSessionRequest: Sendable, Codable, JSONEncodable {
    /// The name of an agent config to start from, as an alternative to config_id. It is what a caller actually knows the agent as: "docs" rather than an id they never chose. A name matching nothing is refused rather than silently starting an unconfigured agent, and naming both this and config_id is refused too, since there is no sensible answer when they disagree.
    let agent: String?
    /// Keys transcripts and statistics. Empty means the call id.
    let agentId: String?
    /// Murmur while a participant is still talking, the way a person does.
    let backchannel: Bool?
    /// An existing Stream call for the agent to join, by its type and id.
    let call: SessionCall?
    /// An agent config to start from. Everything else in this request overrides what the config says, so a caller can reuse a configuration and still change one thing about this call.
    let configId: String?
    /// The connection to use for each of the agent config's connector bindings chosen per session (connection.type session), by its alias. Each must be the verified caller's own connection to the binding's connector: an end user's, or the one a backend names with X-Stream-User-Id, never an anonymous caller's or a guest's. A binding with a fixed connection cannot be given one here, and an alias the config does not declare is refused. A session binding given none here uses the caller's own connection to its connector when exactly one of theirs is connected. Otherwise, with none connected or more than one, a required binding fails the session; an optional one is left out and reported with a connector_unavailable event (no_selection). A fork chooses the same connections again, against the config as it is then and the caller asking for the fork.
    let connectorBindings: [SessionConnectorBinding]?
    /// Older history was omitted from the model context.
    let contextTruncated: Bool?
    /// Anything the caller wants to remember about the session, handed back untouched and never read by the router. Sessions can be queried by these, which is what makes them worth writing.
    let custom: [String: RawJSON]?
    /// A longer note about the conversation, searched alongside the title.
    let description: String?
    /// What the agent says as it joins, before anyone speaks.
    let greeting: GreetingPayload?
    /// The conversation so far, for a backend that keeps its own: a thread in its own Slack app, say, that outlives any one session. Send it when a session closed and the thread goes on: open a new session with the thread's messages here, oldest first, then send the message to answer to the responses endpoint. The model is handed them before the first response, as a resumed conversation's history is. They are recorded nowhere, as turns, transcript or Chat messages, so add incognito to keep nothing at all. Up to 100 messages and 60000 characters of text, the most a session reads back of a conversation the router kept; more is refused rather than cut. Server-side only: a device sending it is refused with a 403, because an assistant message puts words in the agent's mouth.
    let history: [HistoryMessage]?
    /// The id to hold the session by, so a caller can know it before the session exists. It may be any string of up to 64 letters, digits, - and _, such as your own record's id, that nobody has used for a session before. A UUID is held lowercase. Omitted, the router generates a UUIDv7.
    let id: String?
    /// Hold the conversation and record nothing about it: no session row, no turns, no transcript, and no Stream Chat channel. The session still works exactly as any other while it is running; it simply cannot be found afterwards, which is the point. Forking one is refused, because there is nothing to fork from.
    let incognito: Bool?
    /// What the agent is told to be for this session alone, in place of its config's instructions: a role given at call time, such as the brief of a customer somebody practises on, without a stored config per brief. Empty or left out, the config's instructions stand. There is no limit but the request body's, as a config's instructions have none. They are held only by the running session: a session carried on in writing after it ended in another process, a fork of a session no longer running and a fork naming a config run on the config's instructions again. Server-side only: a device sending them is refused with a 403, because what the agent is told decides who it is, and a device could make it impersonate a different agent.
    let instructions: String?
    /// Business-specific words the transcriber would otherwise get wrong. Up to 100 terms, and providers that cannot be told about vocabulary ignore them.
    let keyterms: [String]?
    /// Language hints, which narrow the candidates in every modality.
    let languages: [String]?
    /// A provider/model or a capability shortcut. Omit it and the config decides, or llm-fast when there is no config. These carry no schema default on purpose: a generated client that filled one in would send it, and a caller naming a config would silently lose the model it configured.
    let llm: String?
    let maxTokens: Int?
    /// Who the session's memories are about. Without a user id nothing is recalled or stored, which is the case for a call with nobody identified on it.
    let memory: SessionMemory?
    /// How sure the transcriber must be before the agent answers rather than checks what was meant.
    let minConfidence: Double?
    /// What to change about the models for one session, over whatever its agent config decided.
    /// It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides the config, and a caller reading a session back wants to see what they changed in one place rather than diffed against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because a caller able to rewrite those could make a session impersonate a different agent. A backend sets one session's instructions with the instructions of CreateSessionRequest instead, which a device may not send.
    let modelOverwrites: ModelOverwritesPayload?
    /// The agent placed this call, so let recordings finish and answer their menus.
    let navigating: Bool?
    /// The number the session acts from, which is what turns transferring on.
    let phone: SessionPhone?
    /// What the conversation belongs to. Also recorded as the "project" cost tag, so spend breaks down by project without the caller labelling it twice. A tag spelled out in tags wins.
    let projectId: String?
    /// How a conversation is handed back to the model when a session carries it on.
    let resumeHistory: ResumeHistory?
    /// The app's own page a login or step-up the chat asks for goes back to, as createAuthorization's return_url: the launch page takes the handoff only from a page of its origin, and the browser comes back to it with connection_id and status added to its query. The same rules as createAuthorization's return_url apply, and a URL they refuse is answered with a 400 invalid_request naming the rule (an error frame on the session socket). Absent or empty, those consents go back to the dashboard. It is kept with the session, so a chat carried on after it ended and a fork go back to it too. An end user's device may send it: a consent the chat asks for is only ever for the caller's own connection, shown to the caller alone, and bound to the first browser that hands it off, so the URL only decides where the caller's own consent lands.
    let returnUrl: String?
    /// Omit it and the config decides, or search-fast when there is no config.
    let search: String?
    /// Start voice as the session opens, as startSessionVoice does: the agent joins the call named in call, else agent:<session id>, which joining creates, and returns once it is there. Left out, the conversation is held in writing until voice is started.
    let startVoice: Bool?
    /// A speech-to-speech target. Naming one makes the session native once voice is started: the model hears and speaks for itself, so no transcriber, conversation model or voice is opened. Omit it and the config decides.
    let sts: String?
    /// Omit it and the config decides, or en-low-latency when there is no config.
    let stt: String?
    /// Cost labels, carried onto every request the session makes. Merged with the agent config's tags; where both name a key, this one wins.
    let tags: [String: String]?
    /// The end user's IANA timezone, such as Europe/Amsterdam, which a device knows and the router does not. It sets the zone the agent reads dates in for this session and, for a user the caller may name, is remembered as theirs for the scheduled tasks and sessions that follow. An unknown zone is refused with a 400 invalid_timezone.
    let timezone: String?
    /// What to call the conversation, for a list a person reads, until the router names a persistent one for what was said. Never shown to the model: what a conversation is called is a label on it rather than part of it.
    let title: String?
    /// How long the model waits for a tool result. Zero is the default.
    let toolTimeoutMs: Int?
    let tools: [SessionTool]?
    /// Omit it and the config decides, or en-low-latency when there is no config.
    let tts: String?
    /// Who the agent joins the call as.
    let userId: String?
    let userName: String?
    let video: SessionVideo?
    /// The voice to speak in: a provider's own voice id, or one of your voices by id or by the name you gave it. Prefix it with custom: to mean only the latter: without the prefix a name that is not one of yours is passed through to the provider's library, and with it a name that is not one of yours is refused. A native session (sts) takes the vendor's own voice name instead, such as marin at OpenAI.
    let voice: String?

    init(agent: String? = nil, agentId: String? = nil, backchannel: Bool? = nil, call: SessionCall? = nil, configId: String? = nil, connectorBindings: [SessionConnectorBinding]? = nil, contextTruncated: Bool? = nil, custom: [String: RawJSON]? = nil, description: String? = nil, greeting: GreetingPayload? = nil, history: [HistoryMessage]? = nil, id: String? = nil, incognito: Bool? = nil, instructions: String? = nil, keyterms: [String]? = nil, languages: [String]? = nil, llm: String? = nil, maxTokens: Int? = nil, memory: SessionMemory? = nil, minConfidence: Double? = nil, modelOverwrites: ModelOverwritesPayload? = nil, navigating: Bool? = nil, phone: SessionPhone? = nil, projectId: String? = nil, resumeHistory: ResumeHistory? = nil, returnUrl: String? = nil, search: String? = nil, startVoice: Bool? = nil, sts: String? = nil, stt: String? = nil, tags: [String: String]? = nil, timezone: String? = nil, title: String? = nil, toolTimeoutMs: Int? = nil, tools: [SessionTool]? = nil, tts: String? = nil, userId: String? = "vision-agent", userName: String? = "Vision Agent", video: SessionVideo? = nil, voice: String? = nil) {
        self.agent = agent
        self.agentId = agentId
        self.backchannel = backchannel
        self.call = call
        self.configId = configId
        self.connectorBindings = connectorBindings
        self.contextTruncated = contextTruncated
        self.custom = custom
        self.description = description
        self.greeting = greeting
        self.history = history
        self.id = id
        self.incognito = incognito
        self.instructions = instructions
        self.keyterms = keyterms
        self.languages = languages
        self.llm = llm
        self.maxTokens = maxTokens
        self.memory = memory
        self.minConfidence = minConfidence
        self.modelOverwrites = modelOverwrites
        self.navigating = navigating
        self.phone = phone
        self.projectId = projectId
        self.resumeHistory = resumeHistory
        self.returnUrl = returnUrl
        self.search = search
        self.startVoice = startVoice
        self.sts = sts
        self.stt = stt
        self.tags = tags
        self.timezone = timezone
        self.title = title
        self.toolTimeoutMs = toolTimeoutMs
        self.tools = tools
        self.tts = tts
        self.userId = userId
        self.userName = userName
        self.video = video
        self.voice = voice
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.agent = try container.decodeIfPresent(String.self, forKey: .agent)
        self.agentId = try container.decodeIfPresent(String.self, forKey: .agentId)
        self.backchannel = try container.decodeIfPresent(Bool.self, forKey: .backchannel)
        self.call = try container.decodeIfPresent(SessionCall.self, forKey: .call)
        self.configId = try container.decodeIfPresent(String.self, forKey: .configId)
        self.connectorBindings = try container.decodeIfPresent([SessionConnectorBinding].self, forKey: .connectorBindings)
        self.contextTruncated = try container.decodeIfPresent(Bool.self, forKey: .contextTruncated)
        self.custom = try container.decodeIfPresent([String: RawJSON].self, forKey: .custom)
        self.description = try container.decodeIfPresent(String.self, forKey: .description)
        self.greeting = try container.decodeIfPresent(GreetingPayload.self, forKey: .greeting)
        self.history = try container.decodeIfPresent([HistoryMessage].self, forKey: .history)
        self.id = try container.decodeIfPresent(String.self, forKey: .id)
        self.incognito = try container.decodeIfPresent(Bool.self, forKey: .incognito)
        self.instructions = try container.decodeIfPresent(String.self, forKey: .instructions)
        self.keyterms = try container.decodeIfPresent([String].self, forKey: .keyterms)
        self.languages = try container.decodeIfPresent([String].self, forKey: .languages)
        self.llm = try container.decodeIfPresent(String.self, forKey: .llm)
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        self.memory = try container.decodeIfPresent(SessionMemory.self, forKey: .memory)
        self.minConfidence = try container.decodeIfPresent(Double.self, forKey: .minConfidence)
        self.modelOverwrites = try container.decodeIfPresent(ModelOverwritesPayload.self, forKey: .modelOverwrites)
        self.navigating = try container.decodeIfPresent(Bool.self, forKey: .navigating)
        self.phone = try container.decodeIfPresent(SessionPhone.self, forKey: .phone)
        self.projectId = try container.decodeIfPresent(String.self, forKey: .projectId)
        self.resumeHistory = try container.decodeIfPresent(ResumeHistory.self, forKey: .resumeHistory)
        self.returnUrl = try container.decodeIfPresent(String.self, forKey: .returnUrl)
        self.search = try container.decodeIfPresent(String.self, forKey: .search)
        self.startVoice = try container.decodeIfPresent(Bool.self, forKey: .startVoice)
        self.sts = try container.decodeIfPresent(String.self, forKey: .sts)
        self.stt = try container.decodeIfPresent(String.self, forKey: .stt)
        self.tags = try container.decodeIfPresent([String: String].self, forKey: .tags)
        self.timezone = try container.decodeIfPresent(String.self, forKey: .timezone)
        self.title = try container.decodeIfPresent(String.self, forKey: .title)
        self.toolTimeoutMs = try container.decodeIfPresent(Int.self, forKey: .toolTimeoutMs)
        self.tools = try container.decodeIfPresent([SessionTool].self, forKey: .tools)
        self.tts = try container.decodeIfPresent(String.self, forKey: .tts)
        self.userId = try container.decodeIfPresent(String.self, forKey: .userId)
        self.userName = try container.decodeIfPresent(String.self, forKey: .userName)
        self.video = try container.decodeIfPresent(SessionVideo.self, forKey: .video)
        self.voice = try container.decodeIfPresent(String.self, forKey: .voice)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(agent, forKey: .agent)
        try container.encodeIfPresent(agentId, forKey: .agentId)
        try container.encodeIfPresent(backchannel, forKey: .backchannel)
        try container.encodeIfPresent(call, forKey: .call)
        try container.encodeIfPresent(configId, forKey: .configId)
        try container.encodeIfPresent(connectorBindings, forKey: .connectorBindings)
        try container.encodeIfPresent(contextTruncated, forKey: .contextTruncated)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encodeIfPresent(description, forKey: .description)
        try container.encodeIfPresent(greeting, forKey: .greeting)
        try container.encodeIfPresent(history, forKey: .history)
        try container.encodeIfPresent(id, forKey: .id)
        try container.encodeIfPresent(incognito, forKey: .incognito)
        try container.encodeIfPresent(instructions, forKey: .instructions)
        try container.encodeIfPresent(keyterms, forKey: .keyterms)
        try container.encodeIfPresent(languages, forKey: .languages)
        try container.encodeIfPresent(llm, forKey: .llm)
        try container.encodeIfPresent(maxTokens, forKey: .maxTokens)
        try container.encodeIfPresent(memory, forKey: .memory)
        try container.encodeIfPresent(minConfidence, forKey: .minConfidence)
        try container.encodeIfPresent(modelOverwrites, forKey: .modelOverwrites)
        try container.encodeIfPresent(navigating, forKey: .navigating)
        try container.encodeIfPresent(phone, forKey: .phone)
        try container.encodeIfPresent(projectId, forKey: .projectId)
        try container.encodeIfPresent(resumeHistory, forKey: .resumeHistory)
        try container.encodeIfPresent(returnUrl, forKey: .returnUrl)
        try container.encodeIfPresent(search, forKey: .search)
        try container.encodeIfPresent(startVoice, forKey: .startVoice)
        try container.encodeIfPresent(sts, forKey: .sts)
        try container.encodeIfPresent(stt, forKey: .stt)
        try container.encodeIfPresent(tags, forKey: .tags)
        try container.encodeIfPresent(timezone, forKey: .timezone)
        try container.encodeIfPresent(title, forKey: .title)
        try container.encodeIfPresent(toolTimeoutMs, forKey: .toolTimeoutMs)
        try container.encodeIfPresent(tools, forKey: .tools)
        try container.encodeIfPresent(tts, forKey: .tts)
        try container.encodeIfPresent(userId, forKey: .userId)
        try container.encodeIfPresent(userName, forKey: .userName)
        try container.encodeIfPresent(video, forKey: .video)
        try container.encodeIfPresent(voice, forKey: .voice)
    }

}

extension CreateSessionRequest: Hashable {
    static func == (lhs: CreateSessionRequest, rhs: CreateSessionRequest) -> Bool {
        lhs.agent == rhs.agent &&
        lhs.agentId == rhs.agentId &&
        lhs.backchannel == rhs.backchannel &&
        lhs.call == rhs.call &&
        lhs.configId == rhs.configId &&
        lhs.connectorBindings == rhs.connectorBindings &&
        lhs.contextTruncated == rhs.contextTruncated &&
        lhs.custom == rhs.custom &&
        lhs.description == rhs.description &&
        lhs.greeting == rhs.greeting &&
        lhs.history == rhs.history &&
        lhs.id == rhs.id &&
        lhs.incognito == rhs.incognito &&
        lhs.instructions == rhs.instructions &&
        lhs.keyterms == rhs.keyterms &&
        lhs.languages == rhs.languages &&
        lhs.llm == rhs.llm &&
        lhs.maxTokens == rhs.maxTokens &&
        lhs.memory == rhs.memory &&
        lhs.minConfidence == rhs.minConfidence &&
        lhs.modelOverwrites == rhs.modelOverwrites &&
        lhs.navigating == rhs.navigating &&
        lhs.phone == rhs.phone &&
        lhs.projectId == rhs.projectId &&
        lhs.resumeHistory == rhs.resumeHistory &&
        lhs.returnUrl == rhs.returnUrl &&
        lhs.search == rhs.search &&
        lhs.startVoice == rhs.startVoice &&
        lhs.sts == rhs.sts &&
        lhs.stt == rhs.stt &&
        lhs.tags == rhs.tags &&
        lhs.timezone == rhs.timezone &&
        lhs.title == rhs.title &&
        lhs.toolTimeoutMs == rhs.toolTimeoutMs &&
        lhs.tools == rhs.tools &&
        lhs.tts == rhs.tts &&
        lhs.userId == rhs.userId &&
        lhs.userName == rhs.userName &&
        lhs.video == rhs.video &&
        lhs.voice == rhs.voice
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(agent)
        hasher.combine(agentId)
        hasher.combine(backchannel)
        hasher.combine(call)
        hasher.combine(configId)
        hasher.combine(connectorBindings)
        hasher.combine(contextTruncated)
        hasher.combine(custom)
        hasher.combine(description)
        hasher.combine(greeting)
        hasher.combine(history)
        hasher.combine(id)
        hasher.combine(incognito)
        hasher.combine(instructions)
        hasher.combine(keyterms)
        hasher.combine(languages)
        hasher.combine(llm)
        hasher.combine(maxTokens)
        hasher.combine(memory)
        hasher.combine(minConfidence)
        hasher.combine(modelOverwrites)
        hasher.combine(navigating)
        hasher.combine(phone)
        hasher.combine(projectId)
        hasher.combine(resumeHistory)
        hasher.combine(returnUrl)
        hasher.combine(search)
        hasher.combine(startVoice)
        hasher.combine(sts)
        hasher.combine(stt)
        hasher.combine(tags)
        hasher.combine(timezone)
        hasher.combine(title)
        hasher.combine(toolTimeoutMs)
        hasher.combine(tools)
        hasher.combine(tts)
        hasher.combine(userId)
        hasher.combine(userName)
        hasher.combine(video)
        hasher.combine(voice)
    }
}
