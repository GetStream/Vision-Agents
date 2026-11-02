import Foundation
// `User` and `RawJSON` cross the public API, so an app importing this has them, as Stream Chat
// and Stream Video hand over theirs.
@_exported import StreamCore

/// What a session should be, for the cases the shorthands do not cover.
///
/// Everything is optional because everything has an answer already: a named config decides
/// what it does not say, and the router decides what the config does not. Setting a field here
/// overrides both, for this session only.
public struct SessionOptions: Sendable {
    /// The id to hold the session by, so a caller knows it before the session exists: up to
    /// 64 letters, digits, `-` and `_` no session has had. Nil lets the router choose one.
    public var id: String?
    /// The agent to talk to, by the name its config was synced under. The router resolves
    /// it, and refuses a name that matches nothing rather than starting an agent with no
    /// config.
    public var agent: String?
    /// An agent config to start from, by id, for a caller that holds one instead of a name.
    public var configID: String?
    /// Has the agent join the session's own call, `agent:<session id>`, as soon as it opens.
    /// False holds the conversation in writing until `AgentSession.startVoice` starts it.
    public var startVoice = false
    /// Writes nothing down: no transcript, no memory.
    public var incognito: Bool?
    public var title: String?
    public var description: String?
    public var projectID: String?
    /// Your own labels, handed back untouched. `SessionQuery.custom` finds the session by them.
    public var custom: [String: RawJSON] = [:]
    /// What the agent opens the call with.
    public var greeting: Greeting?
    public var llm: String?
    public var stt: String?
    public var tts: String?
    /// What to change about the models for this session alone: thinking, temperature, a cap on
    /// every reply and the rest.
    public var modelOverwrites: ModelOverwrites?
    /// Has the agent remember the signed-in user: what it learned about them before is recalled
    /// here, and what it learns here is kept for their next session. Nil remembers nothing, and
    /// an incognito session never does.
    public var memory: Memory?
    /// The user's IANA zone, such as `Europe/Amsterdam`, which the device knows and the router
    /// does not. The agent reads dates in it, and it is remembered as the user's for the
    /// scheduled tasks that follow.
    ///
    /// It starts as this device's zone, `TimeZone.current` as it is when the options are made.
    /// A device zone with no IANA name, such as a bare offset like `GMT+0100`, starts it as nil
    /// instead, since the router refuses a zone it does not know and the session with it. Set it
    /// to name another zone, which is sent as it is, or to nil to send none.
    public var timezone: String?
    /// The voice the agent speaks in: a provider's own voice id, or one of your voices by id or by
    /// the name you gave it. Prefix it with `custom:` to mean only yours.
    public var voice: String?
    /// Functions of yours the agent may call, answered on this device.
    public var tools: [AgentTool] = []
    /// Cost labels, carried onto every request the session makes.
    public var tags: [String: String] = [:]
    /// Where a sign-in the conversation asks for goes back to. Name your app's own scheme in
    /// reverse domain form, such as `com.example.app:/connected`, and the router hands the web
    /// authentication sheet back to the app. It is kept with the session, so a conversation
    /// carried on later goes back there too. Nil sends the sign-in back to the dashboard, which a
    /// phone cannot open.
    public var returnURL: URL?

    public init(agent: String? = nil) {
        self.init(agent: agent, deviceZone: TimeZone.current.identifier)
    }

    /// Options whose zone starts as `deviceZone` rather than this device's, for a test.
    init(agent: String?, deviceZone: String) {
        self.agent = agent
        timezone = Self.ianaZone(deviceZone)
    }

    /// `identifier` when it reads as an IANA zone, a region and a place such as
    /// `Europe/Amsterdam` or UTC under either of its names, and nil when it does not. A device can
    /// name its zone by an offset instead, which is no zone the router knows.
    static func ianaZone(_ identifier: String) -> String? {
        identifier.contains("/") || identifier == "UTC" || identifier == "GMT" ? identifier : nil
    }

    /// What a session remembers about the user `setUser` named. It is always theirs: a device
    /// recalls nobody else's.
    public struct Memory: Sendable, Hashable {
        /// Your own labels, which narrow what is recalled. They cannot widen it past the user.
        public var filter: [String: String]
        /// Separates two of your apps sharing one memory account.
        public var appID: String?

        public init(filter: [String: String] = [:], appID: String? = nil) {
            self.filter = filter
            self.appID = appID
        }
    }
}

/// The router, as a phone sees it.
///
/// A few lines get a conversation going:
///
///     let agents = VisionAgents(apiKey: "your_api_key")
///     agents.setUser(User(id: "jlahey")) { try await yourBackend.agentToken() }
///     let session = try await agents.agent("myagent").sessions.create()
///     let turn = try await session.responses.create("Where is order 1042?")
///
/// Opening a conversation, reading back its turns, going back to one of them, branching off
/// and ending it is the whole of what is here, because it is the whole of what the router
/// lets a device do. What an agent is configured as and a token to join a call with are
/// server-side only: they belong to a backend, which has the Go or the Python SDK, and which
/// hands down what the app needs.
public struct VisionAgents: Sendable {
    public let backend: Backend

    public init(apiKey: String, url: URL = Backend.defaultURL, urlSession: URLSession = .shared) {
        backend = Backend(apiKey: apiKey, url: url, urlSession: urlSession)
    }

    /// A router running locally with nothing in front of it. `apiKey` is only for Stream Chat
    /// and Video, which connect with it as the user `setUser` names.
    public init(url: URL, customerID: String, apiKey: String = "", urlSession: URLSession = .shared) {
        backend = Backend(url: url, customerID: customerID, apiKey: apiKey, urlSession: urlSession)
    }

    public init(backend: Backend) {
        self.backend = backend
    }

    /// Every conversation this caller has, whichever agent holds it.
    public var sessions: Sessions { Sessions(agents: self, agent: nil) }

    /// The reminders and later work agents hold for this user.
    public var scheduledTasks: ScheduledTasks { ScheduledTasks(backend: backend) }

    /// Says who this device is acting for, and how to prove it. See `Backend.setUser`.
    public func setUser(_ user: User, token: @escaping TokenProvider) {
        backend.setUser(user, token: token)
    }

    /// Says who this device is acting for, with a token that will not be refreshed.
    public func setUser(_ user: User, token: String) {
        backend.setUser(user, token: token)
    }

    /// Acts as this guest from here on.
    public func setUser(_ guest: Guest) {
        backend.setUser(guest)
    }

    /// Forgets the user, which is what signing out is.
    public func clearUser() {
        backend.clearUser()
    }

    /// Gets or creates a guest, so somebody can talk to an agent before they sign up.
    ///
    /// Remembered in `store` where there is one, so the next launch is the same guest and not
    /// a second one with an empty history. Pass it to `setUser` to act as them. With no user
    /// set it is asked for anonymously, which a router must admit; one that does not is
    /// answered with a guest your backend mints instead. Moving a
    /// guest's conversations onto the account they turn out to be is server-side only: it is
    /// the backend that just authenticated them that knows which guest they were.
    public func guestUser(
        _ options: GuestOptions = GuestOptions(), store: (any GuestStore)? = nil
    ) async throws -> Guest {
        try await backend.guest(options, store: store)
    }

    /// Forgets the remembered guest without minting another, which is what signing out is.
    public func forgetGuest(store: any GuestStore) throws {
        try store.clear()
        backend.clearUser()
    }

    /// Disconnects the Stream Chat and Video clients built for the app's sessions. Clients the
    /// app handed over with `use` are its own and stay connected; closing a session
    /// disconnects nothing.
    public func disconnect() async {
        await backend.disconnectStream()
    }

    /// One agent, by the name its config was synced under.
    public func agent(_ name: String) -> Agent {
        Agent(agents: self, name: name)
    }

    /// Looking something up under a stored router config, which is what decides the model.
    /// Without one, every call says what it wants for itself.
    public func router(config: String = "", tags: [String: String] = [:]) -> Router {
        Router(backend: backend, config: config, tags: tags)
    }

    /// Holds a conversation in writing: no call is joined, nothing is transcribed or spoken.
    ///
    /// The replies still come through the model with the same instructions, skills and
    /// knowledge a call would have had, and arrive as deltas on the session's socket.
    public func chat(agent: String? = nil, tools: [AgentTool] = []) async throws -> AgentSession {
        var options = SessionOptions(agent: agent)
        options.tools = tools
        return try await chat(options)
    }

    /// Holds a conversation in writing, configured in full.
    public func chat(_ options: SessionOptions) async throws -> AgentSession {
        var options = options
        options.startVoice = false
        let session = try await createSession(options)
        return await AgentSession(backend: backend, session: session, tools: options.tools)
    }

    /// Puts an agent on the session's own call, `agent:<session id>`, and follows it.
    ///
    /// The agent joins as soon as this returns. Joining the same call from this device is what
    /// `VoiceSession` is for; this only starts the agent and gives you the state layer.
    public func voice(agent: String? = nil, tools: [AgentTool] = []) async throws -> AgentSession {
        var options = SessionOptions(agent: agent)
        options.tools = tools
        return try await voice(options)
    }

    /// Puts an agent on the session's own call, configured in full.
    public func voice(_ options: SessionOptions) async throws -> AgentSession {
        var options = options
        options.startVoice = true
        let session = try await createSession(options)
        return await AgentSession(backend: backend, session: session, tools: options.tools)
    }

    /// Starts a session without following it, for a caller building its own state layer.
    public func createSession(_ options: SessionOptions) async throws -> Session {
        let body = CreateSessionRequest(
            agent: options.agent.flatMap { $0.isEmpty ? nil : $0 },
            configId: options.configID.flatMap { $0.isEmpty ? nil : $0 },
            custom: options.custom.isEmpty ? nil : options.custom,
            description: options.description,
            greeting: options.greeting?.schema,
            id: options.id.flatMap { $0.isEmpty ? nil : $0 },
            incognito: options.incognito,
            llm: options.llm,
            // The router remembers only for a user id, and a device only ever means its own user.
            memory: options.memory.map {
                SessionMemory(
                    appId: $0.appID, filter: $0.filter.isEmpty ? nil : $0.filter, userId: backend.user?.id)
            },
            modelOverwrites: options.modelOverwrites?.schema,
            projectId: options.projectID,
            returnUrl: options.returnURL?.absoluteString,
            startVoice: options.startVoice ? true : nil,
            stt: options.stt,
            tags: options.tags.isEmpty ? nil : options.tags,
            timezone: options.timezone,
            title: options.title,
            tools: options.tools.map {
                SessionTool(
                    approval: $0.approval.map {
                        .init(
                            allowTitle: $0.allowTitle, declineTitle: $0.declineTitle,
                            message: $0.message, reasonArgument: $0.reasonArgument, title: $0.title)
                    },
                    description: $0.description,
                    displayTitle: $0.displayTitle,
                    executor: $0.executor.map { .init(rawValue: $0.rawValue) },
                    name: $0.name,
                    // A schema that is not an object describes no arguments the router can read.
                    parameters: $0.parameters.map { $0.dictionaryValue ?? [:] })
            },
            tts: options.tts,
            voice: options.voice)

        return Session(try await call { try await $0.createSession(createSessionRequest: body) })
    }

    /// Follows a session this caller already has open, without creating one.
    ///
    /// Use this when the app opened a session and is coming back to it — after a relaunch,
    /// or on another screen — which is how a conversation is carried on. A session opened by
    /// somebody else is not found, because reading one is reading a conversation.
    public func attach(sessionID: String, tools: [AgentTool] = []) async throws -> AgentSession {
        let session = try await sessions.get(sessionID)
        return await AgentSession(backend: backend, session: session, tools: tools)
    }

    /// Stops a session, which is how the agent leaves. What it recorded and remembered is
    /// kept; `sessions.delete` takes it away.
    public func close(sessionID: String) async throws {
        try await call { try await $0.stopSession(id: sessionID) }
    }

    /// One page of a session's turns as the router wrote them down, oldest first.
    ///
    /// A session that records nothing has none, and one rewound has none after the response
    /// it went back to.
    public func responses(
        sessionID: String, limit: Int? = nil, cursor: String? = nil
    ) async throws -> Page<Response> {
        try await Responses(backend: backend, sessionID: sessionID).list(limit: limit, cursor: cursor)
    }

    /// Goes back to a response and carries on from there, as though nothing after it was said.
    ///
    /// The model forgets the later turns and they drop out of `responses`. A transcript an
    /// `AgentSession` is showing still has them, so read it back from `responses` after this.
    /// A conversation kept in Stream Chat cannot be rewound, because the channel would still
    /// hold the later turns: fork it at the response instead.
    public func rewind(sessionID: String, to responseID: String) async throws {
        try await call {
            try await $0.rewindSession(id: sessionID, rewindSessionRequest: .init(responseId: responseID))
        }
    }

    /// Continues a conversation as a new session held in writing, leaving the parent as it was.
    ///
    /// Follow the fork the way any session is followed, with `attach(sessionID:)`.
    public func fork(sessionID: String, _ options: ForkOptions = ForkOptions()) async throws -> Session {
        let body = ForkSessionRequest(
            agent: options.agent.flatMap { $0.isEmpty ? nil : $0 },
            configId: options.configID.flatMap { $0.isEmpty ? nil : $0 },
            custom: options.custom,
            description: options.description,
            incognito: options.incognito,
            messages: options.withoutHistory ? false : nil,
            modelOverwrites: options.modelOverwrites?.schema,
            projectId: options.projectID,
            responseId: options.responseID.flatMap { $0.isEmpty ? nil : $0 },
            title: options.title)

        return Session(try await call { try await $0.forkSession(id: sessionID, forkSessionRequest: body) })
    }

    private func call<T>(_ body: (DefaultAPI) async throws -> T) async throws -> T {
        try await backend.call(body)
    }
}

/// One agent, and the conversations held with it.
public struct Agent: Sendable {
    /// The name the agent's config was synced under.
    public let name: String

    /// This agent's conversations: opening one names it, and listing is narrowed to it.
    public let sessions: Sessions

    init(agents: VisionAgents, name: String) {
        self.name = name
        sessions = Sessions(agents: agents, agent: name)
    }
}

/// Conversations: opening one, and finding the ones there were.
///
/// From `Agent.sessions` every call is about that agent: opening names it, and listing is
/// narrowed to it.
public struct Sessions: Sendable {
    private let agents: VisionAgents
    private let agent: String?

    init(agents: VisionAgents, agent: String?) {
        self.agents = agents
        self.agent = agent
    }

    /// Opens a conversation held in writing: no call is joined, nothing is transcribed or
    /// spoken.
    ///
    /// Ask it things with `responses`. Call `start()` on it to watch the reply arrive and to
    /// answer the tools in `options`, which run on this device.
    public func create(_ options: SessionOptions = SessionOptions()) async throws -> AgentSession {
        var options = options
        if options.agent == nil { options.agent = agent }
        return try await agents.chat(options)
    }

    /// One page of this caller's conversations, most recently active first, the ones that
    /// ended included. Pass the page's `nextCursor` as `query.cursor` for the next one.
    ///
    /// Only ever this caller's own. The router owns a session by whoever opened it, so a
    /// device is never told about anybody else's conversation.
    public func query(_ query: SessionQuery = SessionQuery()) async throws -> Page<Session> {
        try await sessions(matching: nil, query)
    }

    /// Finds conversations by their title, description, project and agent name, best match
    /// first. What was said is not searched. It pages the way `query` does.
    public func search(_ text: String, _ query: SessionQuery = SessionQuery()) async throws -> Page<Session> {
        try await sessions(matching: text, query)
    }

    /// One conversation, whether or not it is still being held.
    public func get(_ id: String) async throws -> Session {
        Session(try await agents.backend.call { try await $0.getSession(id: id) })
    }

    /// Renames or relabels a conversation, running or ended. Nil leaves a field as it is, and
    /// `custom` replaces the labels whole.
    public func update(
        _ id: String, title: String? = nil, description: String? = nil,
        custom: [String: RawJSON]? = nil
    ) async throws -> Session {
        let body = UpdateSessionRequest(custom: custom, description: description, title: title)
        return Session(
            try await agents.backend.call { try await $0.updateSession(id: id, updateSessionRequest: body) })
    }

    /// Has the agent join a session's call, `agent:<session id>`, and carry the conversation on
    /// there. Starting voice that is already on does nothing.
    public func startVoice(_ id: String) async throws -> Session {
        Session(
            try await agents.backend.call {
                try await $0.startSessionVoice(id: id, startSessionVoiceRequest: .init())
            })
    }

    /// Takes the agent off a session's call and carries the conversation on in writing.
    /// Stopping voice that is off does nothing.
    public func stopVoice(_ id: String) async throws -> Session {
        Session(try await agents.backend.call { try await $0.stopSessionVoice(id: id) })
    }

    /// Deletes a conversation, running or ended: it is stopped, and its turns and what it
    /// remembered are deleted with it.
    public func delete(_ id: String) async throws {
        try await agents.backend.call { try await $0.deleteSession(id: id) }
    }

    /// A session's turns, for one read back from `query` rather than held.
    public func responses(_ sessionID: String) -> Responses {
        Responses(backend: agents.backend, sessionID: sessionID)
    }

    private func sessions(matching text: String?, _ query: SessionQuery) async throws -> Page<Session> {
        let body = SessionQueryPayload(
            cursor: query.cursor,
            filter: .init(
                agent: equals(agent),
                agentId: equals(query.agentID),
                createdAt: query.createdSince == nil && query.createdBefore == nil
                    ? nil : .init(gte: query.createdSince, lt: query.createdBefore),
                custom: query.custom.isEmpty ? nil : query.custom,
                modality: equals(query.modality?.rawValue),
                projectId: equals(query.projectID),
                state: equals(query.state?.rawValue),
                text: text.map { .init(q: $0) }),
            limit: query.limit,
            sort: query.sort.map { [.init(field: .init(rawValue: $0.rawValue))] })
        let page = try await agents.backend.call { try await $0.querySessions(sessionQueryPayload: body) }
        return Page(
            items: page.items.map(Session.init), hasMore: page.hasMore, nextCursor: page.nextCursor)
    }
}

/// A filter field matching `value` exactly, or nil to leave the field out.
private func equals(_ value: String?) -> String? {
    value.flatMap { $0.isEmpty ? nil : $0 }
}

extension Backend {
    /// Runs one request, reporting a transport failure as one and leaving cancellation alone.
    func call<T>(_ body: (DefaultAPI) async throws -> T) async throws -> T {
        do {
            return try await body(client())
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AgentsError {
            throw error
        } catch let error as DecodingError {
            // A response arrived, so this is a success whose body did not read.
            throw AgentsError.unreadable("\(error)")
        } catch {
            throw AgentsError.transport(error)
        }
    }
}
