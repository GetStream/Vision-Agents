import Foundation
import OpenAPIRuntime

/// What a session should be, for the cases the shorthands do not cover.
///
/// Everything is optional because everything has an answer already: a named config decides
/// what it does not say, and the router decides what the config does not. Setting a field here
/// overrides both, for this session only.
public struct SessionOptions: Sendable {
    /// The id to hold the session by, so a caller knows it before the session exists. It must
    /// be a UUID no session has had; nil lets the router choose one.
    public var id: String?
    /// The agent to talk to, by the name its config was synced under. The router resolves
    /// it, and refuses a name that matches nothing rather than starting an agent with no
    /// config.
    public var agent: String?
    /// An agent config to start from, by id, for a caller that holds one instead of a name.
    public var configID: String?
    /// Carries on this conversation rather than starting one.
    public var conversationID: String?
    /// Writes nothing down: no transcript, no memory.
    public var incognito: Bool?
    public var title: String?
    public var description: String?
    public var projectID: String?
    /// The system prompt.
    public var instructions: String?
    /// Said on joining without going through the model.
    public var greeting: String?
    public var llm: String?
    public var stt: String?
    public var tts: String?
    /// A provider-specific voice id.
    public var voice: String?
    /// Functions of yours the agent may call, answered on this device.
    public var tools: [AgentTool] = []
    /// Cost labels, carried onto every request the session makes.
    public var tags: [String: String] = [:]

    public init(agent: String? = nil) {
        self.agent = agent
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

    /// Says who this device is acting for, and how to prove it. See `Backend.setUser`.
    public func setUser(_ user: User, token: @escaping TokenProvider) {
        backend.setUser(user, token: token)
    }

    /// Says who this device is acting for, with a token that will not be refreshed.
    public func setUser(_ user: User, token: String) {
        backend.setUser(user, token: token)
    }

    /// Forgets the user, which is what signing out is.
    public func clearUser() {
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
        let session = try await createSession(options, callID: nil)
        return await AgentSession(backend: backend, session: session, tools: options.tools)
    }

    /// Puts an agent on a call and follows it.
    ///
    /// The agent joins as soon as this returns. Joining the same call from this device is what
    /// the RTC package is for; this only starts the agent and gives you the state layer.
    public func voice(
        callID: String,
        agent: String? = nil,
        tools: [AgentTool] = []
    ) async throws -> AgentSession {
        var options = SessionOptions(agent: agent)
        options.tools = tools
        let session = try await createSession(options, callID: callID)
        return await AgentSession(backend: backend, session: session, tools: options.tools)
    }

    /// Puts an agent on a call, configured in full.
    public func voice(callID: String, options: SessionOptions) async throws -> AgentSession {
        let session = try await createSession(options, callID: callID)
        return await AgentSession(backend: backend, session: session, tools: options.tools)
    }

    /// Starts a session without following it, for a caller building its own state layer.
    public func createSession(_ options: SessionOptions, callID: String?) async throws -> Session {
        let body = Components.Schemas.CreateSessionRequest(
            agent: options.agent.flatMap { $0.isEmpty ? nil : $0 },
            callId: callID,
            configId: options.configID.flatMap { $0.isEmpty ? nil : $0 },
            conversationId: options.conversationID,
            description: options.description,
            greeting: options.greeting.map { .init(text: $0) },
            id: options.id.flatMap { $0.isEmpty ? nil : $0 },
            incognito: options.incognito,
            instructions: options.instructions,
            llm: options.llm,
            projectId: options.projectID,
            stt: options.stt,
            tags: options.tags.isEmpty
                ? nil : .init(additionalProperties: options.tags),
            text: callID == nil,
            title: options.title,
            tools: options.tools.map {
                Components.Schemas.SessionTool(
                    description: $0.description,
                    displayTitle: $0.displayTitle,
                    executor: $0.executor.flatMap { .init(rawValue: $0.rawValue) },
                    name: $0.name,
                    parameters: $0.parameters.map(container(for:)))
            },
            tts: options.tts,
            voice: options.voice)

        let output = try await call { try await $0.createSession(body: .json(body)) }
        switch output {
        case .created(let response):
            return Session(try response.body.json)
        default:
            throw AgentsError.undescribedSuccess
        }
    }

    /// Follows a session this caller already has open, without creating one.
    ///
    /// Use this when the app opened a session and is coming back to it — after a relaunch,
    /// or on another screen. A session opened by somebody else is not found, because reading
    /// one is reading a conversation.
    public func attach(sessionID: String, tools: [AgentTool] = []) async throws -> AgentSession {
        let session = try await sessions.get(sessionID)
        return await AgentSession(backend: backend, session: session, tools: tools)
    }

    /// Stops a session, which is how the agent leaves. What it recorded and remembered is
    /// kept; `sessions.delete` takes it away.
    public func close(sessionID: String) async throws {
        let output = try await call { try await $0.stopSession(path: .init(id: sessionID)) }
        switch output {
        case .noContent:
            return
        default:
            throw AgentsError.undescribedSuccess
        }
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
        let output = try await call {
            try await $0.rewindSession(
                path: .init(id: sessionID), body: .json(.init(responseId: responseID)))
        }
        switch output {
        case .noContent:
            return
        default:
            throw AgentsError.undescribedSuccess
        }
    }

    /// Continues a conversation as a new session, leaving the parent as it was.
    ///
    /// Follow the fork the way any session is followed, with `attach(sessionID:)`.
    public func fork(sessionID: String, _ options: ForkOptions = ForkOptions()) async throws -> Session {
        let body = Components.Schemas.ForkSessionRequest(
            callId: options.callID,
            configId: options.agent.flatMap { $0.isEmpty ? nil : $0 },
            instructions: options.instructions,
            messages: options.withoutHistory ? false : nil,
            projectId: options.projectID,
            responseId: options.responseID.flatMap { $0.isEmpty ? nil : $0 },
            title: options.title)

        let output = try await call {
            try await $0.forkSession(path: .init(id: sessionID), body: .json(body))
        }
        switch output {
        case .created(let response):
            return Session(try response.body.json)
        default:
            throw AgentsError.undescribedSuccess
        }
    }

    private func call<T>(_ body: (Client) async throws -> T) async throws -> T {
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
        let output = try await agents.backend.call { try await $0.getSession(path: .init(id: id)) }
        switch output {
        case .ok(let response):
            return Session(try response.body.json)
        default:
            throw AgentsError.undescribedSuccess
        }
    }

    /// Renames or relabels a conversation, running or ended. Nil leaves a field as it is, and
    /// `custom` replaces the labels whole.
    public func update(
        _ id: String, title: String? = nil, description: String? = nil,
        custom: [String: JSONValue]? = nil
    ) async throws -> Session {
        let body = Components.Schemas.UpdateSessionRequest(
            custom: custom.map(labels(for:)), description: description, title: title)
        let output = try await agents.backend.call {
            try await $0.updateSession(path: .init(id: id), body: .json(body))
        }
        switch output {
        case .ok(let response):
            return Session(try response.body.json)
        default:
            throw AgentsError.undescribedSuccess
        }
    }

    /// Deletes a conversation, running or ended: it is stopped, and its turns and what it
    /// remembered are deleted with it.
    public func delete(_ id: String) async throws {
        let output = try await agents.backend.call { try await $0.deleteSession(path: .init(id: id)) }
        switch output {
        case .noContent:
            return
        default:
            throw AgentsError.undescribedSuccess
        }
    }

    /// A session's turns, for one read back from `query` rather than held.
    public func responses(_ sessionID: String) -> Responses {
        Responses(backend: agents.backend, sessionID: sessionID)
    }

    private func sessions(matching text: String?, _ query: SessionQuery) async throws -> Page<Session> {
        let body = Components.Schemas.SessionQuery(
            cursor: query.cursor,
            filter: .init(
                agent: equals(agent),
                agentId: equals(query.agentID),
                modality: equals(query.modality?.rawValue),
                projectId: equals(query.projectID),
                state: equals(query.state?.rawValue),
                text: text.map { .init(_dollar_q: $0) }),
            limit: query.limit.map(Int64.init))
        let output = try await agents.backend.call { try await $0.querySessions(body: .json(body)) }
        switch output {
        case .ok(let response):
            let page = try response.body.json
            return Page(
                items: page.items.map(Session.init), hasMore: page.hasMore,
                nextCursor: page.nextCursor)
        default:
            throw AgentsError.undescribedSuccess
        }
    }
}

/// A filter field matching `value` exactly, or nil to leave the field out.
private func equals(_ value: String?) -> Components.Schemas.Equals? {
    value.flatMap { $0.isEmpty ? nil : .case1($0) }
}

extension Backend {
    /// Runs one request, reporting a transport failure as one and leaving cancellation alone.
    func call<T>(_ body: (Client) async throws -> T) async throws -> T {
        do {
            return try await body(client())
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AgentsError {
            throw error
        } catch let error as ClientError {
            if error.underlyingError is CancellationError { throw CancellationError() }
            if let error = error.underlyingError as? AgentsError { throw error }
            // A response arrived, so this is a success whose body did not read.
            if error.response != nil { throw AgentsError.unreadable("\(error.underlyingError)") }
            throw AgentsError.transport(error.underlyingError)
        } catch {
            throw AgentsError.transport(error)
        }
    }
}

extension AgentsError {
    /// An operation's answer that is neither the success it names nor a failure, which
    /// `FailureMiddleware` threw before the generated client read it: a success status the spec
    /// does not describe.
    static let undescribedSuccess = AgentsError.unreadable("a success status the spec does not describe")
}

/// Hands a JSON Schema object to the generated client, which holds open-ended objects in a
/// container of its own.
private func container(for schema: JSONValue) -> Components.Schemas.SessionTool.ParametersPayload {
    let encoded = try? JSONEncoder().encode(schema)
    let decoded =
        encoded.flatMap {
            try? JSONDecoder().decode([String: OpenAPIValueContainer].self, from: $0)
        } ?? [:]
    return .init(additionalProperties: decoded)
}

private func labels(for custom: [String: JSONValue]) -> Components.Schemas.UpdateSessionRequest.CustomPayload {
    let encoded = try? JSONEncoder().encode(custom)
    let decoded =
        encoded.flatMap {
            try? JSONDecoder().decode([String: OpenAPIValueContainer].self, from: $0)
        } ?? [:]
    return .init(additionalProperties: decoded)
}
