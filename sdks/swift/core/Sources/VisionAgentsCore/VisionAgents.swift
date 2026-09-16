import Foundation
import OpenAPIRuntime

/// What a session should be, for the cases the shorthands do not cover.
///
/// Everything is optional because everything has an answer already: a named config decides
/// what it does not say, and the router decides what the config does not. Setting a field here
/// overrides both, for this session only.
public struct SessionOptions: Sendable {
    /// An agent config to start from, by id.
    ///
    /// An id rather than a name: reading the configs is server-side only, so there is
    /// nothing here to resolve a name against. Whoever built the app knows which agent it
    /// talks to, and passes the id down.
    public var agent: String?
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
/// Two lines get a conversation going:
///
///     let agents = VisionAgents(url: url, customerID: "acme")
///     let chat = try await agents.chat(agent: configID)
///
/// Opening a conversation, finding and ending one is the whole of what is here, because it is
/// the whole of what the router lets a device do. What an agent is configured as, what it
/// said on an earlier call and a token to join a call with are all server-side only: they
/// belong to a backend, which has the Go or the Python SDK, and which hands down what the app
/// needs.
public struct VisionAgents: Sendable {
    public let backend: Backend

    public init(url: URL, customerID: String, urlSession: URLSession = .shared) {
        backend = Backend(url: url, customerID: customerID, urlSession: urlSession)
    }

    public init(backend: Backend) {
        self.backend = backend
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
        let configID = options.agent.flatMap { $0.isEmpty ? nil : $0 }

        let body = Components.Schemas.CreateSessionRequest(
            callId: callID,
            text: callID == nil,
            configId: configID,
            instructions: options.instructions,
            greeting: options.greeting,
            llm: options.llm,
            stt: options.stt,
            tts: options.tts,
            voice: options.voice,
            tools: options.tools.map {
                Components.Schemas.SessionTool(
                    name: $0.name,
                    description: $0.description,
                    parameters: $0.parameters.map(container(for:)))
            },
            tags: options.tags.isEmpty
                ? nil : .init(additionalProperties: options.tags))

        let output = try await call { try await $0.createSession(body: .json(body)) }
        switch output {
        case .created(let response):
            return Session(try response.body.json)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// The sessions this caller has open, newest first.
    ///
    /// Only ever this caller's own. The router owns a session by whoever opened it, so a
    /// device is never told about anybody else's conversation.
    public func sessions() async throws -> [Session] {
        let output = try await call { try await $0.listSessions(.init()) }
        switch output {
        case .ok(let response):
            return try response.body.json.map(Session.init)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// Follows a session this caller already has open, without creating one.
    ///
    /// Use this when the app opened a session and is coming back to it — after a relaunch,
    /// or on another screen. A session opened by somebody else is not found, because reading
    /// one is reading a conversation.
    public func attach(sessionID: String, tools: [AgentTool] = []) async throws -> AgentSession {
        guard let session = try await sessions().first(where: { $0.id == sessionID }) else {
            throw AgentsError.http(status: 404, message: "no such session")
        }
        return await AgentSession(backend: backend, session: session, tools: tools)
    }

    /// Ends a session, which is how the agent leaves.
    public func close(sessionID: String) async throws {
        let output = try await call { try await $0.closeSession(path: .init(id: sessionID)) }
        switch output {
        case .noContent:
            return
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// Runs one request, reporting a transport failure as one and leaving cancellation alone.
    private func call<T>(_ body: (Client) async throws -> T) async throws -> T {
        do {
            return try await body(backend.client())
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AgentsError {
            throw error
        } catch let error as ClientError {
            if error.underlyingError is CancellationError { throw CancellationError() }
            throw AgentsError.transport(error.underlyingError)
        } catch {
            throw AgentsError.transport(error)
        }
    }
}

/// Hands a JSON Schema object to the generated client, which holds open-ended objects in a
/// container of its own.
private func container(for schema: JSONValue) -> Components.Schemas.SessionTool.ParametersPayload {
    let encoded = try? JSONEncoder().encode(schema)
    let decoded =
        encoded.flatMap {
            try? JSONDecoder().decode(OpenAPIObjectContainer.self, from: $0)
        } ?? OpenAPIObjectContainer()
    return .init(additionalProperties: decoded)
}
