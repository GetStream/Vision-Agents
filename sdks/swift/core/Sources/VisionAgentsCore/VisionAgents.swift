import Foundation
import OpenAPIRuntime

/// What a session should be, for the cases the shorthands do not cover.
///
/// Everything is optional because everything has an answer already: a named config decides
/// what it does not say, and the router decides what the config does not. Setting a field here
/// overrides both, for this session only.
public struct SessionOptions: Sendable {
    /// An agent config to start from, by name or by id. Names are resolved first.
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
///     let chat = try await agents.chat(agent: "swift_demo")
///
/// Configuring what an agent is, ingesting knowledge and waiting for dispatched calls are not
/// here and cannot be: the router refuses them from a device. They belong to a backend, which
/// has the Go or the Python SDK.
public struct VisionAgents: Sendable {
    public let backend: Backend

    public init(url: URL, customerID: String, urlSession: URLSession = .shared) {
        backend = Backend(url: url, customerID: customerID, urlSession: urlSession)
    }

    public init(backend: Backend) {
        self.backend = backend
    }

    /// The agent configs this customer holds, newest first.
    public func agentConfigs() async throws -> [AgentConfig] {
        let output = try await call { try await $0.listAgentConfigs(.init()) }
        switch output {
        case .ok(let response):
            return try response.body.json.map(AgentConfig.init)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// The config with this name.
    public func agentConfig(named name: String) async throws -> AgentConfig {
        guard let found = try await agentConfigs().first(where: { $0.name == name }) else {
            throw AgentsError.unknownAgent(name)
        }
        return found
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
        var configID: String?
        if let agent = options.agent, !agent.isEmpty {
            // The router looks configs up by id, so a name is resolved here. Passing an id
            // through costs one list call and is worth not making callers care which they hold.
            configID = try? await agentConfig(named: agent).id
            if configID == nil { configID = agent }
        }

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

    /// Credentials for joining the Stream call an agent is on.
    ///
    /// `sessionID` is the session's own id, not its call id. The token names the call to join.
    public func callToken(
        sessionID: String,
        userID: String? = nil,
        userName: String? = nil
    ) async throws -> CallToken {
        let output = try await call {
            try await $0.createCallToken(
                path: .init(id: sessionID),
                body: .json(.init(userId: userID, userName: userName)))
        }
        switch output {
        case .ok(let response):
            return CallToken(try response.body.json)
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

    /// Credentials for reading and writing an agent's conversation channel.
    public func chatToken(
        agentID: String,
        userID: String? = nil,
        userName: String? = nil
    ) async throws -> ChatToken {
        let output = try await call {
            try await $0.createChatToken(
                body: .json(.init(agentId: agentID, userId: userID, userName: userName)))
        }
        switch output {
        case .ok(let response):
            return ChatToken(try response.body.json)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// The calls this customer has a record of, newest first.
    public func calls(agentID: String? = nil, limit: Int? = nil) async throws -> [CallRecord] {
        let output = try await call {
            try await $0.listCalls(query: .init(agentId: agentID, limit: limit))
        }
        switch output {
        case .ok(let response):
            return try response.body.json.map(CallRecord.init)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// What was said on a call, oldest first.
    public func transcript(callID: String) async throws -> [TranscriptMessage] {
        let output = try await call {
            try await $0.getCallTranscript(path: .init(id: callID))
        }
        switch output {
        case .ok(let response):
            return try response.body.json.map(TranscriptMessage.init)
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
