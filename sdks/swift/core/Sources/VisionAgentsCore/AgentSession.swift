import Foundation
import Observation

/// A live conversation, in a shape SwiftUI can bind to.
///
/// The whole object is on the main actor: it exists to be read by views, and the alternative
/// -- an actor holding the state with a main-actor copy beside it -- means two truths and a
/// window in which they disagree. The concurrency is in the socket, which is an actor of its
/// own, and the read loop below is main-actor isolated, so folding an event into the
/// transcript needs no hop and no lock.
@MainActor
@Observable
public final class AgentSession {
    /// The session the router opened.
    public let session: Session

    /// The transcript and what the agent is doing.
    public private(set) var conversation = Conversation()

    /// Whether the socket is still carrying the conversation.
    public private(set) var isConnected = false

    /// Why the socket stopped, or nil. A conversation that ended normally has none.
    public private(set) var failure: AgentsError?

    /// What the router holds this session by, which is what addresses it and its socket.
    public var id: String { session.id }

    public var turns: [Turn] { conversation.turns }
    public var state: Conversation.State { conversation.state }

    private let socket: SessionSocket
    private let tools: [String: AgentTool]
    private var pump: Task<Void, Never>?

    init(backend: Backend, session: Session, tools: [AgentTool]) {
        self.session = session
        self.tools = Dictionary(tools.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        socket = SessionSocket(
            url: backend.socketURL(
                path: "/v1/agents/sessions/\(session.id)/events",
                // Interim transcripts arrive several times a second and decisions are for
                // somebody watching a call, not for an app holding one.
                query: ["decisions": "false"]),
            headers: backend.headers,
            urlSession: backend.urlSession)
    }

    /// Opens the socket and starts following the conversation. Doing this twice does nothing.
    public func start() async {
        guard pump == nil else { return }
        let stream = await socket.open()
        isConnected = true
        pump = Task { [weak self] in
            do {
                for try await event in stream {
                    // Weakly, so that a session nobody holds any more stops rather than
                    // keeping itself alive through its own read loop.
                    guard let self else { return }
                    self.apply(event)
                }
                self?.stopped(nil)
            } catch let error as AgentsError {
                self?.stopped(error)
            } catch {
                self?.stopped(.transport(error))
            }
        }
    }

    /// Says this to the agent, as though it had been heard.
    public func send(_ text: String) async throws {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        conversation.said(trimmed)
        try await socket.send(.respond(trimmed))
    }

    /// Speaks this without going through the model.
    public func say(_ text: String) async throws {
        try await socket.send(.say(text))
    }

    /// Abandons the reply in flight.
    public func interrupt() async throws {
        try await socket.send(.interrupt)
    }

    /// Replaces the system prompt, from the next turn on.
    public func setInstructions(_ instructions: String) async throws {
        try await socket.send(.instructions(instructions))
    }

    /// Ends the session and closes the socket.
    public func close() async {
        try? await socket.send(.close)
        await socket.close()
        pump?.cancel()
        pump = nil
        isConnected = false
        conversation.state = .ended
    }

    private func stopped(_ error: AgentsError?) {
        failure = error
        isConnected = false
        conversation.state = .ended
    }

    private func apply(_ event: AgentEvent) {
        conversation.apply(event)
        if let call = event.toolCall {
            answer(call)
        }
    }

    /// Runs a tool the model asked for and sends back what it returned.
    ///
    /// A task of its own, so a slow tool does not hold up the transcript. The handler is a
    /// nonisolated async closure, so its body does not run on the main actor even though this
    /// call site is on it.
    private func answer(_ call: AgentEvent.ToolCall) {
        let tool = tools[call.name]
        Task { [socket] in
            guard let tool else {
                try? await socket.send(
                    .toolResult(id: call.id, output: nil, error: "no tool called \(call.name)"))
                return
            }
            do {
                let output = try await tool.run(call.argumentValues)
                try await socket.send(.toolResult(id: call.id, output: output, error: nil))
            } catch {
                try? await socket.send(
                    .toolResult(id: call.id, output: nil, error: error.localizedDescription))
            }
        }
    }
}
