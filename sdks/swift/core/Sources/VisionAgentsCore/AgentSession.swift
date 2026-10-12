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
    /// The session the router opened, as of the last `update`.
    public private(set) var session: Session

    /// The transcript and what the agent is doing.
    public private(set) var conversation = Conversation()

    /// Whether the socket is still carrying the conversation.
    public private(set) var isConnected = false

    /// Why the socket stopped, or nil. A conversation that ended normally has none.
    public private(set) var failure: AgentsError?

    /// Calls of tools that ask first, waiting for the person to answer with `decide`, oldest
    /// first. A conversation kept in Stream Chat also shows each on its reply's step.
    public private(set) var approvals: [ToolApprovalRequest] = []

    /// What the router holds this session by, which is what addresses it and its socket.
    public var id: String { session.id }

    public var turns: [Turn] { conversation.turns }
    public var state: Conversation.State { conversation.state }

    /// This session's turns as the router wrote them down: asking, reading back, rewinding.
    ///
    /// What is asked here is shown in `turns` straight away, since a conversation in writing
    /// is never heard back.
    public var responses: Responses {
        Responses(
            backend: backend, sessionID: session.id,
            asked: { [weak self] text in self?.conversation.said(text) })
    }

    /// Where the session lives and who is asking, for the chat channel to connect as.
    let backend: Backend
    private let socket: SessionSocket
    private let tools: [String: AgentTool]
    private var pump: Task<Void, Never>?

    init(backend: Backend, session: Session, tools: [AgentTool]) {
        self.session = session
        self.backend = backend
        self.tools = Dictionary(tools.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        socket = SessionSocket(
            url: backend.socketURL(
                path: "/v1/agents/sessions/\(session.id)/events",
                // Interim transcripts arrive several times a second and decisions are for
                // somebody watching a call, not for an app holding one.
                query: ["decisions": "false"]),
            headers: [:],
            urlSession: backend.urlSession)
    }

    /// Opens the socket and starts following the conversation. Doing this twice does nothing.
    public func start() async {
        guard pump == nil else { return }
        let headers: [String: String]
        do {
            headers = try await backend.headers()
        } catch let error as AgentsError {
            return stopped(error)
        } catch {
            return stopped(.transport(error))
        }
        guard pump == nil else { return }
        let stream = await socket.open(headers: headers)
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

    /// Speaks this without going through the model.
    public func say(_ text: String) async throws {
        try await socket.send(.say(text))
    }

    /// Abandons the reply in flight.
    public func interrupt() async throws {
        try await socket.send(.interrupt)
    }

    /// Has the agent join the session's call, `agent:<session id>`, and carry the conversation
    /// on there. Starting voice that is already on does nothing.
    @discardableResult
    public func startVoice() async throws -> Session {
        session = try await VisionAgents(backend: backend).sessions.startVoice(session.id)
        return session
    }

    /// Takes the agent off the call and carries the conversation on in writing. Stopping voice
    /// that is off does nothing.
    @discardableResult
    public func stopVoice() async throws -> Session {
        session = try await VisionAgents(backend: backend).sessions.stopVoice(session.id)
        return session
    }

    /// Renames or relabels this conversation. Nil leaves a field as it is, and `custom`
    /// replaces the labels whole.
    @discardableResult
    public func update(
        title: String? = nil, description: String? = nil, custom: [String: JSONValue]? = nil
    ) async throws -> Session {
        session = try await VisionAgents(backend: backend).sessions.update(
            session.id, title: title, description: description, custom: custom)
        return session
    }

    /// Answers a call waiting in `approvals`.
    ///
    /// Allowed, the tool runs and its result goes back as it would have. Declined, it never
    /// runs: the model is told the person declined, and the call's step shows `summary`. A
    /// call that is not waiting is left alone, so answering twice does nothing. If the answer
    /// cannot be sent, the call waits again.
    public func decide(_ toolCallID: String, allowed: Bool, summary: String = "") async throws {
        guard let index = approvals.firstIndex(where: { $0.id == toolCallID }) else { return }
        let request = approvals.remove(at: index)
        let call = request.call
        do {
            try await socket.send(
                .toolApproval(
                    id: call.id, allowed: allowed, summary: summary, requestID: call.requestID,
                    turnID: call.turnID))
        } catch {
            approvals.insert(request, at: min(index, approvals.count))
            throw error
        }
        if allowed {
            run(call, with: request.tool)
        } else {
            try await socket.send(
                .toolResult(
                    id: call.id, output: nil, error: "The person declined this call.",
                    requestID: call.requestID, turnID: call.turnID))
        }
    }

    /// Ends the session and closes the socket. What it recorded and remembered is kept.
    public func close() async {
        try? await socket.send(.close)
        await socket.close()
        pump?.cancel()
        pump = nil
        isConnected = false
        approvals = []
        conversation.state = .ended
    }

    /// Deletes this conversation: it is stopped, and its turns and what it remembered go with
    /// it.
    public func delete() async throws {
        try await VisionAgents(backend: backend).sessions.delete(session.id)
        await close()
    }

    private func stopped(_ error: AgentsError?) {
        failure = error
        isConnected = false
        approvals = []
        conversation.state = .ended
    }

    private func apply(_ event: AgentEvent) {
        conversation.apply(event)
        if let call = event.toolCall {
            answer(call)
        } else if event.kind == .toolCancel {
            approvals.removeAll { $0.id == event["id"].stringValue }
        }
    }

    /// Runs a tool the model asked for, or holds the call for the person when the tool asks
    /// first.
    private func answer(_ call: AgentEvent.ToolCall) {
        // Session events are broadcast to observers as well as tool owners. A Python
        // video worker may own this request; an observer must not resolve it first.
        guard let tool = tools[call.name] else { return }
        if let approval = tool.approval {
            approvals.append(ToolApprovalRequest(call: call, approval: approval, tool: tool))
        } else {
            run(call, with: tool)
        }
    }

    /// Runs a tool and sends back what it returned.
    ///
    /// A task of its own, so a slow tool does not hold up the transcript. The handler is a
    /// nonisolated async closure, so its body does not run on the main actor even though this
    /// call site is on it.
    private func run(_ call: AgentEvent.ToolCall, with tool: AgentTool) {
        Task { [socket] in
            do {
                let output = try await tool.run(call.argumentValues)
                try await socket.send(
                    .toolResult(
                        id: call.id, output: output, error: nil, requestID: call.requestID,
                        turnID: call.turnID))
            } catch {
                try? await socket.send(
                    .toolResult(
                        id: call.id, output: nil, error: error.localizedDescription,
                        requestID: call.requestID, turnID: call.turnID))
            }
        }
    }
}

/// A call of one of your tools, waiting for the person to allow it.
public struct ToolApprovalRequest: Sendable, Identifiable {
    public let call: AgentEvent.ToolCall
    /// The question the tool asks.
    public let approval: AgentTool.Approval
    let tool: AgentTool

    public var id: String { call.id }

    /// The model's own words for why it wants the call, from the argument the approval names,
    /// or the empty string.
    public var reason: String {
        approval.reasonArgument.flatMap { call.argumentValues[$0]?.stringValue } ?? ""
    }
}
