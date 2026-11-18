import Combine
import Foundation
import StreamCore

/// A live conversation, in a shape SwiftUI can bind to: hold it with `@StateObject` or
/// `@ObservedObject`.
///
/// The whole object is on the main actor: it exists to be read by views, and the alternative
/// -- an actor holding the state with a main-actor copy beside it -- means two truths and a
/// window in which they disagree. The concurrency is in the socket, which is an actor of its
/// own, and the read loop below is main-actor isolated, so folding an event into the
/// transcript needs no hop and no lock.
@MainActor
public final class AgentSession: ObservableObject {
    /// The session the router opened, as of the last `update`.
    @Published public private(set) var session: Session

    /// The transcript and what the agent is doing.
    @Published public private(set) var conversation = Conversation()

    /// Whether the socket is still carrying the conversation.
    @Published public private(set) var isConnected = false

    /// Why the socket stopped, or nil. A conversation that ended normally has none.
    @Published public private(set) var failure: AgentsError?

    /// Calls of tools that ask first, waiting for the person to answer with `decide`, oldest
    /// first. A conversation kept in Stream Chat also shows each on its reply's step.
    @Published public private(set) var approvals: [ToolApprovalRequest] = []

    /// Every tool call of the session by its id, as it was last reported: running, waiting for
    /// a person's approval, or how it ended, whichever process runs it. Finished calls stay.
    @Published public private(set) var toolCalls: [String: ToolCallState] = [:]

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
    /// The tools whose calls are answered here, by name.
    private(set) var tools: [String: AgentTool]
    /// The tools whose adding or removal was never answered, so the router may or may not offer
    /// them.
    private var unsettledTools: Set<String> = []
    /// Whether a change of tools is under way, and the ones waiting for it (`oneAtATime`).
    private var changingTools = false
    private var waitingToChangeTools: [CheckedContinuation<Void, Never>] = []
    private var pump: Task<Void, Never>?
    /// Whoever is reading `events()`, each with a stream of their own, or nil once the session
    /// has stopped and there is nothing more to read.
    private var subscribers: [UUID: AsyncStream<AgentEvent>.Continuation]? = [:]
    /// What `on(_:perform:)` registered, locked rather than main-actor isolated so a
    /// cancellable released anywhere takes its handler away at once.
    private let handlers = AllocatedUnfairLock<[UUID: (kind: AgentEvent.Kind, handle: @MainActor (AgentEvent) -> Void)]>([:])
    /// The frames this device's camera showed lately, while it is on a call with it on, which
    /// answer the router's `get_video_frames`.
    var cameraFrames: CameraFrames?

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

    /// Every event as it arrives, for a caller building on more than `conversation` holds:
    /// latency timings, tool events, a frame the router added after this SDK shipped.
    ///
    /// Each call is a stream of its own from that moment on; nothing is replayed. A reader more
    /// than 256 events behind is finished rather than handed a stream with holes in it, and it
    /// ends when the session does. Tool calls are answered whether or not anybody reads this.
    public func events() -> AsyncStream<AgentEvent> {
        let (stream, continuation) = AsyncStream.makeStream(
            of: AgentEvent.self, bufferingPolicy: .bufferingNewest(256))
        if subscribers == nil {
            continuation.finish()
        } else {
            subscribers?[UUID()] = continuation
        }
        return stream
    }

    /// Calls `handler` with every event of `kind`, such as `.toolCallFinished`, until the
    /// returned value is cancelled or released. It needs no loop over `events()`.
    public func on(_ kind: AgentEvent.Kind, perform handler: @escaping @MainActor (AgentEvent) -> Void) -> AnyCancellable {
        let id = UUID()
        handlers.withLock { $0[id] = (kind, handler) }
        return AnyCancellable { [handlers] in handlers.withLock { $0[id] = nil } }
    }

    /// Speaks this without going through the model. `interrupt` stops what the agent is saying
    /// first, rather than waiting for it to finish.
    public func say(_ text: String, interrupt: Bool = false) async throws {
        if interrupt {
            try await socket.send(.interrupt())
        }
        try await socket.send(.say(text))
    }

    /// Abandons the reply in flight, or with `requestID` only the one answering that request,
    /// wherever it got to. An event's `request_id` names it.
    public func interrupt(requestID: String = "") async throws {
        try await socket.send(.interrupt(requestID: requestID))
    }

    /// Has the agent join the session's call, `agent:<session id>`, and carry the conversation
    /// on there. Starting voice that is already on does nothing.
    @discardableResult
    public func startVoice() async throws -> Session {
        session = try await StreamAgents(backend: backend).sessions.startVoice(session.id)
        return session
    }

    /// Takes the agent off the call and carries the conversation on in writing. Stopping voice
    /// that is off does nothing.
    @discardableResult
    public func stopVoice() async throws -> Session {
        session = try await StreamAgents(backend: backend).sessions.stopVoice(session.id)
        return session
    }

    /// Renames, relabels, pins or unpins this conversation. Nil leaves a field as it is,
    /// `custom` replaces the labels whole, and `pinned` true pins it and false unpins.
    @discardableResult
    public func update(
        title: String? = nil, description: String? = nil, custom: [String: RawJSON]? = nil,
        pinned: Bool? = nil
    ) async throws -> Session {
        session = try await StreamAgents(backend: backend).sessions.update(
            session.id, title: title, description: description, custom: custom, pinned: pinned)
        return session
    }

    /// Pins this conversation, which `SessionSort.pinned` lists first. `session.pinnedAt` then
    /// says when.
    @discardableResult
    public func pin() async throws -> Session {
        try await update(pinned: true)
    }

    /// Unpins this conversation.
    @discardableResult
    public func unpin() async throws -> Session {
        try await update(pinned: false)
    }

    /// Continues this conversation as a new session held in writing, leaving this one as it
    /// was. Follow the fork with `StreamAgents.attach(sessionID:)`.
    public func fork(_ options: ForkOptions = ForkOptions()) async throws -> Session {
        try await StreamAgents(backend: backend).fork(sessionID: session.id, options)
    }

    /// Gives the agent more of your tools while the conversation runs, such as the ones a card
    /// shown in it brings. Their calls are answered here, as those of the tools the session was
    /// opened with are.
    ///
    /// The router waits for the reply in progress, and every call it made, to finish, and the
    /// model is offered the tools from the next turn. So do not wait for this from inside one of
    /// this session's tools: it waits for that call. A tool named as one this session already
    /// has is refused: to change one, `remove(toolNamed:)` it first, so that no call of it runs
    /// half under each. Only a session this device opened takes tools from it, and a name the
    /// agent has of its own, such as a connector's tool, is refused. A refusal is a 4xx, and takes
    /// the tools back; after any other failure they stay answered, since the router may have
    /// added them, until the next add or remove of the name settles it.
    public func add(tools: [AgentTool]) async throws {
        guard !tools.isEmpty else { return }
        try await oneAtATime {
            // A name whose adding or removal was never answered may or may not be the router's:
            // it is withdrawn first, so the router and this session agree on which tool it is.
            for tool in tools where unsettledTools.contains(tool.name) {
                try await withdraw(tool.name)
            }
            // Answered from before the router says they are added: the next turn can call one
            // while that answer is still on its way here. A name already answered here keeps
            // its handler, since the router refuses to add it again.
            let added = tools.filter { self.tools[$0.name] == nil }
            for tool in added {
                self.tools[tool.name] = tool
            }
            let id = session.id
            let body = AddSessionToolsRequest(tools: tools.map(\.schema))
            do {
                try await backend.call { @Sendable in try await $0.addSessionTools(id: id, addSessionToolsRequest: body) }
            } catch AgentsError.http(let failure) where (400..<500).contains(failure.status) {
                // Refused, so the router offers none of them.
                for tool in added {
                    self.tools[tool.name] = nil
                }
                throw AgentsError.http(failure)
            } catch {
                // Whether the router offers them is unknown, since a gateway in front of it can
                // answer a 5xx after it added them, so they stay answered until a later add or
                // remove of the name settles it.
                unsettledTools.formUnion(added.map(\.name))
                throw error
            }
        }
    }

    /// Withdraws one of your tools. The router waits for the reply in progress, and every call it
    /// made, to finish, as `add(tools:)` does, so once this returns no call of the tool can still
    /// arrive, and its handler goes. If neither a success nor a 4xx comes back, the handler stays,
    /// since the router may still offer the tool, until the next remove or add of the name
    /// settles it.
    public func remove(toolNamed name: String) async throws {
        try await oneAtATime { try await withdraw(name) }
    }

    /// Withdraws a tool and drops its handler once the router has. A name whose adding or removal
    /// was never answered is settled by the router having none by it as well as by withdrawing it.
    private func withdraw(_ name: String) async throws {
        let id = session.id
        do {
            try await backend.call { @Sendable in try await $0.removeSessionTool(id: id, name: name) }
        } catch AgentsError.http(let failure) where failure.status == 404 && unsettledTools.contains(name) {
        } catch AgentsError.http(let failure) where (400..<500).contains(failure.status) {
            // Refused, so the router offers the tool as it did.
            throw AgentsError.http(failure)
        } catch {
            // Whether the router still offers it is unknown, since a gateway in front of it can
            // answer a 5xx after it withdrew it, so it stays answered until a later add or remove
            // of the name settles it.
            unsettledTools.insert(name)
            throw error
        }
        tools[name] = nil
        unsettledTools.remove(name)
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

    /// Declines the sign-in that a call of one of the agent's connectors waits for, whose step in
    /// the conversation is `awaiting_authorization`. The router answers the call itself, so it
    /// runs nothing and the agent is told the person did not connect. A call that is not waiting
    /// is left alone.
    public func declineLogin(_ toolCallID: String) async throws {
        try await socket.send(.toolApproval(id: toolCallID, allowed: false))
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
        finishSubscribers()
    }

    /// Deletes this conversation: it is stopped, and its turns and what it remembered go with
    /// it.
    public func delete() async throws {
        try await StreamAgents(backend: backend).sessions.delete(session.id)
        await close()
    }

    /// Runs one change of tools at a time, in the order they were asked for. A handler is
    /// put in place before the router is asked and taken back when it refuses, so two changes
    /// interleaved across their requests could take back what the other put in place.
    private func oneAtATime(_ change: () async throws -> Void) async throws {
        if changingTools {
            await withCheckedContinuation { waitingToChangeTools.append($0) }
        } else {
            changingTools = true
        }
        defer {
            if waitingToChangeTools.isEmpty {
                changingTools = false
            } else {
                waitingToChangeTools.removeFirst().resume()
            }
        }
        try await change()
    }

    private func stopped(_ error: AgentsError?) {
        failure = error
        isConnected = false
        approvals = []
        conversation.state = .ended
        finishSubscribers()
    }

    private func finishSubscribers() {
        subscribers?.values.forEach { $0.finish() }
        subscribers = nil
    }

    private func apply(_ event: AgentEvent) {
        conversation.apply(event)
        if let state = event.toolCallState {
            toolCalls[state.id] = state
        }
        if let kind = event.kind {
            for handler in handlers.withLock({ $0.values.filter { $0.kind == kind }.map(\.handle) }) {
                handler(event)
            }
        }
        for (id, continuation) in subscribers ?? [:] {
            switch continuation.yield(event) {
            case .enqueued:
                break
            case .dropped:
                // Fallen behind: finishing says so, where dropping would leave a silent hole.
                continuation.finish()
                subscribers?[id] = nil
            default:
                // Terminated: the reader stopped reading.
                subscribers?[id] = nil
            }
        }
        if let call = event.toolCall {
            answer(call)
        } else if event.kind == .toolCancel {
            approvals.removeAll { $0.id == event.string("id") }
        }
    }

    /// Runs a tool the model asked for, or holds the call for the person when the tool asks
    /// first.
    private func answer(_ call: AgentEvent.ToolCall) {
        // Session events are broadcast to observers as well as tool owners. A Python
        // video worker may own this request; an observer must not resolve it first.
        guard call.run else { return }
        if call.name == CameraFrames.toolName, let cameraFrames {
            return show(cameraFrames, for: call)
        }
        guard let tool = tools[call.name] else { return }
        if let approval = tool.approval, !call.approved {
            approvals.append(ToolApprovalRequest(call: call, approval: approval, tool: tool))
        } else {
            run(call, with: tool)
        }
    }

    /// Answers the router's request for frames with the ones the camera showed at the moment it
    /// names: what the frames are, then the frames, oldest first. Nothing recent enough is said
    /// as the answer rather than as a failure, as the router's own video workers say it, so the
    /// agent can ask the person to show it again.
    private func show(_ frames: CameraFrames, for call: AgentEvent.ToolCall) {
        let arguments = call.argumentValues
        let at = Date(timeIntervalSince1970: (arguments["at_ms"]?.numberValue ?? Date().timeIntervalSince1970 * 1000) / 1000)
        let limit = Int(arguments["limit"]?.numberValue ?? 1)
        let answer: Command
        do {
            let selected = try frames.select(at: at, limit: limit)
            let described = selected.map { frame in
                ["frame_id": "camera-\(frame.id)", "captured_at_ms": Int(frame.capturedAt.timeIntervalSince1970 * 1000)] as [String: Any]
            }
            let words = (try? JSONSerialization.data(
                withJSONObject: ["source": "camera", "clock": "device_unix_ms", "frames": described],
                options: [.sortedKeys])).map { String(decoding: $0, as: UTF8.self) } ?? ""
            answer = .toolResult(
                id: call.id, output: words, error: nil,
                images: selected.map { ImageSource(url: "data:image/jpeg;base64," + $0.jpeg.base64EncodedString()) },
                requestID: call.requestID, turnID: call.turnID)
        } catch {
            answer = .toolResult(
                id: call.id, output: error.localizedDescription, error: nil, requestID: call.requestID,
                turnID: call.turnID)
        }
        Task { [socket] in try? await socket.send(answer) }
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
                        id: call.id, output: output.text, error: nil, images: output.images,
                        requestID: call.requestID, turnID: call.turnID))
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
