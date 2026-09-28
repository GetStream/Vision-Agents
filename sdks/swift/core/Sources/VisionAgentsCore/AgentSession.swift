import AGUI
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

    /// What the agent is waiting to be allowed to do, in the order it asked.
    ///
    /// A tool declared with an `approval` question is never run on the strength of the model
    /// asking for it: what turns up here is the AG-UI interrupt the run ended on, the same
    /// value `aguiEvents()` carried, and `approve(_:)` or `decline(_:reason:)` answers it.
    ///
    /// The router is told the question is on screen, so it waits for a person rather than for
    /// a machine. An entry leaves this by being answered, or because the router gave up
    /// waiting and told the model nothing was approved.
    public private(set) var pendingApprovals: [Interrupt] = []

    /// What the router holds this session by, which is what addresses it and its socket.
    public var id: String { session.id }

    public var turns: [Turn] { conversation.turns }
    public var state: Conversation.State { conversation.state }

    private let socket: SessionSocket
    private let tools: [String: AgentTool]
    private var pump: Task<Void, Never>?

    /// The conversation as the AG-UI protocol describes it, which is what `aguiEvents()`
    /// publishes. One translator, fed by the one read loop, so every subscriber sees the same
    /// runs in the same order.
    private var translator: AGUITranslator

    /// The tool calls somebody has yet to allow, by the id of the interrupt asking.
    private var awaiting: [String: AgentEvent.ToolCall] = [:]

    private var listeners: [UUID: AsyncStream<AGUI.Event>.Continuation] = [:]

    init(backend: Backend, session: Session, tools: [AgentTool]) {
        self.session = session
        self.tools = Dictionary(tools.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        translator = AGUITranslator(threadID: session.id)
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
        // Before the send, so the question is on the stream ahead of the answer to it. The
        // router only reports what it heard on a call, so a written session would otherwise
        // carry a reply to nothing.
        emit(translator.said(trimmed))
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

    /// This conversation as AG-UI protocol events.
    ///
    /// Every caller gets a stream of its own, because one stored stream is not a broadcast:
    /// two iterators would compete for the same elements. A stream nobody keeps up with is
    /// finished rather than left quietly missing events, since a consumer that lost some
    /// cannot rebuild what they described.
    ///
    ///     for await event in session.aguiEvents() {
    ///         changes = conversation.apply(event)   // AGUI.ConversationState
    ///     }
    ///
    /// - Parameter buffering: how many events to hold for a caller that is behind.
    public func aguiEvents(buffering limit: Int = 256) -> AsyncStream<AGUI.Event> {
        let id = UUID()
        let (stream, continuation) = AsyncStream<AGUI.Event>.makeStream(
            bufferingPolicy: .bufferingNewest(limit))
        listeners[id] = continuation
        continuation.onTermination = { [weak self] _ in
            Task { @MainActor in self?.listeners.removeValue(forKey: id) }
        }
        return stream
    }

    /// Allows a call the agent asked permission for. The tool runs now.
    public func approve(_ approval: Interrupt) async throws {
        try await resume([.resolved(approval.id, payload: ["approved": true])])
    }

    /// Refuses a call the agent asked permission for. The tool never runs, and the model is
    /// told nobody allowed it so that it can say so.
    public func decline(_ approval: Interrupt, reason: String? = nil) async throws {
        var payload: [String: AGUI.JSONValue] = ["approved": false]
        if let reason { payload["reason"] = .string(reason) }
        try await resume([.resolved(approval.id, payload: .object(payload))])
    }

    /// Answers what the agent is waiting for, in the protocol's own terms.
    ///
    /// An entry allows the call when its payload says `approved` is true; anything else
    /// refuses it, because a payload that does not say yes is not a yes. An entry for an
    /// approval that is not pending does nothing, which is what a second tap on a card that
    /// has already been answered should do.
    public func resume(_ entries: [ResumeEntry]) async throws {
        for entry in entries {
            guard let call = awaiting.removeValue(forKey: entry.interruptId) else { continue }
            pendingApprovals.removeAll { $0.id == entry.interruptId }
            guard approves(entry) else {
                let refused = refusal(entry)
                // The result rather than the error: the tool did not fail, it did not run.
                // A model told a tool failed apologises for a fault, where it should be
                // telling the caller their answer was no.
                try await socket.send(.toolResult(id: call.id, output: refused, error: nil))
                emit(translator.answered(call.id, with: refused))
                continue
            }
            answer(call)
        }
    }

    /// Ends the session and closes the socket.
    public func close() async {
        try? await socket.send(.close)
        await socket.close()
        pump?.cancel()
        pump = nil
        emit(translator.finish())
        ended()
    }

    private func stopped(_ error: AgentsError?) {
        failure = error
        emit(error.map { translator.failed($0.localizedDescription) } ?? translator.finish())
        ended()
    }

    /// What is true once the conversation is over, however it ended. An approval nobody
    /// answered goes with it: there is no longer a turn waiting on the answer, so a card
    /// still asking for one would be asking on behalf of nobody.
    private func ended() {
        isConnected = false
        conversation.state = .ended
        pendingApprovals.removeAll()
        awaiting.removeAll()
        finishListeners()
    }

    private func apply(_ event: AgentEvent) {
        if event.kind == .toolExpired {
            expired(event.toolCallID)
            return
        }

        conversation.apply(event)
        let call = event.toolCall
        let approval = call.flatMap(approval(for:))
        emit(translator.translate(event, awaiting: approval))
        guard let call else { return }
        if let approval {
            pendingApprovals.append(approval)
            awaiting[approval.id] = call
            // The router is told a person has it, so the conversation waits for them to
            // read the card rather than giving them the seconds it gives a machine.
            Task { [socket] in
                try? await socket.send(.toolWaiting(id: call.id, question: approval.message))
            }
        } else {
            answer(call)
        }
    }

    /// Takes down a question nobody answered in time.
    ///
    /// The router has already told the model that nothing was approved, so the card is asking
    /// on behalf of a turn that has moved on: tapping it would run the tool for nobody.
    private func expired(_ id: String) {
        guard awaiting.removeValue(forKey: id) != nil else { return }
        pendingApprovals.removeAll { $0.id == id }
        emit(translator.answered(id, with: expiredRefusal))
    }

    /// What somebody has to allow before this call can run, or nil for a tool that just runs.
    private func approval(for call: AgentEvent.ToolCall) -> Interrupt? {
        guard let question = tools[call.name]?.approval else { return nil }
        return Interrupt(
            id: call.id,
            reason: InterruptReason.toolCall,
            message: question(call.argumentValues),
            toolCallId: call.id,
            responseSchema: [
                "type": "object",
                "properties": ["approved": ["type": "boolean"]],
                "required": ["approved"],
            ])
    }

    private func approves(_ entry: ResumeEntry) -> Bool {
        guard entry.status == .resolved else { return false }
        return entry.payload?.objectValue?["approved"] == .bool(true)
    }

    /// What the AG-UI stream records for a call nobody got to in time. The model has been
    /// told this already, by the router, which is what ended the wait.
    private let expiredRefusal = "Nobody answered in time, so nothing was done."

    /// What the model is told when nobody allowed a call.
    private func refusal(_ entry: ResumeEntry) -> String {
        let given = entry.payload?.objectValue?["reason"]?.stringValue
        let because = given.map { ": \($0)" } ?? ""
        return "The person you are talking to did not approve this, "
            + "so nothing was done\(because)."
    }

    /// Runs a tool the model asked for and sends back what it returned.
    ///
    /// A task of its own, so a slow tool does not hold up the transcript. The handler is a
    /// nonisolated async closure, so its body does not run on the main actor even though this
    /// call site is on it.
    private func answer(_ call: AgentEvent.ToolCall) {
        let tool = tools[call.name]
        // The socket is held by the task rather than reached through `self`, so a session
        // nobody holds any more still answers the call it took on. Whoever is watching the
        // AG-UI stream is told only if there is still a session to tell them from.
        Task { [weak self, socket] in
            guard let tool else {
                let missing = "no tool called \(call.name)"
                try? await socket.send(.toolResult(id: call.id, output: nil, error: missing))
                self?.answered(call.id, with: missing)
                return
            }
            do {
                let output = try await tool.run(call.argumentValues)
                try await socket.send(.toolResult(id: call.id, output: output, error: nil))
                self?.answered(call.id, with: output)
            } catch {
                let failure = error.localizedDescription
                try? await socket.send(.toolResult(id: call.id, output: nil, error: failure))
                self?.answered(call.id, with: failure)
            }
        }
    }

    /// Publishes what a tool returned, which the router does not report back to the device
    /// that ran it.
    private func answered(_ id: String, with content: String) {
        emit(translator.answered(id, with: content))
    }

    /// Hands events to everybody watching, and stops watching for anybody who has fallen
    /// behind or gone away.
    private func emit(_ events: [AGUI.Event]) {
        guard !events.isEmpty, !listeners.isEmpty else { return }
        let behind = listeners.filter { !deliver(events, to: $0.value) }.keys
        for id in behind {
            listeners.removeValue(forKey: id)?.finish()
        }
    }

    private func deliver(
        _ events: [AGUI.Event],
        to continuation: AsyncStream<AGUI.Event>.Continuation
    ) -> Bool {
        for event in events {
            guard case .enqueued = continuation.yield(event) else { return false }
        }
        return true
    }

    private func finishListeners() {
        for continuation in listeners.values {
            continuation.finish()
        }
        listeners.removeAll()
    }
}
