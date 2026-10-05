import Foundation
import Testing

@testable import VisionAgentsCore

/// Tests that need a router running.
///
/// The Swift answer to `@pytest.mark.integration`: they are skipped unless
/// `VISION_AGENTS_URL` is set, so the ordinary `swift test` stays offline and fast.
///
///     VISION_AGENTS_URL=http://localhost:8080 VISION_AGENTS_CUSTOMER_ID=acme VISION_AGENTS_AGENT=myagent swift test
struct Live {
    static let url = ProcessInfo.processInfo.environment["VISION_AGENTS_URL"]
    static let customerID =
        ProcessInfo.processInfo.environment["VISION_AGENTS_CUSTOMER_ID"] ?? "acme"
    /// The name an agent config was synced under.
    static let agent = ProcessInfo.processInfo.environment["VISION_AGENTS_AGENT"] ?? ""

    static var available: Bool { url != nil }

    static var agents: VisionAgents {
        VisionAgents(url: URL(string: url!)!, customerID: customerID)
    }
}

// On the main actor because `AgentSession` is: a view is what reads it, so that is where
// its state lives and where a test has to look at it.
@MainActor
@Suite(.enabled(if: Live.available), .serialized)
struct LiveTests {
    @Test func aTextSessionJoinsNoCallAndOpensASocket() async throws {
        let session = try await Live.agents.chat(agent: Live.agent)
        defer { Task { await session.close() } }

        #expect(session.session.isText)
        #expect(session.session.callID.isEmpty)
        #expect(session.session.state == .live)

        await session.start()
        #expect(session.isConnected)
    }

    /// The whole round trip: the socket carries a question in and an answer back, one delta at
    /// a time, and the transcript ends up with both sides of it.
    @Test func askingSomethingGetsAnAnswer() async throws {
        let session = try await Live.agents.chat(agent: Live.agent)
        await session.start()
        defer { Task { await session.close() } }

        _ = try await session.responses.create("What are your opening hours? Answer in one sentence.")

        try await until(20) { session.state == .idle && session.turns.count >= 2 }

        let reply = try #require(session.turns.last)
        #expect(reply.speaker == .agent)
        #expect(!reply.text.isEmpty)
        #expect(session.failure == nil)
    }

    /// A tool the model asks for runs here and its answer goes back over the same socket.
    @Test func aToolOnThisSideIsCalledAndAnswered() async throws {
        let asked = Asked()
        let tool = AgentTool(
            name: "lookup_order",
            description: "Look up one of the caller's orders by its order number.",
            parameters: .strings(["order_id": "the order number"], required: ["order_id"])
        ) { arguments in
            await asked.record(arguments["order_id"]?.stringValue ?? "")
            return "Order A-1042: 2 wool throws, 78.00, delivered 14 August, unopened."
        }

        let session = try await Live.agents.chat(agent: Live.agent, tools: [tool])
        await session.start()
        defer { Task { await session.close() } }

        _ = try await session.responses.create("Look up order A-1042 and tell me what is in it.")

        try await until(30) { await asked.orders.isEmpty == false }

        #expect(await asked.orders.first?.uppercased() == "A-1042")
    }

    /// Asking over HTTP gets an answer the router writes down, without the socket.
    @Test func askingThroughResponsesGetsAnAnswerWrittenDown() async throws {
        let session = try await Live.agents.agent(Live.agent).sessions.create(SessionOptions())
        defer { Task { await session.close() } }

        let turn = try await session.responses.create("What are your opening hours? Answer in one sentence.")
        try await until(30) {
            (try? await session.responses.list().items.first { $0.id == turn.id }?.status) == .completed
        }

        // A guardrail refusing the question is a reply too, and is written down the same way.
        let reply = try await session.responses.items(responseID: turn.id).items
            .filter { $0.kind == .answer || $0.kind == .blocked }.map(\.text).joined()
        #expect(!reply.isEmpty)
    }

    /// A fork at the first turn starts a session of its own and leaves the original as it was.
    @Test func aForkAtAResponseBranchesOffAndLeavesTheOriginal() async throws {
        let agents = Live.agents
        let session = try await agents.chat(agent: Live.agent)
        await session.start()
        defer { Task { await session.close() } }

        _ = try await session.responses.create("My name is Ada. Reply with one word.")
        try await until(20) { session.state == .idle && session.turns.count >= 2 }
        _ = try await session.responses.create("What is my name? Reply with one word.")
        try await until(20) { session.state == .idle && session.turns.count >= 4 }
        try await until(10) { (try? await agents.responses(sessionID: session.id).items.count) == 2 }

        let kept = try #require(try await agents.responses(sessionID: session.id).items.first)
        #expect(kept.said.contains("Ada"))

        let fork = try await agents.fork(sessionID: session.id, ForkOptions(responseID: kept.id))
        #expect(fork.id != session.id)
        #expect(try await agents.responses(sessionID: session.id).items.count == 2)
        try await agents.sessions.delete(fork.id)
        await #expect(throws: AgentsError.self) { try await agents.sessions.get(fork.id) }
    }

    /// A query narrowed to the sessions still running finds the one just opened.
    @Test func aQueryFindsTheLiveSessionItWasNarrowedTo() async throws {
        let session = try await Live.agents.chat(agent: Live.agent)
        defer { Task { await session.close() } }

        var query = SessionQuery(limit: 200)
        query.state = .live
        let page = try await Live.agents.agent(Live.agent).sessions.query(query)

        #expect(page.items.contains { $0.id == session.id })
        #expect(page.items.allSatisfy { $0.state == .live })
    }

    @Test func aRenamedSessionReadsBackItsNewTitle() async throws {
        let session = try await Live.agents.chat(agent: Live.agent)
        defer { Task { await session.close() } }

        try await session.update(title: "Renamed from Swift")

        #expect(session.session.title == "Renamed from Swift")
        #expect(try await Live.agents.sessions.get(session.id).title == "Renamed from Swift")
    }

    /// Runs until the condition holds, so a test waits for what the model does rather than for
    /// a fixed number of seconds.
    private func until(
        _ seconds: Int,
        _ done: () async -> Bool
    ) async throws {
        for _ in 0..<(seconds * 10) {
            if await done() { return }
            try await Task.sleep(for: .milliseconds(100))
        }
        Issue.record("gave up after \(seconds)s")
    }
}

/// What a tool was asked for, collected across the actor boundary the handler runs on.
private actor Asked {
    var orders: [String] = []

    func record(_ order: String) {
        orders.append(order)
    }
}
