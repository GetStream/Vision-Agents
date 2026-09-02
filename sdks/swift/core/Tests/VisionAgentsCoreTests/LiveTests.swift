import Foundation
import Testing

@testable import VisionAgentsCore

/// Tests that need a router running.
///
/// The Swift answer to `@pytest.mark.integration`: they are skipped unless
/// `VISION_AGENTS_URL` is set, so the ordinary `swift test` stays offline and fast.
///
///     VISION_AGENTS_URL=http://localhost:8080 VISION_AGENTS_CUSTOMER_ID=examples swift test
struct Live {
    static let url = ProcessInfo.processInfo.environment["VISION_AGENTS_URL"]
    static let customerID =
        ProcessInfo.processInfo.environment["VISION_AGENTS_CUSTOMER_ID"] ?? "acme"
    static let agent = ProcessInfo.processInfo.environment["VISION_AGENTS_AGENT"] ?? "swift_demo"

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
    @Test func theAgentTheGoExampleConfiguredIsThere() async throws {
        let config = try await Live.agents.agentConfig(named: Live.agent)

        #expect(!config.id.isEmpty)
        #expect(!config.instructions.isEmpty, "run `go run ./configure` first")
    }

    @Test func aNameThatIsNotAnAgentIsReportedAsOne() async {
        await #expect(throws: AgentsError.self) {
            try await Live.agents.agentConfig(named: "no-such-agent")
        }
    }

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

        try await session.send("What are your opening hours? Answer in one sentence.")

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

        try await session.send("Look up order A-1042 and tell me what is in it.")

        try await until(30) { await asked.orders.isEmpty == false }

        #expect(await asked.orders.first?.uppercased() == "A-1042")
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
