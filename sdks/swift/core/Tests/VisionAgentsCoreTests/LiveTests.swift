import AGUI
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

    /// A tool that needs allowing is not run on the model's word: the session asks, and holds
    /// the call until somebody answers.
    ///
    /// Both tools are declared, because the agent hands the money decision to a skill and the
    /// skill decides out of the order: given no way to look one up it asks the caller what the
    /// item cost and what condition it was in, and never reaches the refund at all.
    @Test func aToolThatNeedsApprovalWaitsForOne() async throws {
        let asked = Asked()
        let lookup = AgentTool(
            name: "lookup_order",
            description: "Look up one of the caller's orders by its order number.",
            parameters: .strings(["order_id": "the order number"], required: ["order_id"])
        ) { _ in
            "Order A-1042: 2 wool throws, 78.00, paid by card ending 4242, "
                + "delivered 6 days ago, unopened."
        }
        let tool = AgentTool(
            name: "refund_order",
            description: "Refund an order the caller is owed money for.",
            parameters: .strings(
                ["order_id": "the order number", "amount": "how much to refund"],
                required: ["order_id", "amount"]),
            approval: { arguments in
                "Refund order \(arguments["order_id"]?.stringValue ?? "")?"
            }
        ) { arguments in
            await asked.record(arguments["order_id"]?.stringValue ?? "")
            return "Refunded to the original card."
        }

        let session = try await Live.agents.chat(agent: Live.agent, tools: [lookup, tool])
        await session.start()
        defer { Task { await session.close() } }

        try await session.send("I want a refund for order A-1042. It is unopened.")

        try await until(60) { session.pendingApprovals.isEmpty == false }
        let approval = try #require(session.pendingApprovals.first)
        #expect(approval.toolCallId == approval.id)
        #expect(approval.message?.contains("Refund order") == true)
        #expect(await asked.orders.isEmpty, "nothing runs before somebody says so")

        try await session.approve(approval)

        try await until(10) { await asked.orders.isEmpty == false }
        #expect(session.pendingApprovals.isEmpty)
    }

    /// The same conversation, in the protocol's own terms: a run that opens on the question and
    /// finishes on the answer.
    @Test func theSameConversationArrivesAsAGUIEvents() async throws {
        let session = try await Live.agents.chat(agent: Live.agent)
        await session.start()
        defer { Task { await session.close() } }

        let seen = Seen()
        let events = session.aguiEvents()
        let watching = Task {
            for await event in events {
                await seen.record(event)
            }
        }
        defer { watching.cancel() }

        try await session.send("What are your opening hours? Answer in one sentence.")

        try await until(30) { await seen.types.contains(.runFinished) }

        #expect(await seen.types.first == .runStarted)
        #expect(await seen.types.contains(.textMessageContent))
        #expect(await seen.threads == [session.id], "the session is the thread")
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

/// The protocol events a session published, in the order they arrived.
private actor Seen {
    var types: [EventType] = []
    var threads: Set<String> = []

    func record(_ event: AGUI.Event) {
        types.append(event.type)
        if case .runStarted(let started) = event {
            threads.insert(started.threadId)
        }
    }
}
