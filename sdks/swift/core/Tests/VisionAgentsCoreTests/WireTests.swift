import Foundation
import Testing

@testable import VisionAgentsCore

/// The frames in both directions, checked against what the router actually writes and reads.
@Suite struct WireTests {
    @Test func anUnknownEventKeepsItsNameAndItsFields() throws {
        let event = try JSONDecoder().decode(
            AgentEvent.self,
            from: Data(#"{"type":"overheard","who":"nobody","confidence":0.5}"#.utf8))

        #expect(event.type == "overheard")
        #expect(event.kind == nil)
        #expect(event["who"].stringValue == "nobody")
        #expect(event["confidence"].doubleValue == 0.5)
    }

    @Test func anEventWithoutATypeIsRefused() {
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(AgentEvent.self, from: Data(#"{"text":"hi"}"#.utf8))
        }
    }

    @Test func aMissingFieldReadsAsEmptyRatherThanFailing() throws {
        let event = try JSONDecoder().decode(
            AgentEvent.self, from: Data(#"{"type":"heard"}"#.utf8))

        #expect(event.text == "")
        #expect(event.turnID == "")
        #expect(event.participant == nil)
        #expect(event["nothing"].intValue == nil)
    }

    @Test func aToolCallArrivesWithItsArgumentsDecoded() throws {
        let event = try JSONDecoder().decode(
            AgentEvent.self,
            from: Data(
                #"{"type":"tool_call","id":"c1","name":"lookup_order","arguments":"{\"order_id\":\"A-1\",\"full\":true}"}"#
                    .utf8))

        let call = try #require(event.toolCall)
        #expect(call.id == "c1")
        #expect(call.name == "lookup_order")
        #expect(call.argumentValues["order_id"]?.stringValue == "A-1")
        #expect(call.argumentValues["full"]?.boolValue == true)
    }

    @Test func aToolCallWhoseArgumentsAreNotAnObjectYieldsNoArguments() throws {
        let event = try JSONDecoder().decode(
            AgentEvent.self,
            from: Data(#"{"type":"tool_call","id":"c1","name":"now","arguments":"oops"}"#.utf8))

        #expect(try #require(event.toolCall).argumentValues.isEmpty)
    }

    @Test func onlyToolCallEventsAreToolCalls() throws {
        let event = try JSONDecoder().decode(
            AgentEvent.self,
            from: Data(#"{"type":"tool_ran","tool":"lookup_order","result":"A-1"}"#.utf8))

        #expect(event.toolCall == nil)
    }

    /// The router reads one flat struct, so a command has to put its fields at the top level
    /// under the names in `readCommands`.
    @Test(arguments: [
        (Command.respond("hi"), #"{"text":"hi","type":"respond"}"#),
        (Command.say("welcome"), #"{"text":"welcome","type":"say"}"#),
        (Command.interrupt, #"{"type":"interrupt"}"#),
        (Command.instructions("be brief"), #"{"instructions":"be brief","type":"instructions"}"#),
        (Command.close, #"{"type":"close"}"#),
        (
            Command.toolResult(id: "c1", output: "20 degrees", error: nil),
            #"{"error":"","output":"20 degrees","tool_call_id":"c1","type":"tool_result"}"#
        ),
        (
            Command.toolResult(id: "c1", output: nil, error: "no such order"),
            #"{"error":"no such order","output":"","tool_call_id":"c1","type":"tool_result"}"#
        ),
    ])
    func aCommandEncodesToWhatTheRouterReads(command: Command, expected: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys

        #expect(String(decoding: try encoder.encode(command), as: UTF8.self) == expected)
    }

    @Test func aToolSchemaSaysWhichArgumentsAreRequired() throws {
        let schema = JSONValue.strings(
            ["order_id": "the order number"], required: ["order_id"])

        #expect(schema.objectValue["type"]?.stringValue == "object")
        #expect(schema.objectValue["required"]?.arrayValue.map(\.stringValue) == ["order_id"])
        let properties = schema.objectValue["properties"]?.objectValue ?? [:]
        #expect(properties["order_id"]?.objectValue["description"]?.stringValue == "the order number")
    }
}

@Suite struct BackendTests {
    private let backend = Backend(
        url: URL(string: "http://localhost:8080")!, customerID: "acme")

    @Test func everyRequestSaysItComesFromADevice() {
        // Without this the router treats a caller with no proxy in front of it as a backend,
        // which would hand a phone the paths that rewrite an agent.
        #expect(backend.headers["Stream-Auth-Type"] == "jwt")
        #expect(backend.headers["X-Customer-Id"] == "acme")
    }

    @Test func aSocketURLSwitchesSchemeAndCarriesTheCustomer() {
        let url = backend.socketURL(
            path: "/v1/agents/sessions/s1/events", query: ["decisions": "false"])

        #expect(url.absoluteString == "ws://localhost:8080/v1/agents/sessions/s1/events?customer_id=acme&decisions=false")
    }

    @Test func aSecureRouterGetsASecureSocket() {
        let secure = Backend(url: URL(string: "https://router.example.com")!, customerID: "acme")

        #expect(secure.socketURL(path: "/v1/agents/sessions/s1/events").scheme == "wss")
    }

    /// Go writes as many fractional digits as the value needs, and none when it needs none,
    /// so both of these come off the same endpoint on consecutive calls.
    @Test(arguments: [
        "2026-09-02T18:46:50.89279Z",
        "2026-09-02T18:46:50.123456789Z",
        "2026-09-02T18:46:50Z",
        "2026-09-02T12:46:50-06:00",
    ])
    func aTimestampTheRouterWritesIsRead(text: String) throws {
        #expect(try RouterDates().decode(text).timeIntervalSince1970 > 0)
    }

    @Test func somethingThatIsNotATimestampIsRefused() {
        #expect(throws: AgentsError.self) { try RouterDates().decode("last Tuesday") }
    }

    @Test func a403IsRecognisedAsAServerSideOnlyPath() {
        #expect(AgentsError.http(status: 403, message: "server-side only").isServerSideOnly)
        #expect(!AgentsError.http(status: 404, message: "no such session").isServerSideOnly)
    }
}
