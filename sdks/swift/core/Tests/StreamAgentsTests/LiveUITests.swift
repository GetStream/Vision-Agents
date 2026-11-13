import Foundation
import StreamCore
import Testing

@testable import StreamAgents

/// The UI a reply's turn streamed on the socket, shown on its message in Stream Chat until the
/// message's own `ai_ui` settles.
@Suite struct LiveUITests {
    private static func event(_ json: String) throws -> AgentEvent {
        try JSONDecoder().decode(AgentEvent.self, from: Data(json.utf8))
    }

    private static func streaming() throws -> Conversation {
        var conversation = Conversation()
        conversation.apply(try event(#"{"type":"responding","turn_id":"t1"}"#))
        conversation.apply(
            try event(#"{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"root = Card([hdr])\nhdr = CardHeader(\"Kaneo\")"}"#))
        return conversation
    }

    private static let streamedUI: RawJSON = .dictionary([
        "v": .number(1), "format": .string("openui"), "status": .string("streaming"),
        "content": .string("root = Card([hdr])\nhdr = CardHeader(\"Kaneo\")"),
    ])

    @Test func aReplyBeingWrittenShowsTheSocketsUIForItsTurn() throws {
        var live = LiveUI()
        #expect(live.observe(try Self.streaming()))
        #expect(!live.observe(try Self.streaming()))

        let reply: [String: RawJSON] = [
            "ai_generated": .bool(true), "generating": .bool(true),
            "support_runtime": .dictionary(["role": .string("assistant"), "turn_id": .string("t1")]),
        ]
        let overlaid = live.overlay([(id: "question", custom: [:]), (id: "m1", custom: reply)])

        #expect(overlaid.keys.sorted() == ["m1"])
        #expect(overlaid["m1"]?["ai_ui"] == Self.streamedUI)
        #expect(overlaid["m1"]?["generating"] == .bool(true))
    }

    @Test func aReplyWhoseOwnUIIsCompleteShowsItsOwn() throws {
        var live = LiveUI()
        live.observe(try Self.streaming())
        let settled: [String: RawJSON] = [
            "ai_generated": .bool(true), "generating": .bool(false), "turn_id": .string("t1"),
            "ai_ui": .dictionary([
                "v": .number(1), "format": .string("openui"), "status": .string("complete"),
                "content": .string("root = Card([hdr])\nhdr = CardHeader(\"Kaneo Cliffside Kitchen\")"),
            ]),
        ]

        #expect(live.overlay([(id: "m1", custom: settled)]).isEmpty)
    }

    @Test func aReplyStillStreamingItsOwnUIShowsTheSockets() throws {
        var live = LiveUI()
        live.observe(try Self.streaming())
        let behind: [String: RawJSON] = [
            "ai_generated": .bool(true), "generating": .bool(true), "turn_id": .string("t1"),
            "ai_ui": .dictionary([
                "v": .number(1), "format": .string("openui"), "status": .string("streaming"),
                "content": .string("root = Card([hdr])"),
            ]),
        ]

        #expect(live.overlay([(id: "m1", custom: behind)])["m1"]?["ai_ui"] == Self.streamedUI)
    }

    @Test func theNewestReplyBeingWrittenTakesTheLatestUIBeforeItsTurnIsKnown() throws {
        var live = LiveUI()
        live.observe(try Self.streaming())
        let earlier: [String: RawJSON] = ["ai_generated": .bool(true), "generating": .bool(false)]
        let newest: [String: RawJSON] = ["ai_generated": .bool(true), "generating": .bool(true)]

        let overlaid = live.overlay([(id: "m0", custom: earlier), (id: "m1", custom: newest)])

        #expect(overlaid.keys.sorted() == ["m1"])
        #expect(live.latest?.content == "root = Card([hdr])\nhdr = CardHeader(\"Kaneo\")")
    }

    @Test func aReplyWhoseUIWasTooLargeForItsMessageShowsTheSockets() throws {
        var live = LiveUI()
        live.observe(try Self.streaming())
        let spilled: [String: RawJSON] = [
            "ai_generated": .bool(true), "turn_id": .string("t1"),
            "ai_ui": .dictionary([
                "v": .number(1), "format": .string("openui"), "status": .string("complete"),
                "overflow": .bool(true), "artifact_id": .string("a1"),
            ]),
        ]

        #expect(live.overlay([(id: "m1", custom: spilled)])["m1"]?["ai_ui"] == Self.streamedUI)
    }

    @Test func thePersonsMessagesAreLeftAlone() throws {
        var live = LiveUI()
        live.observe(try Self.streaming())

        #expect(live.overlay([(id: "q1", custom: ["turn_id": .string("t1"), "generating": .bool(true)])]).isEmpty)
    }
}
