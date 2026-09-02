import Foundation
import Testing

@testable import VisionAgentsCore

/// Frames exactly as `frameOf` in the router writes them, so that a change to the wire format
/// on that side fails here rather than in somebody's app.
private func event(_ json: String) throws -> AgentEvent {
    try JSONDecoder().decode(AgentEvent.self, from: Data(json.utf8))
}

@Suite struct ConversationTests {
    @Test func aReplyArrivesOneDeltaAtATime() throws {
        var conversation = Conversation()

        conversation.apply(try event(#"{"type":"responding","turn_id":"t1","prompt":"hi"}"#))
        #expect(conversation.state == .responding)
        #expect(conversation.turns.count == 1)

        conversation.apply(try event(#"{"type":"response_delta","turn_id":"t1","text":"Hel"}"#))
        conversation.apply(try event(#"{"type":"response_delta","turn_id":"t1","text":"lo"}"#))
        #expect(conversation.turns.last?.text == "Hello")

        conversation.apply(
            try event(
                #"{"type":"responded","turn_id":"t1","text":"Hello there.","time_to_first_token_ms":90}"#
            ))
        #expect(conversation.turns.count == 1)
        #expect(conversation.turns.last?.text == "Hello there.")
        #expect(conversation.turns.last?.speaker == .agent)
        #expect(conversation.state == .idle)
    }

    @Test func aDeltaForATurnNobodySawBeginStillLands() throws {
        var conversation = Conversation()

        conversation.apply(try event(#"{"type":"response_delta","turn_id":"t9","text":"mid"}"#))

        #expect(conversation.turns.map(\.text) == ["mid"])
        #expect(conversation.turns.last?.speaker == .agent)
    }

    @Test func aSpokenTurnWithNoFinalTextKeepsWhatWasStreamed() throws {
        var conversation = Conversation()

        conversation.apply(try event(#"{"type":"responding","turn_id":"t1"}"#))
        conversation.apply(try event(#"{"type":"response_delta","turn_id":"t1","text":"Sure."}"#))
        conversation.apply(try event(#"{"type":"responded","turn_id":"t1","text":""}"#))

        #expect(conversation.turns.map(\.text) == ["Sure."])
    }

    @Test func whatWasHeardOnACallBecomesAParticipantTurn() throws {
        var conversation = Conversation()

        conversation.apply(
            try event(
                #"{"type":"heard","participant":{"id":"p1","user_id":"u1","name":"Alice"},"text":"what are your hours","language":"en"}"#
            ))

        #expect(conversation.turns.count == 1)
        #expect(conversation.turns[0].text == "what are your hours")
        if case .participant(let who) = conversation.turns[0].speaker {
            #expect(who?.display == "Alice")
        } else {
            Issue.record("heard should be a participant turn")
        }
    }

    @Test func typingSomethingIsNotShownTwiceWhenTheRouterEchoesIt() throws {
        var conversation = Conversation()

        conversation.said("what are your hours")
        conversation.apply(
            try event(#"{"type":"heard","participant":{},"text":"what are your hours"}"#))

        #expect(conversation.turns.count == 1)
    }

    @Test func aDelegatedSkillIsNamedWhileItRuns() throws {
        var conversation = Conversation()

        conversation.apply(
            try event(#"{"type":"delegated","task_id":"k1","skill":"lookup_order"}"#))
        #expect(conversation.state == .working(["lookup_order"]))

        conversation.apply(
            try event(#"{"type":"task_settled","task_id":"k1","skill":"lookup_order","text":"done"}"#))
        #expect(conversation.state == .responding)
    }

    @Test func twoSkillsAtOnceBothShowUntilBothSettle() throws {
        var conversation = Conversation()

        conversation.apply(try event(#"{"type":"delegated","task_id":"k1","skill":"think"}"#))
        conversation.apply(try event(#"{"type":"delegated","task_id":"k2","skill":"recall"}"#))
        #expect(conversation.state == .working(["think", "recall"]))

        conversation.apply(try event(#"{"type":"task_settled","task_id":"k1","skill":"think"}"#))
        #expect(conversation.state == .working(["recall"]))
    }

    @Test func theConversationEndsWhenTheAgentLeaves() throws {
        var conversation = Conversation()

        conversation.apply(try event(#"{"type":"left","at":"2026-09-02T17:05:00Z"}"#))

        #expect(conversation.state == .ended)
    }

    @Test func aReportedFailureIsKeptRatherThanThrown() throws {
        var conversation = Conversation()

        conversation.apply(
            try event(#"{"type":"error","context":"tts","error":"the voice is unknown"}"#))

        #expect(conversation.failure == "the voice is unknown")
    }

    @Test func anEventThisSDKHasNeverHeardOfChangesNothing() throws {
        var conversation = Conversation()
        conversation.apply(try event(#"{"type":"responding","turn_id":"t1"}"#))
        let before = conversation

        conversation.apply(try event(#"{"type":"astonished","turn_id":"t1","degree":11}"#))

        #expect(conversation == before)
    }
}
