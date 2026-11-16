import Foundation
import Testing

@testable import StreamAgents

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

    @Test func aRepliesUIStreamsOntoItsTurnBesideItsText() throws {
        var conversation = Conversation()
        conversation.apply(try event(#"{"type":"responding","turn_id":"t1"}"#))
        conversation.apply(try event(#"{"type":"response_delta","turn_id":"t1","text":"Table at 19:15."}"#))

        conversation.apply(
            try event(#"{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"root = Card([hdr])\n"}"#))
        conversation.apply(
            try event(#"{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"hdr = CardHeader(\"Kan"}"#))

        #expect(conversation.turns.count == 1)
        #expect(conversation.turns[0].text == "Table at 19:15.")
        #expect(conversation.turns[0].ui?.isStreaming == true)
        // The header shows once it is finished, not a letter at a time.
        #expect(conversation.turns[0].ui?.content == "root = Card([hdr])")

        conversation.apply(
            try event(
                #"{"type":"ui_completed","turn_id":"t1","format":"openui","content":"root = Card([hdr])\nhdr = CardHeader(\"Kaneo\")"}"#
            ))
        conversation.apply(try event(#"{"type":"responded","turn_id":"t1","text":"Table at 19:15."}"#))

        #expect(conversation.turns[0].ui?.isStreaming == false)
        #expect(conversation.turns[0].ui?.content == "root = Card([hdr])\nhdr = CardHeader(\"Kaneo\")")
        #expect(conversation.turns[0].text == "Table at 19:15.")
        #expect(conversation.state == .idle)
    }

    @Test func aValueShowsOnceItsLineEnds() throws {
        var conversation = Conversation()
        conversation.apply(
            try event(#"{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"root = Card([gauge])\ngauge = AvailabilityGauge(8"}"#))
        conversation.apply(
            try event(#"{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"6, \"12 min wait\")\n$party = 4"}"#))

        #expect(
            conversation.turns[0].ui?.content == "root = Card([gauge])\ngauge = AvailabilityGauge(86, \"12 min wait\")")

        conversation.apply(try event(#"{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"2\n"}"#))

        #expect(conversation.turns[0].ui?.content.hasSuffix("\n$party = 42") == true)
    }

    @Test func aStatementOverSeveralLinesShowsOnceItCloses() throws {
        var conversation = Conversation()
        conversation.apply(
            try event(
                #"{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"root = Card([map])\nmap = TripMap([\n  {name: \"Kaneo\"},\n"}"#
            ))

        #expect(conversation.turns[0].ui?.content == "root = Card([map])")

        conversation.apply(try event(#"{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"])\n"}"#))

        #expect(conversation.turns[0].ui?.content == "root = Card([map])\nmap = TripMap([\n  {name: \"Kaneo\"},\n])")
    }

    @Test func aCompletedUIWithNoContentKeepsWhatStreamed() throws {
        var conversation = Conversation()
        conversation.apply(
            try event(#"{"type":"ui_delta","turn_id":"t1","format":"a2ui","version":"1.0","delta":"[{\"version\":\"v1.0\"}]"}"#))

        conversation.apply(try event(#"{"type":"ui_completed","turn_id":"t1","format":"a2ui","version":"1.0","content":""}"#))

        #expect(conversation.turns.map(\.speaker) == [.agent])
        #expect(conversation.turns[0].ui == TurnUI(
            format: "a2ui", version: "1.0", isStreaming: false, content: #"[{"version":"v1.0"}]"#))
    }

    @Test func anOpenUIPatchIsMergedOntoTheCardBeforeIt() throws {
        var conversation = Conversation()
        conversation.apply(try event(#"{"type":"responding","turn_id":"t1"}"#))
        conversation.apply(
            try event(
                #"{"type":"ui_completed","turn_id":"t1","format":"openui","content":"$party = 2\nroot = Card([hdr, gauge])\nhdr = CardHeader(\"Kaneo\")\ngauge = AvailabilityGauge(86, \"12 min wait\")"}"#
            ))
        conversation.apply(try event(#"{"type":"responding","turn_id":"t2"}"#))

        conversation.apply(
            try event(#"{"type":"ui_delta","turn_id":"t2","format":"openui","mode":"patch","delta":"$party = 4\n"}"#))
        #expect(conversation.turns[1].ui?.isStreaming == true)
        #expect(
            conversation.turns[1].ui?.content
                == "$party = 4\nroot = Card([hdr, gauge])\nhdr = CardHeader(\"Kaneo\")\ngauge = AvailabilityGauge(86, \"12 min wait\")")

        conversation.apply(
            try event(
                #"{"type":"ui_completed","turn_id":"t2","format":"openui","mode":"patch","content":"$party = 4\ngauge = null\ntip = Callout(\"info\", \"Book\", \"Ask for the terrace.\")"}"#
            ))

        #expect(
            conversation.turns[1].ui?.content
                == "$party = 4\nroot = Card([hdr, gauge])\nhdr = CardHeader(\"Kaneo\")\ntip = Callout(\"info\", \"Book\", \"Ask for the terrace.\")")
        // The card it was merged onto stays as it was.
        #expect(conversation.turns[0].ui?.content.contains("AvailabilityGauge") == true)
    }

    @Test func aPatchWithNoCardBeforeItSetsNoUI() throws {
        var conversation = Conversation()
        conversation.apply(try event(#"{"type":"responding","turn_id":"t1"}"#))

        // As after a reload: the card it changes is only in Stream Chat, which shows it merged.
        conversation.apply(
            try event(#"{"type":"ui_delta","turn_id":"t1","format":"openui","mode":"patch","delta":"$party = 4\n"}"#))
        conversation.apply(
            try event(#"{"type":"ui_completed","turn_id":"t1","format":"openui","mode":"patch","content":"$party = 4"}"#))

        #expect(conversation.turns.map(\.ui) == [nil])
    }

    @Test func aPatchDeltaThatShowsNothingNewIsStillKept() throws {
        var conversation = Conversation()
        conversation.apply(
            try event(#"{"type":"ui_completed","turn_id":"t1","format":"openui","content":"$party = 2\nroot = Card([hdr])"}"#))
        conversation.apply(try event(#"{"type":"responding","turn_id":"t2"}"#))

        conversation.apply(try event(#"{"type":"ui_delta","turn_id":"t2","format":"openui","mode":"patch","delta":"$party"}"#))
        #expect(conversation.turns[1].ui?.content == "$party = 2\nroot = Card([hdr])")
        conversation.apply(try event(#"{"type":"ui_delta","turn_id":"t2","format":"openui","mode":"patch","delta":" = 4\n"}"#))

        #expect(conversation.turns[1].ui?.content == "$party = 4\nroot = Card([hdr])")
    }

    @Test func aStatementOverSeveralLinesIsReplacedWhole() {
        let base = """
            root = Card([map, tip])
            map = TripMap([
              {name: "Kaneo", latitude: 41.11, longitude: 20.78},
              {name: "St. Naum", latitude: 40.91, longitude: 20.74}
            ])
            // the tip
            tip = Callout("info", "Note", "Closed (Mondays) = no boats")
            """

        let merged = TurnUI.merge(base, patch: #"map = TripMap([{name: "Bay of Bones", latitude: 41.01, longitude: 20.79}])"#)

        #expect(
            merged == """
                root = Card([map, tip])
                map = TripMap([{name: "Bay of Bones", latitude: 41.01, longitude: 20.79}])
                tip = Callout("info", "Note", "Closed (Mondays) = no boats")
                """)
    }

    @Test func aVariableSetToNullKeepsItsNull() {
        let merged = TurnUI.merge("$open = true\nroot = Card([hdr])", patch: "$open = null")

        #expect(merged == "$open = null\nroot = Card([hdr])")
    }
}
