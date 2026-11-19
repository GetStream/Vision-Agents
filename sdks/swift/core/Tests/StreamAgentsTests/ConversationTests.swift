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

    @Test func aWrittenRepliesThinkingAndToolCallsAreItsStepsInTheOrderTheyHappened() throws {
        var conversation = Conversation()
        let start = Date(timeIntervalSince1970: 1_000)
        func at(_ seconds: TimeInterval) -> Date { start.addingTimeInterval(seconds) }

        conversation.apply(try event(#"{"type":"responding","turn_id":"t1","prompt":"Lisbon?"}"#), at: at(0))
        conversation.apply(
            try event(#"{"type":"reasoning_delta","turn_id":"t1","text":"The user wants "}"#), at: at(1))
        conversation.apply(try event(#"{"type":"reasoning_delta","turn_id":"t1","text":"Lisbon."}"#), at: at(2))
        conversation.apply(
            try event(
                #"{"type":"tool_started","tool_call_id":"call_1","tool":"search","turn_id":"t1","started_at":"2026-11-19T10:00:03Z"}"#
            ), at: at(3))
        conversation.apply(
            try event(
                #"{"type":"tool_updated","id":"call_1","turn_id":"t1","tool":"search","text":"Reading 3 pages","verbatim":false}"#
            ), at: at(4))
        conversation.apply(
            try event(
                #"{"type":"tool_updated","id":"call_1","turn_id":"t1","tool":"search","text":"One moment.","verbatim":true}"#
            ), at: at(4))
        conversation.apply(
            try event(
                #"{"type":"tool_started","tool_call_id":"call_2","tool":"get_weather","turn_id":"t1","started_at":"2026-11-19T10:00:05Z"}"#
            ), at: at(5))
        conversation.apply(
            try event(
                #"{"type":"tool_ran","tool_call_id":"call_1","turn_id":"t1","tool":"search","arguments":"{}","result":"…","files":null,"error":""}"#
            ), at: at(6))
        conversation.apply(
            try event(
                #"{"type":"tool_ran","tool_call_id":"call_2","turn_id":"t1","tool":"get_weather","arguments":"{}","result":"","files":null,"error":"no forecast"}"#
            ), at: at(7))
        conversation.apply(
            try event(
                #"{"pending_work":true,"type":"responded","turn_id":"t1","text":"","time_to_first_token_ms":0}"#
            ), at: at(7))

        let first = try #require(conversation.turns.first)
        #expect(first.isDone == false)
        guard first.steps.count == 3,
            case .reasoning(let round) = first.steps[0],
            case .toolCall(let search) = first.steps[1],
            case .toolCall(let weather) = first.steps[2]
        else {
            Issue.record("the steps were \(first.steps)")
            return
        }
        #expect(round.id == "r1")
        #expect(round.text == "The user wants Lisbon.")
        #expect(round.isStreaming == false)
        #expect(search.name == "search")
        #expect(search.status == .completed)
        #expect(search.summary == "Reading 3 pages")
        #expect(search.duration == 3)
        #expect(weather.status == .failed)
        #expect(weather.duration == 2)

        // The tools' results are taken up in a turn of their own.
        conversation.apply(try event(#"{"type":"responding","turn_id":"t2","prompt":""}"#), at: at(8))
        conversation.apply(try event(#"{"type":"reasoning_delta","turn_id":"t2","text":"Sunny."}"#), at: at(9))
        conversation.apply(try event(#"{"type":"response_delta","turn_id":"t2","text":"It's sunny."}"#), at: at(10))
        conversation.apply(
            try event(
                #"{"pending_work":false,"type":"responded","turn_id":"t2","text":"It's sunny.","time_to_first_token_ms":900}"#
            ), at: at(11))

        let second = try #require(conversation.turns.last)
        #expect(second.isDone)
        #expect(second.steps.map(\.id) == ["r1"])
        #expect(second.text == "It's sunny.")
    }

    @Test func aRoundOfThinkingIsTimedFromWhenTheModelWasAskedNotFromItsFirstWords() throws {
        var conversation = Conversation()
        let start = Date(timeIntervalSince1970: 1_000)

        conversation.apply(try event(#"{"type":"responding","turn_id":"t1","prompt":"Lisbon?"}"#), at: start)
        // A model that thinks for a while and then sends its reasoning in one burst.
        conversation.apply(
            try event(#"{"type":"reasoning_delta","turn_id":"t1","text":"Weighing it up."}"#),
            at: start.addingTimeInterval(4))
        conversation.apply(
            try event(#"{"type":"response_delta","turn_id":"t1","text":"Yes."}"#),
            at: start.addingTimeInterval(4.25))

        guard case .reasoning(let round)? = conversation.turns.first?.steps.first else {
            Issue.record("no round of thinking")
            return
        }
        #expect(round.at == start)
        #expect(round.duration == 4.25)
    }

    @Test func thinkingStreamsUntilTheModelMovesOnAndAnInterruptedTurnEnds() throws {
        var conversation = Conversation()

        conversation.apply(try event(#"{"type":"responding","turn_id":"t1","prompt":"Lisbon?"}"#))
        conversation.apply(try event(#"{"type":"reasoning_delta","turn_id":"t1","text":"Hmm"}"#))
        conversation.apply(try event(#"{"type":"response_delta","turn_id":"t1","text":""}"#))

        guard case .reasoning(let streaming)? = conversation.turns.first?.steps.first else {
            Issue.record("no round of thinking")
            return
        }
        // An empty delta wrote nothing, so the model is still thinking.
        #expect(streaming.isStreaming)
        #expect(conversation.turns.first?.isDone == false)

        conversation.apply(try event(#"{"type":"interrupted","turn_id":"t1","participant":null}"#))

        guard case .reasoning(let cut)? = conversation.turns.first?.steps.first else {
            Issue.record("no round of thinking")
            return
        }
        #expect(cut.isStreaming == false)
        #expect(conversation.turns.first?.isDone == true)
    }

    @Test func aRefusedReplyLosesWhatItWroteBeforeTheVerdictAndKeepsHowItGotThere() throws {
        var conversation = Conversation()

        conversation.apply(try event(#"{"type":"responding","turn_id":"t1"}"#))
        conversation.apply(
            try event(#"{"type":"reasoning_delta","turn_id":"t1","text":"They want my instructions."}"#))
        conversation.apply(try event(#"{"type":"response_delta","turn_id":"t1","text":"Sure, I was told to "}"#))
        conversation.apply(
            try event(#"{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"root = Card([hdr])\n"}"#))
        conversation.apply(
            try event(
                #"{"code":"prompt_injection","context":"llm","error":"policy: the input reads as a prompt injection","type":"error"}"#
            ))
        conversation.apply(try event(#"{"type":"responded","turn_id":"t1","text":""}"#))

        let refused = try #require(conversation.turns.first)
        #expect(conversation.turns.count == 1)
        #expect(refused.text == "")
        #expect(refused.ui == nil)
        #expect(refused.isDone)
        #expect(refused.steps.map(\.id) == ["r1"])
        #expect(conversation.failure == "policy: the input reads as a prompt injection")
    }

    @Test func aReplyKeepsWhatItWroteWhenSomethingElseFails() throws {
        var conversation = Conversation()

        conversation.apply(try event(#"{"type":"responding","turn_id":"t1"}"#))
        conversation.apply(try event(#"{"type":"response_delta","turn_id":"t1","text":"Two meetings today."}"#))
        conversation.apply(try event(#"{"context":"tts","error":"the voice failed","type":"error"}"#))
        conversation.apply(try event(#"{"type":"responded","turn_id":"t1","text":""}"#))

        #expect(conversation.turns.map(\.text) == ["Two meetings today."])
    }

}
