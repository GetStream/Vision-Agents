import AGUI
import Foundation
import Testing

@testable import VisionAgentsCore

/// Frames exactly as `frameOf` in the router writes them, so that a change to the wire format
/// on that side fails here rather than in somebody's app.
private func frame(_ json: String) throws -> AgentEvent {
    try JSONDecoder().decode(AgentEvent.self, from: Data(json.utf8))
}

/// One exchange as the router publishes it: a question, a reply written a delta at a time, and
/// then a tool the model wanted, whose answer it reads in a turn of its own.
private let exchange = [
    #"{"type":"heard","participant":{"id":"p1","user_id":"u1","name":"Alice"},"text":"refund order A-1042","language":"en"}"#,
    #"{"type":"responding","turn_id":"t1","participant":{"id":"p1","user_id":"u1","name":"Alice"},"prompt":"refund order A-1042"}"#,
    #"{"type":"response_delta","turn_id":"t1","text":"Let me "}"#,
    #"{"type":"response_delta","turn_id":"t1","text":"check."}"#,
    #"{"type":"responded","turn_id":"t1","text":"Let me check.","time_to_first_token_ms":90}"#,
    #"{"type":"tool_call","id":"c1","name":"lookup_order","arguments":"{\"order_id\":\"A-1042\"}"}"#,
]

@Suite struct AGUITranslatorTests {
    /// What a whole exchange means in the protocol: one run, the person's message, the agent's
    /// message as it streams, and the tool the turn ended by asking for.
    @Test func anExchangeBecomesRunsWithMessagesInThem() throws {
        var translator = AGUITranslator(threadID: "s1")

        let events = try exchange.flatMap { translator.translate(try frame($0)) }

        #expect(
            events.map(\.type) == [
                .runStarted,
                .textMessageStart, .textMessageContent, .textMessageEnd,
                .textMessageStart, .textMessageContent, .textMessageContent, .textMessageEnd,
                .runFinished,
                .runStarted,
                .toolCallStart, .toolCallArgs, .toolCallEnd,
            ])

        guard case .runStarted(let started) = events.first else {
            Issue.record("the stream has to open a run")
            return
        }
        #expect(started.threadId == "s1", "the session is the thread")
        #expect(!started.runId.isEmpty)
    }

    /// The agent's reply is one message that grows, not a message per delta.
    @Test func theReplyIsOneMessageWithTheDeltasAsItsContent() throws {
        var translator = AGUITranslator(threadID: "s1")

        let events = try exchange.flatMap { translator.translate(try frame($0)) }
        var conversation = ConversationState()
        for event in events {
            conversation.apply(event)
        }

        #expect(conversation.messages.count == 2)
        #expect(conversation.messages.first?.role == .user)
        #expect(conversation.messages.first?.textContent == "refund order A-1042")
        #expect(conversation.messages.last?.role == .assistant)
        #expect(conversation.messages.last?.textContent == "Let me check.")
        #expect(conversation.messages.last?.toolCalls.map(\.name) == ["lookup_order"])
    }

    /// A reply that never streamed still arrives, which is what a turn the agent only spoke
    /// looks like on a call.
    @Test func aReplyThatArrivedWholeIsStillAMessage() throws {
        var translator = AGUITranslator(threadID: "s1")

        var events = try translator.translate(frame(#"{"type":"responding","turn_id":"t1"}"#))
        events += try translator.translate(
            frame(#"{"type":"responded","turn_id":"t1","text":"We are open every day."}"#))

        guard case .runStarted(let started) = events.first else {
            Issue.record("a turn the agent started for itself opens a run")
            return
        }
        #expect(started.runId == "t1", "a run a turn opened is named after it")

        var conversation = ConversationState()
        for event in events {
            conversation.apply(event)
        }
        #expect(conversation.messages.map(\.textContent) == ["We are open every day."])
    }

    /// The final text a turn reports is the authoritative one, so a tail the deltas never
    /// carried is published rather than lost.
    @Test func aReplyEndingWithMoreThanStreamedPublishesTheRest() throws {
        var translator = AGUITranslator(threadID: "s1")

        var events = try translator.translate(frame(#"{"type":"responding","turn_id":"t1"}"#))
        events += try translator.translate(
            frame(#"{"type":"response_delta","turn_id":"t1","text":"Hel"}"#))
        events += try translator.translate(
            frame(#"{"type":"responded","turn_id":"t1","text":"Hello there."}"#))

        var conversation = ConversationState()
        for event in events {
            conversation.apply(event)
        }
        #expect(conversation.messages.map(\.textContent) == ["Hello there."])
    }

    /// A final text that is not what streamed is left as it streamed: the message is what
    /// somebody has already read, and AG-UI has no way to correct one.
    @Test func aReplyEndingWithSomethingElseKeepsWhatStreamed() throws {
        var translator = AGUITranslator(threadID: "s1")

        var events = try translator.translate(frame(#"{"type":"responding","turn_id":"t1"}"#))
        events += try translator.translate(
            frame(#"{"type":"response_delta","turn_id":"t1","text":"Sure."}"#))
        events += try translator.translate(
            frame(#"{"type":"responded","turn_id":"t1","text":"Something else."}"#))

        var conversation = ConversationState()
        for event in events {
            conversation.apply(event)
        }
        #expect(conversation.messages.map(\.textContent) == ["Sure."])
    }

    /// The result of a tool this device ran, which the router never sends back to it, lands in
    /// the run the call was made in.
    @Test func whatAToolOnThisSideReturnedBecomesTheCallsResult() throws {
        var translator = AGUITranslator(threadID: "s1")
        for json in exchange {
            _ = try translator.translate(frame(json))
        }

        let answered = translator.answered("c1", with: "Order A-1042: 2 wool throws, 78.00.")

        #expect(answered.map(\.type) == [.toolCallResult])
        guard case .toolCallResult(let result) = answered.first else {
            Issue.record("a tool answered is a result")
            return
        }
        #expect(result.toolCallId == "c1")
        #expect(result.content == "Order A-1042: 2 wool throws, 78.00.")
    }

    /// A call that needs allowing ends its run on an interrupt rather than waiting for a
    /// result, which is what tells a client the agent is not going to get one on its own.
    @Test func aCallWaitingToBeAllowedEndsTheRunOnAnInterrupt() throws {
        var translator = AGUITranslator(threadID: "s1")
        let approval = Interrupt(
            id: "c2",
            reason: InterruptReason.toolCall,
            message: "Refund 78.00 for order A-1042?",
            toolCallId: "c2")

        let events = try translator.translate(
            frame(
                #"{"type":"tool_call","id":"c2","name":"refund_order","arguments":"{\"order_id\":\"A-1042\"}"}"#
            ),
            awaiting: approval)

        #expect(
            events.map(\.type) == [
                .runStarted, .toolCallStart, .toolCallArgs, .toolCallEnd, .runFinished,
            ])
        guard case .runFinished(let finished) = events.last else {
            Issue.record("the run has to end for the person to be asked")
            return
        }
        #expect(finished.outcome == .interrupt([approval]))
        #expect(finished.interrupts.first?.message == "Refund 78.00 for order A-1042?")
    }

    /// Answering an approval a minute later still has somewhere to go: the run it was asked in
    /// is over, so the result opens one of its own and the reply to it lands there.
    @Test func answeringAnApprovalOpensARunForTheResult() throws {
        var translator = AGUITranslator(threadID: "s1")
        let approval = Interrupt(id: "c2", reason: InterruptReason.toolCall, toolCallId: "c2")
        _ = try translator.translate(
            frame(#"{"type":"tool_call","id":"c2","name":"refund_order","arguments":"{}"}"#),
            awaiting: approval)

        var events = translator.answered("c2", with: "Refunded 78.00 to the card ending 4242.")
        events += try translator.translate(
            frame(#"{"type":"responded","turn_id":"t2","text":"That is refunded."}"#))

        #expect(
            events.map(\.type) == [
                .runStarted, .toolCallResult,
                .textMessageStart, .textMessageContent, .textMessageEnd, .runFinished,
            ])
    }

    /// A skill is work in progress, which the protocol has activities for. They need no
    /// closing, which is the point: a skill outlives the turn that dispatched it.
    @Test func aSkillIsPublishedAsAnActivity() throws {
        var translator = AGUITranslator(threadID: "s1")

        var events = try translator.translate(
            frame(
                #"{"type":"delegated","task_id":"k1","skill":"refund_decision","prompt":"is A-1042 refundable","turn_id":"t1"}"#
            ))
        events += try translator.translate(
            frame(
                #"{"type":"task_settled","task_id":"k1","skill":"refund_decision","text":"Owed 78.00 to the original card.","question":"","elapsed_ms":1200,"error":""}"#
            ))

        #expect(events.map(\.type) == [.runStarted, .activitySnapshot, .activitySnapshot])
        guard case .activitySnapshot(let settled) = events.last else {
            Issue.record("settled work is an activity")
            return
        }
        #expect(settled.messageId == "k1")
        #expect(settled.activityType == "skill")
        #expect(settled.content["skill"] == .string("refund_decision"))
        #expect(settled.content["state"] == .string("settled"))
    }

    /// What was typed is on the stream already, so a call transcribing it back is not a second
    /// question.
    @Test func whatWasTypedIsNotPublishedTwiceWhenItIsHeardBack() throws {
        var translator = AGUITranslator(threadID: "s1")

        let typed = translator.said("what are your hours")
        let heard = try translator.translate(
            frame(
                #"{"type":"heard","participant":{"id":"p1","user_id":"u1","name":""},"text":"what are your hours","language":"en"}"#
            ))

        #expect(typed.map(\.type) == [.runStarted, .textMessageStart, .textMessageContent, .textMessageEnd])
        #expect(heard.isEmpty)
    }

    /// A failure ends the run as one. Nothing follows it until something opens a run again.
    @Test func aFailureEndsTheRunAsAnError() throws {
        var translator = AGUITranslator(threadID: "s1")
        _ = try translator.translate(frame(#"{"type":"responding","turn_id":"t1"}"#))

        let events = try translator.translate(
            frame(#"{"type":"error","context":"llm","error":"the model timed out"}"#))

        #expect(events.map(\.type) == [.runError])
        guard case .runError(let failure) = events.first else {
            Issue.record("an error frame is a run error")
            return
        }
        #expect(failure.message == "the model timed out")
    }

    /// The router publishes a great deal a run has no place for, and an event this SDK has
    /// never heard of on top of that. Neither belongs on the protocol's stream.
    @Test func aFrameWithNoPlaceInTheProtocolIsDropped() throws {
        var translator = AGUITranslator(threadID: "s1")
        let ignored = [
            #"{"type":"joined","at":"2026-01-01T00:00:00Z"}"#,
            #"{"type":"participant_joined","participant":{"id":"p1","user_id":"u1","name":"Alice"},"at":"2026-01-01T00:00:00Z"}"#,
            #"{"type":"hearing","participant":{"id":"p1","user_id":"u1","name":"Alice"},"text":"refund","language":"en"}"#,
            #"{"type":"spoke","turn_id":"t1","audio_duration_ms":900,"time_to_first_byte_ms":120}"#,
            #"{"type":"turn","turn_id":"t1","roundtrip_ms":800,"interrupted":false}"#,
            #"{"type":"overheard","who":"nobody"}"#,
        ]

        for json in ignored {
            #expect(try translator.translate(frame(json)).isEmpty, "\(json)")
        }
    }

    /// Everything the SDK publishes has to satisfy the protocol's own sequencing rules, which
    /// is what makes the stream worth calling AG-UI. The verifier is AG-UI's, not ours.
    @Test func everythingPublishedSatisfiesTheProtocolsVerifier() throws {
        var translator = AGUITranslator(threadID: "s1")
        var verifier = EventVerifier()
        let approval = Interrupt(id: "c2", reason: InterruptReason.toolCall, toolCallId: "c2")

        var events = translator.said("refund order A-1042")
        for json in exchange {
            events += try translator.translate(frame(json))
        }
        events += translator.answered("c1", with: "Order A-1042: unopened.")
        events += try translator.translate(
            frame(
                #"{"type":"delegated","task_id":"k1","skill":"refund_decision","prompt":"?","turn_id":"t2"}"#
            ))
        events += try translator.translate(
            frame(
                #"{"type":"task_settled","task_id":"k1","skill":"refund_decision","text":"78.00","question":"","elapsed_ms":9,"error":""}"#
            ))
        events += try translator.translate(
            frame(#"{"type":"responding","turn_id":"t2","prompt":"?"}"#))
        events += try translator.translate(
            frame(#"{"type":"response_delta","turn_id":"t2","text":"You are owed 78.00."}"#))
        events += try translator.translate(
            frame(#"{"type":"responded","turn_id":"t2","text":"You are owed 78.00."}"#))
        events += try translator.translate(
            frame(#"{"type":"tool_call","id":"c2","name":"refund_order","arguments":"{}"}"#),
            awaiting: approval)
        events += translator.answered("c2", with: "Refunded.")
        events += try translator.translate(
            frame(#"{"type":"interrupted","turn_id":"t3"}"#))
        events += try translator.translate(frame(#"{"type":"left","at":"2026-01-01T00:00:00Z"}"#))

        for event in events {
            try verifier.verify(event)
        }
    }

    /// A conversation nobody interrupted still leaves no run open, so a client is never left
    /// waiting for the end of one.
    @Test func aConversationThatEndsClosesTheRunItWasIn() throws {
        var translator = AGUITranslator(threadID: "s1")
        _ = try translator.translate(frame(#"{"type":"responding","turn_id":"t1"}"#))
        _ = try translator.translate(
            frame(#"{"type":"response_delta","turn_id":"t1","text":"Half a sen"}"#))

        let events = translator.finish()

        #expect(events.map(\.type) == [.textMessageEnd, .runFinished])
        #expect(translator.finish().isEmpty, "there is nothing left to close")
    }
}
