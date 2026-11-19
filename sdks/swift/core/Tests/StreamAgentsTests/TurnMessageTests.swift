import Foundation
import Testing

@testable import StreamAgents

/// Frames exactly as `frameOf` in the router writes them.
private func event(_ json: String) throws -> AgentEvent {
    try JSONDecoder().decode(AgentEvent.self, from: Data(json.utf8))
}

/// A step's payload as StreamChatAI reads it.
private func fields(_ step: TurnMessage.Step) throws -> [String: RawJSON] {
    try JSONDecoder().decode([String: RawJSON].self, from: step.payload)
}

/// A question that needs a search, asked in an incognito session: the model thinks, calls the
/// search, and takes its result up in a turn of its own.
private func searched() throws -> Conversation {
    var conversation = Conversation()
    let start = Date(timeIntervalSince1970: 1_000)
    func at(_ seconds: TimeInterval) -> Date { start.addingTimeInterval(seconds) }

    conversation.said("Is the Belem Tower open on Sunday?", at: at(0))
    conversation.apply(try event(#"{"type":"responding","turn_id":"t1","prompt":"Is the Belem Tower open on Sunday?"}"#), at: at(0))
    conversation.apply(try event(#"{"type":"reasoning_delta","turn_id":"t1","text":"Look up its hours."}"#), at: at(2))
    conversation.apply(
        try event(
            #"{"type":"tool_started","tool_call_id":"call_1","tool":"search","turn_id":"t1","started_at":"2026-11-19T10:00:03Z"}"#
        ), at: at(3))
    conversation.apply(
        try event(
            #"{"type":"tool_ran","tool_call_id":"call_1","turn_id":"t1","tool":"search","arguments":"{}","result":"10:00-18:30","files":null,"error":""}"#
        ), at: at(3.5))
    conversation.apply(
        try event(#"{"pending_work":true,"type":"responded","turn_id":"t1","text":"","time_to_first_token_ms":0}"#),
        at: at(3.5))
    conversation.apply(try event(#"{"type":"responding","turn_id":"t2","prompt":""}"#), at: at(4))
    conversation.apply(try event(#"{"type":"reasoning_delta","turn_id":"t2","text":"Sundays are open."}"#), at: at(5))
    conversation.apply(try event(#"{"type":"response_delta","turn_id":"t2","text":"Yes, from 10:00."}"#), at: at(6))
    return conversation
}

@Suite struct TurnMessageTests {
    @Test func anIncognitoReplyIsOneMessageAcrossItsToolsTurnsWithEveryStep() throws {
        var conversation = try searched()
        conversation.apply(
            try event(
                #"{"pending_work":false,"type":"responded","turn_id":"t2","text":"Yes, from 10:00.","time_to_first_token_ms":900}"#
            ))

        let messages = TurnMessage.messages(of: conversation, userID: "jlahey", isConnected: true)

        #expect(messages.map(\.isAgentReply) == [false, true])
        #expect(messages[0].text == "Is the Belem Tower open on Sunday?")
        let reply = messages[1]
        #expect(reply.id == "t1")
        #expect(reply.text == "Yes, from 10:00.")
        #expect(reply.isGenerating == false)
        #expect(reply.steps.map(\.type) == ["ai_reasoning", "ai_tool_call", "ai_reasoning"])
        let thinking = try reply.steps.map { try fields($0)["id"]?.stringValue }
        // Numbered across the reply, as Stream Chat's copy of a kept one is.
        #expect(thinking == ["r1", "call_1", "r2"])
        #expect(try fields(reply.steps[0])["duration_ms"] == .number(3000))
        #expect(try fields(reply.steps[1])["status"] == .string("completed"))
        #expect(try fields(reply.steps[1])["duration_ms"] == .number(500))
        #expect(reply.reasoning == ["r1": "Look up its hours.", "r2": "Sundays are open."])
    }

    @Test func onlyTheLatestReplyIsBeingWrittenAndOnlyWhileTheSessionIsConnected() throws {
        let conversation = try searched()

        let connected = TurnMessage.messages(of: conversation, userID: "jlahey", isConnected: true)
        let disconnected = TurnMessage.messages(of: conversation, userID: "jlahey", isConnected: false)

        #expect(connected.last?.isGenerating == true)
        #expect(try fields(try #require(connected.last?.steps.last))["status"] == .string("completed"))
        #expect(disconnected.last?.isGenerating == false)

        var askedAgain = conversation
        askedAgain.said("And on Monday?")
        let asked = TurnMessage.messages(of: askedAgain, userID: "jlahey", isConnected: true)
        #expect(asked.map(\.isGenerating) == [false, false, false])
    }

    @Test func aCallWaitingForThePersonAsksTheSessionsUserOnItsStep() throws {
        var conversation = Conversation()
        conversation.said("Remind me to call Bob at 5.")
        conversation.apply(try event(#"{"type":"responding","turn_id":"t1","prompt":"Remind me to call Bob at 5."}"#))
        conversation.apply(
            try event(
                #"{"type":"tool_started","tool_call_id":"call_9","tool":"add_reminder","turn_id":"t1","started_at":"2026-11-19T10:00:03Z"}"#
            ))
        let tool = AgentTool(
            name: "add_reminder", description: "Adds a reminder.",
            approval: AgentTool.Approval(
                title: "Add a reminder?", message: "Juno wants to add one.", reasonArgument: "why",
                allowTitle: "Add", declineTitle: "Not now")
        ) { _ in "added" }
        let waiting = ToolApprovalRequest(
            call: AgentEvent.ToolCall(
                id: "call_9", name: "add_reminder", arguments: #"{"why":"You asked to call Bob."}"#),
            approval: try #require(tool.approval), tool: tool)

        let messages = TurnMessage.messages(
            of: conversation, approvals: [waiting], userID: "jlahey", isConnected: true)
        let step = try fields(try #require(messages.last?.steps.first))

        #expect(step["status"] == .string("awaiting_approval"))
        #expect(step["target_user_id"] == .string("jlahey"))
        #expect(
            step["approval"]
                == .dictionary([
                    "title": .string("Add a reminder?"), "message": .string("Juno wants to add one."),
                    "reason": .string("You asked to call Bob."), "allow_title": .string("Add"),
                    "decline_title": .string("Not now"),
                ]))

        // Once answered, it is a call like any other.
        let answered = TurnMessage.messages(of: conversation, userID: "jlahey", isConnected: true)
        let after = try fields(try #require(answered.last?.steps.first))
        #expect(after["status"] == .string("running"))
        #expect(after["approval"] == nil)
    }

    @Test func aRepliesUIIsItsLatestTurnsAsAKeptReplyHoldsIt() throws {
        var conversation = Conversation()
        conversation.apply(try event(#"{"type":"responding","turn_id":"t1","prompt":"Plan Ohrid"}"#))
        conversation.apply(
            try event(
                #"{"type":"ui_completed","turn_id":"t1","format":"openui","content":"root = Card([hdr])\nhdr = CardHeader(\"Ohrid\")"}"#
            ))

        let reply = try #require(TurnMessage.messages(of: conversation, userID: "jlahey", isConnected: true).last)

        #expect(reply.customData["ai_ui"] == conversation.turns[0].ui?.customData)
        #expect(reply.customData["ai_ui"]?.dictionaryValue?["status"] == .string("complete"))
    }
}
