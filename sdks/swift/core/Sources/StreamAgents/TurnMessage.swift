import Foundation
import StreamCore

/// A message of a conversation kept in no channel, an incognito session's, in the shape a kept
/// one has in Stream Chat, so an app shows both alike with Stream's AI components.
///
/// Each of the person's turns is a message, and the agent's turns between two of them are one
/// reply, as the router writes a reply whose tools it waited on: their text, the latest UI, and
/// each round of thinking and each tool call as a step, generating until the agent is done
/// with it. A call waiting in the session's `approvals` asks its user on its step, as a kept
/// reply's does.
public struct TurnMessage: Sendable, Hashable, Identifiable {
    /// Its first turn's id.
    public let id: String
    /// Whether the agent wrote it, rather than the person.
    public let isAgentReply: Bool
    public let text: String
    /// When its first turn began.
    public let at: Date
    /// Whether the agent is still writing it.
    public let isGenerating: Bool
    /// The reply's steps, in order, as a kept reply's attachments hold them: an `ai_reasoning`
    /// for each round of thinking, numbered from `r1` across the reply, and an `ai_tool_call`
    /// for each call. StreamChatAI's `AIMessagePart.parts(from:)` reads them.
    public let steps: [Step]
    /// The whole of each round of thinking, by its step's id.
    public let reasoning: [String: String]
    /// The reply's custom data as a kept one holds it: its UI under `ai_ui`, which
    /// `AIUIDocument(customData:)` reads, or nothing.
    public let customData: [String: RawJSON]

    /// A step of a reply, as an attachment's type and its JSON payload.
    public struct Step: Sendable, Hashable {
        public let type: String
        public let payload: Data

        init(type: String, fields: [String: RawJSON]) {
            self.type = type
            // Sorted, so a step that did not change is the same bytes and a view leaves it be.
            let encoder = JSONEncoder()
            encoder.outputFormatting = .sortedKeys
            // A dictionary of strings, numbers and dictionaries of them always encodes.
            payload = (try? encoder.encode(RawJSON.dictionary(fields))) ?? Data()
        }
    }

    /// The messages of `conversation`, oldest first.
    ///
    /// - Parameters:
    ///   - approvals: The calls waiting for the person, each of which asks on its step.
    ///   - userID: The person the session is with, whom a waiting call asks.
    ///   - isConnected: Whether the session's socket still carries the conversation. The latest
    ///     reply is written only while it does.
    public static func messages(
        of conversation: Conversation,
        approvals: [ToolApprovalRequest] = [],
        userID: String,
        isConnected: Bool
    ) -> [TurnMessage] {
        var groups: [[Turn]] = []
        for turn in conversation.turns {
            if turn.speaker.isAgent, groups.last?.first?.speaker.isAgent == true {
                groups[groups.count - 1].append(turn)
            } else {
                groups.append([turn])
            }
        }
        return groups.enumerated().map { index, turns in
            // Only the latest reply can still be written: one the person asked over is as it
            // was left.
            let isLatest = index == groups.count - 1
            return TurnMessage(
                turns, isGenerating: isLatest && isConnected && turns.last?.isDone != true,
                approvals: approvals, userID: userID)
        }
    }

    private init(_ turns: [Turn], isGenerating: Bool, approvals: [ToolApprovalRequest], userID: String) {
        let first = turns[0]
        id = first.id
        at = first.at
        isAgentReply = first.speaker.isAgent
        guard isAgentReply else {
            text = first.text
            self.isGenerating = false
            steps = []
            reasoning = [:]
            customData = [:]
            return
        }
        self.isGenerating = isGenerating
        text = turns.map(\.text).filter { !$0.isEmpty }.joined(separator: "\n\n")
        var steps: [Step] = []
        var reasoning: [String: String] = [:]
        for step in turns.flatMap(\.steps) {
            switch step {
            case .reasoning(let round):
                // Numbered across the reply, which a tool's result carries on in a turn of its own.
                let id = "r\(reasoning.count + 1)"
                reasoning[id] = round.text
                var fields: [String: RawJSON] = [
                    "id": .string(id), "status": .string(round.isStreaming ? "streaming" : "completed"),
                ]
                fields["duration_ms"] = round.duration.map(Self.milliseconds)
                steps.append(Step(type: "ai_reasoning", fields: fields))
            case .toolCall(let call):
                var fields: [String: RawJSON] = [
                    "id": .string(call.id), "name": .string(call.name), "status": .string(call.status.rawValue),
                ]
                if !call.summary.isEmpty {
                    fields["summary"] = .string(call.summary)
                }
                fields["duration_ms"] = call.duration.map(Self.milliseconds)
                if call.status == .running, let waiting = approvals.first(where: { $0.id == call.id }) {
                    fields["status"] = .string("awaiting_approval")
                    fields["target_user_id"] = .string(userID)
                    fields["approval"] = .dictionary(Self.approval(waiting))
                }
                steps.append(Step(type: "ai_tool_call", fields: fields))
            }
        }
        self.steps = steps
        self.reasoning = reasoning
        customData = turns.last(where: { $0.ui != nil })?.ui.map { ["ai_ui": $0.customData] } ?? [:]
    }

    private static func milliseconds(_ interval: TimeInterval) -> RawJSON {
        .number((interval * 1000).rounded())
    }

    /// The question a waiting call asks, as its step in Stream Chat holds it.
    private static func approval(_ waiting: ToolApprovalRequest) -> [String: RawJSON] {
        var fields: [String: RawJSON] = ["title": .string(waiting.approval.title)]
        fields["message"] = waiting.approval.message.map(RawJSON.string)
        fields["reason"] = waiting.reason.isEmpty ? nil : .string(waiting.reason)
        fields["allow_title"] = waiting.approval.allowTitle.map(RawJSON.string)
        fields["decline_title"] = waiting.approval.declineTitle.map(RawJSON.string)
        return fields
    }
}

extension AgentSession {
    /// The conversation as messages, for a session kept in no channel, an incognito one, to show
    /// with Stream's AI components. A kept session's messages are its channel's (`chat()`).
    public var turnMessages: [TurnMessage] {
        TurnMessage.messages(of: conversation, approvals: approvals, userID: session.userID, isConnected: isConnected)
    }
}
