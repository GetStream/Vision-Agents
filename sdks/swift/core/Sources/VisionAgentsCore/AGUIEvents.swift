import AGUI
import Foundation

/// The router's session frames, in the AG-UI protocol's own terms.
///
/// AG-UI describes a *run*: the work an agent does in answer to something somebody said, as a
/// message that streams, the tools it called on the way, and how it ended. The router describes
/// a *turn*, and publishes a good deal a run has no place for -- who joined a call, how long
/// the audio took, what the flow controller decided about a cough. What carries a meaning in
/// the protocol is translated here; the rest is dropped, and all of it is still on `AgentEvent`
/// for a caller who wants it.
///
/// A value type with no network in it, for the reason `Conversation` is one: what a stream of
/// frames means is decided here and tested against real frames without a router.
/// `AgentSession` owns one and publishes what it returns on `aguiEvents()`.
///
/// Runs are opened by whatever begins an exchange and closed by the reply that ends it. A turn
/// the model finished by asking for a tool is a run of its own, because that is what the router
/// does: the tool call arrives after the turn it came from has been reported, and the answer to
/// it is a second turn.
public struct AGUITranslator: Sendable {
    /// The AG-UI thread these runs belong to, which is the session's id.
    public let threadID: String

    /// The run in flight, or nil between runs.
    private var runID: String?

    /// The text message being written, or nil when none is.
    private var messageID: String?

    /// What has been published for that message, so that the final text a turn reports can be
    /// compared with what streamed. A message in AG-UI is the deltas it was sent, and there is
    /// no correcting it afterwards short of a whole snapshot.
    private var written = ""

    /// The last agent message of this conversation, which is what a tool call afterwards hangs
    /// off: the model asked for the tool in that message, and saying so is what puts the call
    /// and its result in the right place in the history.
    private var lastMessageID: String?

    /// What the caller last typed, so that a call transcribing it back does not say it twice.
    private var lastSaid: String?

    /// Numbers the runs this has opened for something that carries no turn id.
    private var runs = 0

    public init(threadID: String) {
        self.threadID = threadID
    }

    /// Folds one frame into the run in flight and returns the protocol events it means.
    ///
    /// - Parameters:
    ///   - event: the frame, as it arrived.
    ///   - approval: what somebody has to allow before this tool call can run, when it is one
    ///     that needs allowing. The run ends on it rather than waiting for a result.
    public mutating func translate(
        _ event: AgentEvent,
        awaiting approval: Interrupt? = nil
    ) -> [AGUI.Event] {
        switch event.kind {
        case .heard:
            // A text session hears nothing back, so this is a call transcribing what was
            // spoken -- unless it is transcribing what was typed during one, which is
            // already on the stream.
            guard event.text != lastSaid else {
                lastSaid = nil
                return []
            }
            return heard(event.text)

        case .responding:
            // The exchange this answers has a run already. One the agent started for itself,
            // to say what a tool returned or what a skill came back with, does not.
            return open(event.turnID)

        case .responseDelta:
            var events = open(event.turnID)
            events += write(event.turnID)
            events += content(event.text, to: event.turnID)
            return events

        case .responded:
            var events: [AGUI.Event] = []
            if messageID == nil && !event.text.isEmpty {
                // Nothing streamed: a reply that arrived whole, which is what a turn the
                // agent only spoke looks like.
                events += open(event.turnID)
                events += write(event.turnID)
                events += content(event.text, to: event.turnID)
            } else if messageID == event.turnID, event.text.hasPrefix(written),
                event.text.count > written.count
            {
                // The final text is the authoritative one and the deltas are what was being
                // written, so a tail the deltas never carried is published before the message
                // closes. A final text that is not what streamed at all is left alone: the
                // message is what somebody has already read.
                events += content(String(event.text.dropFirst(written.count)), to: event.turnID)
            }
            return events + close(.success)

        case .toolCall:
            guard let call = event.toolCall else { return [] }
            var events = open(nil)
            events.append(
                .toolCallStart(
                    ToolCallStartEvent(
                        toolCallId: call.id,
                        toolCallName: call.name,
                        parentMessageId: lastMessageID)))
            if !call.arguments.isEmpty {
                events.append(
                    .toolCallArgs(
                        ToolCallArgsEvent(toolCallId: call.id, delta: call.arguments)))
            }
            events.append(.toolCallEnd(ToolCallEndEvent(toolCallId: call.id)))
            if let approval {
                events += close(.interrupt([approval]))
            }
            return events

        case .delegated:
            var events = open(nil)
            events.append(
                activity(
                    event["task_id"].stringValue,
                    [
                        "skill": .string(event["skill"].stringValue),
                        "prompt": .string(event["prompt"].stringValue),
                        "state": .string("running"),
                    ]))
            return events

        case .taskSettled:
            var events = open(nil)
            events.append(
                activity(
                    event["task_id"].stringValue,
                    [
                        "skill": .string(event["skill"].stringValue),
                        "text": .string(event.text),
                        "state": .string(event.errorText.isEmpty ? "settled" : "failed"),
                        "error": .string(event.errorText),
                    ]))
            return events

        case .taskCancelled:
            var events = open(nil)
            events.append(
                activity(
                    event["task_id"].stringValue,
                    [
                        "skill": .string(event["skill"].stringValue),
                        "state": .string("cancelled"),
                        "reason": .string(event["reason"].stringValue),
                    ]))
            return events

        case .interrupted:
            // Somebody talked over the reply. The run is over either way, and abandoning one
            // is not a failure of it.
            return close(.success)

        case .error:
            return failed(event.errorText)

        case .left:
            return finish()

        default:
            // Everything the protocol has no place for: who joined, what was pressed, how
            // long the audio took. A frame this SDK has never heard of lands here too.
            return []
        }
    }

    /// Something the caller typed, which the router answers without saying it back.
    public mutating func said(_ text: String) -> [AGUI.Event] {
        lastSaid = text
        return heard(text)
    }

    /// What a tool on this device returned, which the router does not report to the device
    /// that ran it.
    ///
    /// A result that arrives after its run ended -- an approval answered a minute later --
    /// opens a run of its own, so the reply it produces has somewhere to be.
    public mutating func answered(_ id: String, with content: String) -> [AGUI.Event] {
        var events = open(nil)
        events.append(
            .toolCallResult(
                ToolCallResultEvent(
                    messageId: "result-\(id)", toolCallId: id, content: content)))
        return events
    }

    /// Ends the run in flight, for a conversation that is over.
    public mutating func finish() -> [AGUI.Event] {
        close(.success)
    }

    /// Reports the run in flight as failed. Nothing follows a failure until a run starts.
    public mutating func failed(_ message: String) -> [AGUI.Event] {
        runID = nil
        messageID = nil
        return [.runError(RunErrorEvent(message: message))]
    }

    /// Somebody's utterance, as a message of theirs. It ends whatever the agent was doing:
    /// a new question is a new exchange, and an answer to the last one is abandoned.
    private mutating func heard(_ text: String) -> [AGUI.Event] {
        guard !text.isEmpty else { return [] }
        var events = close(.success)
        events += open(nil)
        let id = Message.generateID()
        events.append(.textMessageStart(TextMessageStartEvent(messageId: id, role: .user)))
        events.append(.textMessageContent(TextMessageContentEvent(messageId: id, delta: text)))
        events.append(.textMessageEnd(TextMessageEndEvent(messageId: id)))
        return events
    }

    /// Starts a run unless one is already in flight, named after the turn when there is one.
    private mutating func open(_ turnID: String?) -> [AGUI.Event] {
        guard runID == nil else { return [] }
        runs += 1
        let id = turnID.flatMap { $0.isEmpty ? nil : $0 } ?? "run-\(runs)"
        runID = id
        return [.runStarted(RunStartedEvent(threadId: threadID, runId: id))]
    }

    /// Publishes a piece of the message being written and remembers it.
    private mutating func content(_ text: String, to messageID: String) -> [AGUI.Event] {
        guard !text.isEmpty else { return [] }
        written += text
        return [.textMessageContent(TextMessageContentEvent(messageId: messageID, delta: text))]
    }

    /// Starts the agent's message for this turn unless it is already being written.
    private mutating func write(_ turnID: String) -> [AGUI.Event] {
        guard messageID != turnID else { return [] }
        var events = end()
        messageID = turnID
        lastMessageID = turnID
        written = ""
        events.append(
            .textMessageStart(TextMessageStartEvent(messageId: turnID, role: .assistant)))
        return events
    }

    /// Closes the message being written, if there is one. A run cannot finish over an open one.
    private mutating func end() -> [AGUI.Event] {
        guard let id = messageID else { return [] }
        messageID = nil
        return [.textMessageEnd(TextMessageEndEvent(messageId: id))]
    }

    /// Ends the run in flight, closing what it left open.
    private mutating func close(_ outcome: RunFinishedOutcome) -> [AGUI.Event] {
        guard let id = runID else { return [] }
        var events = end()
        runID = nil
        events.append(
            .runFinished(RunFinishedEvent(threadId: threadID, runId: id, outcome: outcome)))
        return events
    }

    /// A skill, as the activity the protocol shows work in progress with. An activity needs no
    /// closing, which is what makes it the right shape for work that outlives its turn.
    private func activity(_ id: String, _ content: [String: AGUI.JSONValue]) -> AGUI.Event {
        .activitySnapshot(
            ActivitySnapshotEvent(messageId: id, activityType: "skill", content: content))
    }
}
