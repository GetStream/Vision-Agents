import Foundation
import StreamCore

/// One line of a conversation.
public struct Turn: Sendable, Hashable, Identifiable {
    /// The router's turn id for an agent turn, and one of our own for a participant's.
    public let id: String
    public let speaker: Speaker
    public var text: String
    public let at: Date
    /// The interactive UI an agent turn wrote, as it streams in, or nil for a turn without any.
    public var ui: TurnUI?
    /// What the agent did on its way to the turn's text, in order: each round of the model's
    /// thinking on a written reply, and each tool it called. Empty on a participant's turn.
    public var steps: [TurnStep] = []
    /// Whether the agent is done with the turn: it answered with no tools or work left to come,
    /// or it was interrupted. A turn whose tools are still running is not. Never set on a
    /// participant's turn.
    public var isDone = false

    public enum Speaker: Sendable, Hashable {
        /// A person, on the call or typing. Nil when the router did not say who.
        case participant(Participant?)
        /// The agent.
        case agent

        public var isAgent: Bool { self == .agent }
    }
}

/// One thing the agent did in a turn.
public enum TurnStep: Sendable, Hashable, Identifiable {
    /// A round of the model's thinking, which is read, never spoken.
    case reasoning(Reasoning)
    /// A call of a tool: what was called and how it went, not what it was given or gave.
    case toolCall(ToolCall)

    public var id: String {
        switch self {
        case .reasoning(let round): round.id
        case .toolCall(let call): call.id
        }
    }

    public struct Reasoning: Sendable, Hashable {
        /// `r1`, `r2` and on, in the order the turn thought.
        public let id: String
        /// Whether the model is still thinking: until it writes, calls a tool or finishes.
        public var isStreaming: Bool
        /// The thinking so far.
        public var text: String
        /// When the model began on it: when it was asked, or took the reply up again after
        /// writing, as a kept reply's step is timed.
        public let at: Date
        /// How long the model thought, once it is done.
        public var duration: TimeInterval?
    }

    public struct ToolCall: Sendable, Hashable {
        /// The call's id, which `AgentSession.decide` answers it by while it waits in `approvals`.
        public let id: String
        /// The tool.
        public let name: String
        public var status: Status
        /// What whoever runs it last said about how it is going (`tool_update`), or the empty
        /// string.
        public var summary: String
        /// When it began.
        public let at: Date
        /// How long it ran, once it is no longer running.
        public var duration: TimeInterval?

        public enum Status: String, Sendable {
            case running
            case completed
            case failed
        }
    }
}

/// A conversation, as the events so far have left it.
///
/// This is a value type with no network in it, which is the whole design: what a stream of
/// frames means for the transcript is decided here, and `AgentSession` is only the actor
/// boundary and the socket around it.
public struct Conversation: Sendable, Hashable {
    /// The conversation so far, oldest first. The agent's turn in flight is the last entry and
    /// grows as deltas arrive.
    public var turns: [Turn] = []

    /// What the agent is doing.
    public var state: State = .idle

    /// What the agent reported going wrong, or nil. Errors arrive as events rather than being
    /// thrown, because nobody is awaiting the socket.
    public var failure: String?

    /// What each agent turn's UI block wrote so far, by turn id: the deltas, or a patch before
    /// it is merged. Kept apart from `Turn.ui`, which is what shows and is left unchanged by a
    /// delta that shows nothing new.
    private var uiWritten: [String: String] = [:]

    /// When the model last took each turn in flight up again, which a round of thinking is
    /// timed from.
    private var resumedAt: [String: Date] = [:]

    public init() {}

    public enum State: Sendable, Hashable {
        /// Waiting to be spoken to.
        case idle
        /// Somebody is talking and being transcribed.
        case listening
        /// The model is answering.
        case responding
        /// Skills are thinking, named here so a view can say what about.
        case working([String])
        /// The conversation is over.
        case ended
    }

    /// Whatever the caller typed, shown before the router has confirmed hearing it.
    public mutating func said(_ text: String, at now: Date = Date()) {
        turns.append(
            Turn(id: UUID().uuidString, speaker: .participant(nil), text: text, at: now))
    }

    /// Folds one event into the conversation.
    ///
    /// An event with no bearing on the transcript, and one this SDK has never heard of, both
    /// leave it alone.
    public mutating func apply(_ event: AgentEvent, at now: Date = Date()) {
        switch event.kind {
        case .heard:
            // A text session echoes nothing back, so what the caller typed is already here.
            // A call transcribes what was spoken, which is the first anyone hears of it.
            if !matchesLastParticipantTurn(event.text) {
                turns.append(
                    Turn(
                        id: UUID().uuidString, speaker: .participant(event.participant),
                        text: event.text, at: now))
            }
            state = .idle

        case .hearing:
            state = .listening

        case .responding:
            state = .responding
            turns.append(Turn(id: event.turnID, speaker: .agent, text: "", at: now))
            resumedAt[event.turnID] = now

        case .responseDelta:
            state = .responding
            if !event.text.isEmpty {
                amend(event.turnID) { $0.settle(at: now) }
                resumedAt[event.turnID] = now
            }
            write(event.text, to: event.turnID, at: now) { $0 += event.text }

        case .reasoningDelta:
            guard !event.text.isEmpty else { return }
            let began = resumedAt[event.turnID] ?? now
            amend(event.turnID, startingAt: now) { $0.think(event.text, began: began) }

        case .toolStarted:
            let call = TurnStep.ToolCall(
                id: event.string("tool_call_id"), name: event.string("tool"), status: .running,
                summary: "", at: now)
            amend(event.turnID, startingAt: now) {
                $0.settle(at: now)
                $0.call(call)
            }

        case .toolUpdated:
            // Words to say as written are a line of the reply, not how the call is going.
            guard event["verbatim"].boolValue != true, !event.text.isEmpty else { return }
            amend(event.turnID) { $0.changeCall(event.string("id")) { $0.summary = event.text } }

        case .toolRan:
            let status: TurnStep.ToolCall.Status = event.errorText.isEmpty ? .completed : .failed
            amend(event.turnID) {
                $0.changeCall(event.string("tool_call_id")) { call in
                    call.status = status
                    call.duration = now.timeIntervalSince(call.at)
                }
            }

        case .uiDelta:
            let delta = event.string("delta")
            let format = event.string("format")
            if !delta.isEmpty {
                amend(event.turnID) { $0.settle(at: now) }
                resumedAt[event.turnID] = now
            }
            guard !delta.isEmpty, !format.isEmpty else { return }
            let written = uiWritten[event.turnID, default: ""] + delta
            uiWritten[event.turnID] = written
            writeUI(event, format: format, written: written, isStreaming: true, at: now)
            if case .working = state {} else { state = .responding }

        case .uiCompleted:
            let format = event.string("format")
            guard !format.isEmpty else { return }
            let content = event.string("content")
            let written = content.isEmpty ? uiWritten[event.turnID, default: ""] : content
            uiWritten[event.turnID] = nil
            writeUI(event, format: format, written: written, isStreaming: false, at: now)

        case .responded:
            // The final text is authoritative: the deltas are what was being written, this is
            // what was said. An empty one adds nothing, which is what a spoken-only turn is.
            if !event.text.isEmpty {
                write(event.text, to: event.turnID, at: now) { $0 = event.text }
            }
            // A reply whose tools or work are still to come is carried on by a turn of its own.
            let isDone = event["pending_work"].boolValue != true
            resumedAt[event.turnID] = nil
            amend(event.turnID) {
                $0.settle(at: now)
                if isDone { $0.isDone = true }
            }
            state = .idle

        case .delegated:
            state = .working(working + [event.string("skill")])

        case .taskSettled, .taskCancelled:
            let left = working.filter { $0 != event.string("skill") }
            state = left.isEmpty ? .responding : .working(left)

        case .interrupted:
            resumedAt[event.turnID] = nil
            amend(event.turnID) {
                $0.settle(at: now)
                $0.isDone = true
            }
            state = .idle

        case .error:
            failure = event.errorText

        case .left:
            state = .ended

        default:
            break
        }
    }

    private var working: [String] {
        if case .working(let skills) = state { return skills }
        return []
    }

    private func matchesLastParticipantTurn(_ text: String) -> Bool {
        guard case .participant = turns.last?.speaker else { return false }
        return turns.last?.text == text
    }

    /// Applies a change to the agent turn this event belongs to, starting one if the router
    /// sent a delta for a turn we never saw begin.
    private mutating func write(
        _ text: String,
        to turnID: String,
        at now: Date,
        _ change: (inout String) -> Void
    ) {
        if let index = turns.lastIndex(where: { $0.id == turnID && $0.speaker.isAgent }) {
            change(&turns[index].text)
        } else {
            turns.append(Turn(id: turnID, speaker: .agent, text: text, at: now))
        }
    }

    /// Changes the agent turn `turnID`. A turn this never saw begin is started at `start` when
    /// the event begins something, and left alone otherwise.
    private mutating func amend(_ turnID: String, startingAt start: Date? = nil, _ change: (inout Turn) -> Void) {
        if let index = turns.lastIndex(where: { $0.id == turnID && $0.speaker.isAgent }) {
            change(&turns[index])
        } else if let start {
            var turn = Turn(id: turnID, speaker: .agent, text: "", at: start)
            change(&turn)
            turns.append(turn)
        }
    }

    /// Shows `written`, what the turn's UI block wrote so far, as the UI of the agent turn a
    /// `ui_delta` or `ui_completed` belongs to, starting the turn if the router sent UI for one we
    /// never saw begin. While OpenUI streams, only its finished statements show. A patch is merged
    /// onto the UI of the latest earlier turn in the same format; with none in this transcript,
    /// such as after a reload, it sets no UI, and Stream Chat's copy, which the router merged,
    /// is the one to show.
    private mutating func writeUI(
        _ event: AgentEvent,
        format: String,
        written: String,
        isStreaming: Bool,
        at now: Date
    ) {
        let turnID = event.turnID
        var content = isStreaming && format == TurnUI.openUI ? TurnUI.finished(written) : written
        if event.string("mode") == "patch", format == TurnUI.openUI {
            guard
                let base = turns.last(where: {
                    $0.speaker.isAgent && $0.id != turnID && $0.ui?.format == TurnUI.openUI
                        && $0.ui?.content.isEmpty == false
                })?.ui
            else { return }
            content = TurnUI.merge(base.content, patch: content)
        }
        let version = event.string("version")
        let ui = TurnUI(
            format: format, version: version.isEmpty ? nil : version, isStreaming: isStreaming,
            content: content)
        let index = turns.lastIndex(where: { $0.id == turnID && $0.speaker.isAgent })
        if let index {
            turns[index].ui = ui
        } else {
            turns.append(Turn(id: turnID, speaker: .agent, text: "", at: now, ui: ui))
        }
    }
}

extension Turn {
    /// Adds `delta` to the round of thinking under way, or to a new one when the model did
    /// something else since: every round is numbered from `r1`, as a kept reply's steps are.
    fileprivate mutating func think(_ delta: String, began: Date) {
        if case .reasoning(var round)? = steps.last, round.isStreaming {
            round.text += delta
            steps[steps.count - 1] = .reasoning(round)
            return
        }
        let rounds = steps.filter { if case .reasoning = $0 { true } else { false } }.count
        steps.append(.reasoning(TurnStep.Reasoning(id: "r\(rounds + 1)", isStreaming: true, text: delta, at: began)))
    }

    /// Completes the round of thinking under way, if there is one.
    fileprivate mutating func settle(at now: Date) {
        guard case .reasoning(var round)? = steps.last, round.isStreaming else { return }
        round.isStreaming = false
        round.duration = now.timeIntervalSince(round.at)
        steps[steps.count - 1] = .reasoning(round)
    }

    /// Begins `call`, unless the turn already shows it.
    fileprivate mutating func call(_ call: TurnStep.ToolCall) {
        guard !steps.contains(where: { $0.id == call.id }) else { return }
        steps.append(.toolCall(call))
    }

    /// Changes the running call `id`, if the turn has one.
    fileprivate mutating func changeCall(_ id: String, _ change: (inout TurnStep.ToolCall) -> Void) {
        for (index, step) in steps.enumerated() {
            guard case .toolCall(var call) = step, call.id == id, call.status == .running else { continue }
            change(&call)
            steps[index] = .toolCall(call)
            return
        }
    }
}

/// The interactive UI an agent turn writes, such as a card in OpenUI Lang, as the session's
/// socket streams it: `ui_delta` as the model writes it, before stage directions are taken out
/// or anything is spoken, then `ui_completed`.
///
/// The router writes the same UI on the reply's message in Stream Chat as `ai_ui`, later and
/// less often; ``LiveUI`` shows this one on the message until that one is complete. On a call,
/// where nothing is spoken of it, this is the only place it shows.
public struct TurnUI: Sendable, Hashable {
    /// The format OpenUI Lang is written in, the one whose patches are merged.
    public static let openUI = "openui"

    /// The format it is written in, such as `openui` or `a2ui`.
    public var format: String
    /// The format's version, when the model named one.
    public var version: String?
    /// Whether the model is still writing it.
    public var isStreaming: Bool
    /// The UI to show. While OpenUI is written, that is its finished statements: one still being
    /// written shows once a newline ends it, so the UI grows a part at a time rather than
    /// changing with every delta, such as a gauge reading 8 before 86. A patch, which changes
    /// only some statements of the turn before's OpenUI, is already merged onto it here.
    public var content: String

    /// The UI as a message's `ai_ui` custom field holds it, the shape StreamChatAI's
    /// `AIUIDocument(customData:)` reads under that key.
    public var customData: RawJSON {
        var fields: [String: RawJSON] = [
            "v": .number(1),
            "format": .string(format),
            "status": .string(isStreaming ? "streaming" : "complete"),
            "content": .string(content),
        ]
        if let version {
            fields["version"] = .string(version)
        }
        return .dictionary(fields)
    }

    /// An OpenUI program with `patch` applied, as the router merges one: a statement the patch
    /// assigns again is replaced where it was, a new one goes at the end, and a component
    /// assigned `null` is removed. A `$variable` assigned `null` keeps the `null`.
    static func merge(_ base: String, patch: String) -> String {
        if base.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return patch }
        if patch.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return base }
        var order: [String] = []
        var statements: [String: String] = [:]
        for statement in openUIStatements(base) {
            guard let (name, _) = openUIAssignment(statement) else { continue }
            if statements[name] == nil { order.append(name) }
            statements[name] = statement
        }
        for statement in openUIStatements(patch) {
            guard let (name, value) = openUIAssignment(statement) else { continue }
            if value == "null", !name.hasPrefix("$") {
                statements[name] = nil
                order.removeAll { $0 == name }
                continue
            }
            if statements[name] == nil { order.append(name) }
            statements[name] = name + " = " + value
        }
        return order.compactMap { statements[$0] }.joined(separator: "\n")
    }

    /// The statements of a program being written that are finished, the one still being written
    /// left out.
    static func finished(_ program: String) -> String {
        openUIStatements(program, finishedOnly: true).joined(separator: "\n")
    }

    /// A program's statements: each line, or the lines a bracket or a string left open joined
    /// to the one that closes it. Blank and comment lines are left out. With `finishedOnly`, so
    /// is a statement no newline has ended yet.
    private static func openUIStatements(_ program: String, finishedOnly: Bool = false) -> [String] {
        var statements: [String] = []
        var current = ""
        var quote: Character?
        var escaped = false
        var depth = 0
        var lines = program.split(separator: "\n", omittingEmptySubsequences: false)
        if finishedOnly {
            // What follows the last newline is still being written.
            lines.removeLast()
        }
        for line in lines {
            if quote == nil, depth == 0 {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty || trimmed.hasPrefix("//") || trimmed.hasPrefix("#") { continue }
            }
            if !current.isEmpty { current += "\n" }
            current += line
            var previous: Character?
            for character in line {
                defer { previous = character }
                if escaped {
                    escaped = false
                    continue
                }
                if let open = quote {
                    if character == "\\" {
                        escaped = true
                    } else if character == open {
                        quote = nil
                    }
                    continue
                }
                if character == "/", previous == "/" { break }
                switch character {
                case "\"", "'": quote = character
                case "(", "[", "{": depth += 1
                case ")", "]", "}": depth = max(0, depth - 1)
                default: break
                }
            }
            if quote == nil, depth == 0 {
                let finished = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !finished.isEmpty { statements.append(finished) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty, !finishedOnly { statements.append(tail) }
        return statements
    }

    /// Whether `name` can be assigned: an identifier, or a `$variable`.
    private static func isOpenUIName(_ name: String) -> Bool {
        let identifier = name.hasPrefix("$") ? name.dropFirst() : Substring(name)
        guard let first = identifier.first, first == "_" || (first.isASCII && first.isLetter) else { return false }
        return identifier.allSatisfy { $0 == "_" || ($0.isASCII && ($0.isLetter || $0.isNumber)) }
    }

    /// The name a statement assigns and the value it assigns, or nil when it assigns nothing
    /// a name can be read from.
    private static func openUIAssignment(_ statement: String) -> (name: String, value: String)? {
        let characters = Array(statement)
        var index = 0
        while index < characters.count {
            defer { index += 1 }
            guard characters[index] == "=" else { continue }
            if index > 0, "!<>=".contains(characters[index - 1]) { continue }
            if index + 1 < characters.count, characters[index + 1] == "=" {
                index += 1
                continue
            }
            let name = String(characters[..<index]).trimmingCharacters(in: .whitespaces)
            guard isOpenUIName(name) else { return nil }
            let value = String(characters[(index + 1)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            return (name, value)
        }
        return nil
    }
}
