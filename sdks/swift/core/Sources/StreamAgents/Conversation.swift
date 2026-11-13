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

    public enum Speaker: Sendable, Hashable {
        /// A person, on the call or typing. Nil when the router did not say who.
        case participant(Participant?)
        /// The agent.
        case agent

        public var isAgent: Bool { self == .agent }
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

        case .responseDelta:
            state = .responding
            write(event.text, to: event.turnID, at: now) { $0 += event.text }

        case .uiDelta:
            let delta = event.string("delta")
            let format = event.string("format")
            guard !delta.isEmpty, !format.isEmpty else { return }
            writeUI(event, format: format, to: event.turnID, at: now) { previous in
                ((previous?.raw ?? "") + delta, true)
            }
            if case .working = state {} else { state = .responding }

        case .uiCompleted:
            let format = event.string("format")
            guard !format.isEmpty else { return }
            let content = event.string("content")
            writeUI(event, format: format, to: event.turnID, at: now) { previous in
                (content.isEmpty ? previous?.raw ?? "" : content, false)
            }

        case .responded:
            // The final text is authoritative: the deltas are what was being written, this is
            // what was said. An empty one adds nothing, which is what a spoken-only turn is.
            if !event.text.isEmpty {
                write(event.text, to: event.turnID, at: now) { $0 = event.text }
            }
            state = .idle

        case .delegated:
            state = .working(working + [event.string("skill")])

        case .taskSettled, .taskCancelled:
            let left = working.filter { $0 != event.string("skill") }
            state = left.isEmpty ? .responding : .working(left)

        case .interrupted:
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

    /// Writes the UI of the agent turn a `ui_delta` or `ui_completed` belongs to, starting the
    /// turn if the router sent UI for one we never saw begin. `next` gives what the turn wrote
    /// so far, from its UI before, and whether it is still streaming. A patch is merged onto
    /// the UI of the latest earlier turn in the same format.
    private mutating func writeUI(
        _ event: AgentEvent,
        format: String,
        to turnID: String,
        at now: Date,
        _ next: (TurnUI?) -> (raw: String, isStreaming: Bool)
    ) {
        let index = turns.lastIndex(where: { $0.id == turnID && $0.speaker.isAgent })
        let (raw, isStreaming) = next(index.flatMap { turns[$0].ui })
        var content = raw
        if event.string("mode") == "patch", format == TurnUI.openUI,
            let base = turns.last(where: {
                $0.speaker.isAgent && $0.id != turnID && $0.ui?.format == TurnUI.openUI
                    && $0.ui?.content.isEmpty == false
            })?.ui
        {
            content = TurnUI.merge(base.content, patch: raw)
        }
        let version = event.string("version")
        let ui = TurnUI(
            format: format, version: version.isEmpty ? nil : version, isStreaming: isStreaming,
            content: content, raw: raw)
        if let index {
            turns[index].ui = ui
        } else {
            turns.append(Turn(id: turnID, speaker: .agent, text: "", at: now, ui: ui))
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
    /// The UI to show. A patch, which changes only some statements of the turn before's OpenUI,
    /// is already merged onto it here.
    public var content: String
    /// What the turn itself wrote so far, before a patch was merged.
    var raw: String

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

    /// A program's statements: each line, or the lines a bracket or a string left open joined
    /// to the one that closes it. Blank and comment lines are left out.
    private static func openUIStatements(_ program: String) -> [String] {
        var statements: [String] = []
        var current = ""
        var quote: Character?
        var escaped = false
        var depth = 0
        for line in program.split(separator: "\n", omittingEmptySubsequences: false) {
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
        if !tail.isEmpty { statements.append(tail) }
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
