import Foundation

/// One line of a conversation.
public struct Turn: Sendable, Hashable, Identifiable {
    /// The router's turn id for an agent turn, and one of our own for a participant's.
    public let id: String
    public let speaker: Speaker
    public var text: String
    public let at: Date

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

        case .responded:
            // The final text is authoritative: the deltas are what was being written, this is
            // what was said. An empty one adds nothing, which is what a spoken-only turn is.
            if !event.text.isEmpty {
                write(event.text, to: event.turnID, at: now) { $0 = event.text }
            }
            state = .idle

        case .delegated:
            state = .working(working + [event["skill"].stringValue])

        case .taskSettled, .taskCancelled:
            let left = working.filter { $0 != event["skill"].stringValue }
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
}
