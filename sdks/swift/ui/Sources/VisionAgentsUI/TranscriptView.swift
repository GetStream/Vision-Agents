import StreamChatAI
import SwiftUI
import VisionAgentsCore

/// The conversation, scrolling as it grows.
///
/// This is a `ScrollView`, not a `List`: appending to a list several times a second while a
/// reply streams in makes it rebuild rows it did not need to. It brings no navigation and no
/// colour scheme of its own, so it drops into whatever the host has.
public struct TranscriptView: View {
    private let turns: [Turn]
    private let bubble: (Turn) -> AnyView

    /// - Parameters:
    ///   - turns: the conversation, oldest first.
    ///   - bubble: how to draw one line. Omit it for the built-in bubble.
    public init<Bubble: View>(
        turns: [Turn],
        @ViewBuilder bubble: @escaping (Turn) -> Bubble
    ) {
        self.turns = turns
        self.bubble = { AnyView(bubble($0)) }
    }

    /// - Parameters:
    ///   - turns: the conversation, oldest first.
    ///   - state: what the agent is doing, which is how the turn being written now is told
    ///     from the ones already finished. Only that one animates in.
    public init(turns: [Turn], state: Conversation.State = .idle) {
        let writing = state == .responding ? turns.last?.id : nil
        self.init(turns: turns) { TurnBubble(turn: $0, isWriting: $0.id == writing) }
    }

    public var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
                ForEach(turns) { turn in
                    bubble(turn).id(turn.id)
                }
            }
            .padding()
        }
        // A reply arrives several times a second and is then written out a letter at a time,
        // so the bottom has to be held as the content grows rather than scrolled to when a
        // delta lands -- scrolling on the delta would leave the text growing off the screen.
        .defaultScrollAnchor(.bottom)
    }
}

/// One line of the conversation.
public struct TurnBubble: View {
    public let turn: Turn

    /// Whether the agent is still writing this turn, which is what makes it appear a letter
    /// at a time instead of whole.
    public let isWriting: Bool

    public init(turn: Turn, isWriting: Bool = false) {
        self.turn = turn
        self.isWriting = isWriting
    }

    public var body: some View {
        switch turn.speaker {
        case .agent:
            agent
        case .participant(let who):
            participant(who)
        }
    }

    /// The agent's turn is the page, not a bubble in it: what it says is markdown, and a code
    /// block or a table has a background and a width of its own that a bubble fights.
    private var agent: some View {
        StreamingMessageView(content: turn.text, isGenerating: isWriting)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func participant(_ who: Participant?) -> some View {
        VStack(alignment: .trailing, spacing: 3) {
            if let name = who?.display, !name.isEmpty {
                Text(name)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(turn.text)
                .textSelection(.enabled)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.tint, in: .rect(cornerRadius: 16))
                .foregroundStyle(Color.white)
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
    }
}
