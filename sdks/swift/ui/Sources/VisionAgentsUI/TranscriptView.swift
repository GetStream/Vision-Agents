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

    public init(turns: [Turn]) {
        self.init(turns: turns) { TurnBubble(turn: $0) }
    }

    public var body: some View {
        ScrollViewReader { scroll in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(turns) { turn in
                        bubble(turn).id(turn.id)
                    }
                    // Scrolling to a zero-height anchor rather than to the last turn keeps the
                    // bottom in view while that turn is still growing.
                    Color.clear.frame(height: 1).id(bottom)
                }
                .padding()
            }
            // A reply arrives several times a second, and animating each delta is what makes a
            // transcript judder while it is being written. Growing text follows the bottom
            // without animating; a whole new line is worth animating to.
            .onChange(of: turns.last?.text) { _, _ in
                scroll.scrollTo(bottom, anchor: .bottom)
            }
            .onChange(of: turns.count) { _, _ in
                withAnimation(.easeOut(duration: 0.15)) { scroll.scrollTo(bottom, anchor: .bottom) }
            }
        }
    }

    private var bottom: String { "vision-agents.transcript.bottom" }
}

/// One line of the conversation.
public struct TurnBubble: View {
    public let turn: Turn

    public init(turn: Turn) {
        self.turn = turn
    }

    public var body: some View {
        VStack(alignment: turn.speaker.isAgent ? .leading : .trailing, spacing: 3) {
            if case .participant(let who) = turn.speaker, let name = who?.display, !name.isEmpty {
                Text(name)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(turn.text)
                .textSelection(.enabled)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    turn.speaker.isAgent ? AnyShapeStyle(.quaternary) : AnyShapeStyle(.tint),
                    in: .rect(cornerRadius: 16))
                .foregroundStyle(turn.speaker.isAgent ? AnyShapeStyle(.primary) : AnyShapeStyle(Color.white))
        }
        .frame(
            maxWidth: .infinity,
            alignment: turn.speaker.isAgent ? .leading : .trailing)
    }
}
