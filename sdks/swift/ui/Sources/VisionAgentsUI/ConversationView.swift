import SwiftUI
import VisionAgentsCore

/// A whole conversation: the transcript, what the agent is doing, and somewhere to type.
///
/// The three parts below are public and work on their own, so a host that wants a different
/// arrangement can take them apart rather than fight this. This is the arrangement most apps
/// want, and it opens the socket when it appears and closes it when it goes away.
public struct ConversationView: View {
    private let session: AgentSession

    public init(session: AgentSession) {
        self.session = session
    }

    public var body: some View {
        VStack(spacing: 0) {
            TranscriptView(turns: session.turns, state: session.state)
                .frame(maxHeight: .infinity)

            if let failure = session.failure {
                Text(failure.localizedDescription)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
            }

            HStack {
                AgentStatusView(state: session.state)
                Spacer()
            }
            .padding(.horizontal)

            Composer(
                isEnabled: session.isConnected,
                isGenerating: isGenerating,
                send: { try? await session.send($0) },
                stop: { try? await session.interrupt() }
            )
        }
        .task {
            await session.start()
        }
    }

    /// Answering and thinking are both a reply in flight, and both are what the composer's
    /// stop button abandons.
    private var isGenerating: Bool {
        switch session.state {
        case .responding, .working: return true
        case .idle, .listening, .ended: return false
        }
    }
}
