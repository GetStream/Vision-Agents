import AGUI
import SwiftUI
import VisionAgentsCore

/// A whole conversation: the transcript, what the agent is doing, what it is waiting to be
/// allowed to do, and somewhere to type.
///
/// The four parts below are public and work on their own, so a host that wants a different
/// arrangement can take them apart rather than fight this. This is the arrangement most apps
/// want, and it opens the socket when it appears and closes it when it goes away.
public struct ConversationView: View {
    private let session: AgentSession
    private let approval: (Interrupt) -> AnyView

    public init(session: AgentSession) {
        self.init(session: session) { approval in
            ApprovalView(
                approval: approval,
                approve: { try? await session.approve(approval) },
                decline: { try? await session.decline(approval) })
        }
    }

    /// - Parameters:
    ///   - session: the conversation.
    ///   - approval: how to ask for one thing the agent wants allowed. Omit it for the
    ///     built-in card. Answer it with `session.approve(_:)` or `session.decline(_:reason:)`.
    public init<Approval: View>(
        session: AgentSession,
        @ViewBuilder approval: @escaping (Interrupt) -> Approval
    ) {
        self.session = session
        self.approval = { AnyView(approval($0)) }
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

            // Above the composer rather than in the transcript: what it asks for is the next
            // thing to happen, and a card that scrolled away with the conversation would be
            // a question nobody can find.
            ForEach(session.pendingApprovals) { pending in
                approval(pending)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
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
        .animation(.default, value: session.pendingApprovals.map(\.id))
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
