import SwiftUI
import VisionAgentsCore

/// A whole conversation: the transcript, what the agent is doing, and somewhere to type.
///
/// The three parts below are public and work on their own, so a host that wants a different
/// arrangement can take them apart rather than fight this. This is the arrangement most apps
/// want, and it opens the socket when it appears and closes it when it goes away.
public struct ConversationView: View {
    private let session: AgentSession
    private let prompt: String

    public init(session: AgentSession, prompt: String = "Message") {
        self.session = session
        self.prompt = prompt
    }

    public var body: some View {
        VStack(spacing: 0) {
            TranscriptView(turns: session.turns)
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

            Composer(prompt: prompt, isEnabled: session.isConnected) { text in
                try? await session.send(text)
            }
            .padding()
        }
        .task {
            await session.start()
        }
    }
}
