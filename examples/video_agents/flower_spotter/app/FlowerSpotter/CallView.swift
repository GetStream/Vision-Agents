import StreamChatAI
import SwiftUI
import VisionAgentsCore

/// Joins a call the Python worker already started, camera on.
struct CallView: View {
    let record: Session

    @State private var voice: VoiceSession?
    @State private var failure: String?

    var body: some View {
        VStack(spacing: 0) {
            if let voice {
                AgentVideoView(voice: voice)
                    .frame(maxHeight: .infinity)
                SpokenTranscript(session: voice.session)
                    .frame(maxHeight: 180)
                CallControls(voice: voice, camera: true)
                    .task { await voice.session.start() }
                    .padding(.vertical)
            } else if let failure {
                ContentUnavailableView(
                    "Could not join",
                    systemImage: "exclamationmark.triangle",
                    description: Text(failure))
            } else {
                ProgressView()
            }
        }
        .navigationTitle("Walk")
        .navigationBarTitleDisplayMode(.inline)
        .task(join)
        .onChange(of: voice?.session.state) { _, state in
            if state == .ended { voice = nil }
        }
        .onDisappear {
            let leaving = voice
            voice = nil
            Task { await leaving?.leave() }
        }
    }

    @Sendable private func join() async {
        guard voice == nil else { return }
        do {
            voice = try await VoiceSession.attach(
                agents: Demo.agents, sessionID: record.id)
        } catch is CancellationError {
            return
        } catch {
            failure = error.localizedDescription
        }
    }
}

/// What was said on the call, the agent's words streaming in as they are written, and what it
/// is doing while there is nothing to read yet: listening, or looking with the vision skill.
private struct SpokenTranscript: View {
    let session: AgentSession

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                ForEach(session.turns) { turn in
                    if turn.speaker.isAgent {
                        StreamingMessageView(
                            content: turn.text,
                            isGenerating: turn.id == session.turns.last?.id && session.state == .responding)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        Text(turn.text)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(.tint.opacity(0.15), in: .rect(cornerRadius: 16))
                            .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                }
                if let status {
                    AITypingIndicatorView(text: status)
                }
            }
            .padding(.horizontal)
        }
        .defaultScrollAnchor(.bottom)
    }

    private var status: String? {
        switch session.state {
        case .listening:
            return "Listening"
        case .responding where session.turns.last.map { !$0.speaker.isAgent || $0.text.isEmpty } ?? true:
            return "Thinking"
        case .working:
            return "Looking"
        default:
            return nil
        }
    }
}
