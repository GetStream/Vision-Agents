import SwiftUI
import VisionAgentsCore
import VisionAgentsRTC
import VisionAgentsUI

/// A spoken conversation, with what was said written out as it happens.
///
/// Tapping talk does three things: the router starts a session, which puts the agent on a
/// call; the router mints a token for that call; and this device joins it. The transcript
/// underneath comes off the session socket rather than out of the call.
struct VoiceView: View {
    let agent: String

    @State private var voice: VoiceSession?
    @State private var failure: String?
    @State private var isStarting = false

    var body: some View {
        VStack(spacing: 16) {
            if let voice {
                TranscriptView(turns: voice.session.turns, state: voice.session.state)
                    .frame(maxHeight: .infinity)
                AgentStatusView(state: voice.session.state)
                VoiceCallView(voice: voice)
                    .task { await voice.session.start() }
                    .padding(.bottom)
            } else {
                Spacer()
                if let failure {
                    Text(failure)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                Button(action: start) {
                    Label("Talk to the agent", systemImage: "phone.fill")
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isStarting)
                Spacer()
            }
        }
        .onChange(of: voice?.session.state) { _, state in
            // The agent leaving is the call being over, so let go of it and offer to start
            // another rather than leaving dead controls on screen.
            if state == .ended { voice = nil }
        }
        .onDisappear {
            let leaving = voice
            voice = nil
            Task { await leaving?.leave() }
        }
    }

    private func start() {
        isStarting = true
        failure = nil
        Task {
            do {
                voice = try await VoiceSession.start(
                    agents: Demo.agents, agent: agent, tools: [Demo.lookupOrder])
            } catch is CancellationError {
            } catch {
                failure = error.localizedDescription
            }
            isStarting = false
        }
    }
}
