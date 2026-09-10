import SwiftUI
import VisionAgentsCore
import VisionAgentsRTC
import VisionAgentsUI

/// Joins a call the Python worker already started, camera on.
struct CallView: View {
    let record: CallRecord

    @State private var voice: VoiceSession?
    @State private var failure: String?

    var body: some View {
        VStack(spacing: 0) {
            if let voice {
                AgentVideoView(voice: voice)
                    .frame(maxHeight: .infinity)
                TranscriptView(turns: voice.session.turns)
                    .frame(maxHeight: 180)
                AgentStatusView(state: voice.session.state)
                VoiceCallView(voice: voice, camera: true)
                    .task { await voice.session.start() }
                    .padding(.bottom)
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
