import SwiftUI
import VisionAgentsCore

/// The controls for a spoken conversation: mute and hang up.
///
/// Whether the call is still being joined is the glow's to show, beside these. There is
/// nothing here for playing the agent: joining the call is what does that.
struct CallControls: View {
    let voice: VoiceSession

    var body: some View {
        VStack(spacing: 16) {
            if let failure = voice.failure {
                Text(failure.localizedDescription)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            }

            HStack(spacing: 24) {
                Button {
                    Task { await voice.setMuted(!voice.isMuted) }
                } label: {
                    Image(systemName: voice.isMuted ? "mic.slash.fill" : "mic.fill")
                        .font(.title2)
                        .frame(width: 56, height: 56)
                }
                .buttonStyle(.bordered)
                .clipShape(.circle)
                .disabled(voice.call == nil)

                Button {
                    Task { await voice.end() }
                } label: {
                    Image(systemName: "phone.down.fill")
                        .font(.title2)
                        .frame(width: 56, height: 56)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .clipShape(.circle)
            }
        }
        .task { await voice.join() }
    }
}
