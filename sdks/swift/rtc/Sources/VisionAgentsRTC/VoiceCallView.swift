import SwiftUI
import VisionAgentsCore

/// The controls for a spoken conversation: mute, camera, and hang up.
///
/// It draws no transcript. Showing what was said is the UI package's job, over the same
/// `session` the voice session holds, so a host can put the two together however it likes.
/// There is nothing here for playing the agent: joining the call is what does that.
public struct VoiceCallView: View {
    private let voice: VoiceSession
    private let camera: Bool

    public init(voice: VoiceSession, camera: Bool = false) {
        self.voice = voice
        self.camera = camera
    }

    public var body: some View {
        VStack(spacing: 16) {
            if let failure = voice.failure {
                Text(failure.localizedDescription)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            } else if voice.call == nil {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("joining").font(.caption).foregroundStyle(.secondary)
                }
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

                if camera {
                    Button {
                        Task { await voice.setCameraEnabled(!voice.isCameraEnabled) }
                    } label: {
                        Image(
                            systemName: voice.isCameraEnabled
                                ? "video.fill" : "video.slash.fill"
                        )
                        .font(.title2)
                        .frame(width: 56, height: 56)
                    }
                    .buttonStyle(.bordered)
                    .clipShape(.circle)
                    .disabled(voice.call == nil)
                }

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
        .task { await voice.join(camera: camera) }
    }
}
