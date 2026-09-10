import StreamVideo
import StreamVideoSwiftUI
import SwiftUI
import UIKit

/// The device camera, with the agent's annotated track inset beside it.
///
/// There is no overlay drawing here: bounding boxes are already burned into the agent's
/// published track. That track has crossed the network to the processor and back, so it
/// lags the camera by however long a round trip takes and is inset rather than filling the
/// frame. What fills the frame is this device's own camera, which is immediate. A call where
/// the device publishes nothing shows the agent's track full-frame instead.
public struct AgentVideoView: View {
    private let voice: VoiceSession

    public init(voice: VoiceSession) {
        self.voice = voice
    }

    public var body: some View {
        if let call = voice.call {
            CallCanvas(call: call)
        } else {
            WaitingForVideo()
        }
    }
}

private struct WaitingForVideo: View {
    var body: some View {
        ZStack {
            Color.black
            Text("waiting for video")
                .font(.caption)
                .foregroundStyle(.white)
        }
    }
}

private struct CallCanvas: View {
    let call: Call
    @ObservedObject var state: CallState

    init(call: Call) {
        self.call = call
        self.state = call.state
    }

    var body: some View {
        GeometryReader { geometry in
            let frame = geometry.frame(in: .local)
            ZStack {
                // A renderer handed a zero frame logs "invalid setDrawableSize" and draws
                // nothing, which is what the first layout pass offers.
                if frame.isEmpty {
                    WaitingForVideo()
                } else if let local = localVideo {
                    camera(local, in: frame)
                    if let remote = remoteVideo {
                        inset(remote)
                    }
                } else if let remote = remoteVideo {
                    video(remote, in: frame)
                } else {
                    WaitingForVideo()
                }
            }
        }
    }

    /// This device's own camera.
    ///
    /// `LocalVideoView` rather than the participant view the remote track uses: the local
    /// track is registered against `"{userID}-local"` rather than against the participant's
    /// id, so a participant view asked for it renders an empty layer.
    private func camera(_ participant: CallParticipant, in frame: CGRect) -> some View {
        LocalVideoView(
            viewFactory: DefaultViewFactory.shared,
            participant: participant,
            callSettings: state.callSettings,
            call: call,
            availableFrame: frame
        )
    }

    private func video(_ participant: CallParticipant, in frame: CGRect) -> some View {
        VideoCallParticipantView(
            viewFactory: DefaultViewFactory.shared,
            participant: participant,
            id: participant.id,
            availableFrame: frame,
            contentMode: UIView.ContentMode.scaleAspectFill,
            customData: [:],
            call: call
        )
    }

    private func inset(_ participant: CallParticipant) -> some View {
        VideoCallParticipantView(
            viewFactory: DefaultViewFactory.shared,
            participant: participant,
            id: participant.id,
            availableFrame: CGRect(x: 0, y: 0, width: 180, height: 240),
            contentMode: UIView.ContentMode.scaleAspectFill,
            customData: [:],
            call: call
        )
        .frame(width: 180, height: 240)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
        .padding()
    }

    /// `callSettings.videoOn` rather than the participant's `hasVideo`: the camera being on
    /// is what this device decided, and it is true from the moment it is turned on rather
    /// than once a track has been negotiated and reported back.
    private var localVideo: CallParticipant? {
        guard state.callSettings.videoOn else { return nil }
        return state.localParticipant
    }

    /// The video worker joins as `{agent_user_id}-video`; anyone else with video is a
    /// fallback for a call where the annotated track has not arrived yet.
    private var remoteVideo: CallParticipant? {
        if let worker = state.remoteParticipants.first(where: {
            $0.userId.hasSuffix("-video") && $0.hasVideo
        }) {
            return worker
        }
        return state.remoteParticipants.first { $0.hasVideo }
    }
}
