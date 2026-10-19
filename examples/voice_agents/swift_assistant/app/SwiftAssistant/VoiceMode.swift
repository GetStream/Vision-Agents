import StreamChatAI
import StreamVideo
import StreamVideoSwiftUI
import StreamWebRTC
import SwiftUI
import VisionAgentsCore

/// Talking to Juno, full screen, in the conversation that was open.
///
/// Voice is started on the chat's own session, so the agent joins its call with everything
/// said so far, and what is said on the call is written into the same channel: the chat shows
/// it once voice mode is closed. The chat's socket stays open the whole time and is what runs
/// this phone's tools, so the call is followed with a second, tool-less session of its own.
///
/// The camera button shows Juno what the phone is pointing at: the call's session answers its
/// vision skill's requests to look with what the camera showed, and the preview is that view.
struct VoiceMode: View {
    let chat: AgentChat

    @Environment(\.dismiss) private var dismiss
    @State private var voice: VoiceSession?
    @State private var failure: String?

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Spacer()
                Button {
                    Task { await hangUp() }
                } label: {
                    Image(systemName: "xmark")
                        .font(.title3.weight(.semibold))
                        .padding(12)
                        .background(.secondary.opacity(0.15), in: .circle)
                }
                .accessibilityLabel("Back to chat")
            }
            .padding(.horizontal)

            if let call = voice?.call, voice?.isCameraEnabled == true {
                CameraPreview(state: call.state)
            }

            AgentPresence(voice: voice, isStarting: voice == nil && failure == nil)

            if let request = chat.session.approvals.first {
                ApprovalCard(request: request, session: chat.session)
                    .id(request.id)
                    .padding(.horizontal)
            }
            if let failure = failure ?? voice?.failure?.localizedDescription {
                Text(failure)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
            if let voice {
                CallControls(voice: voice) {
                    Task { await hangUp() }
                }
                .padding(.bottom)
            }
        }
        .task { await start() }
        .onChange(of: voice?.session.state) { _, state in
            if state == .ended { dismiss() }
        }
    }

    private func start() async {
        do {
            try await chat.session.startVoice()
            let voice = try await VoiceSession.attach(
                agents: Assistant.agents, sessionID: chat.session.id)
            self.voice = voice
            await voice.session.start()
            await voice.join()
        } catch is CancellationError {
        } catch {
            failure = error.localizedDescription
        }
    }

    /// Leaves the call and takes the agent off it, which carries the conversation on in
    /// writing. Leaving an attached call keeps the session, and so does not closing the call's
    /// own `AgentSession`: closing one ends the session, the chat's included.
    private func hangUp() async {
        await voice?.leave()
        _ = try? await chat.session.stopVoice()
        dismiss()
    }
}

/// What the camera shows, which is what Juno sees when it looks.
private struct CameraPreview: View {
    @ObservedObject var state: CallState

    var body: some View {
        GeometryReader { proxy in
            if let track = state.localParticipant?.track {
                VideoRendererView(id: "juno-camera", size: proxy.size) { $0.add(track: track) }
            }
        }
        .background(.secondary.opacity(0.15))
        .clipShape(.rect(cornerRadius: 20))
        .frame(height: 260)
        .padding(.horizontal)
        .accessibilityLabel("What Juno sees")
    }
}

/// The camera, mute, and back to the chat.
private struct CallControls: View {
    let voice: VoiceSession
    let onHangUp: () -> Void

    var body: some View {
        HStack(spacing: 24) {
            Button {
                Task { await voice.setCameraEnabled(!voice.isCameraEnabled) }
            } label: {
                Image(systemName: voice.isCameraEnabled ? "video.fill" : "video.slash.fill")
                    .font(.title2)
                    .frame(width: 64, height: 64)
            }
            .buttonStyle(.bordered)
            .clipShape(.circle)
            .disabled(voice.call == nil)
            .accessibilityLabel(voice.isCameraEnabled ? "Stop showing Juno" : "Show Juno the camera")

            Button {
                Task { await voice.setMuted(!voice.isMuted) }
            } label: {
                Image(systemName: voice.isMuted ? "mic.slash.fill" : "mic.fill")
                    .font(.title2)
                    .frame(width: 64, height: 64)
            }
            .buttonStyle(.bordered)
            .clipShape(.circle)
            .disabled(voice.call == nil)

            Button(action: onHangUp) {
                Image(systemName: "phone.down.fill")
                    .font(.title2)
                    .frame(width: 64, height: 64)
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
            .clipShape(.circle)
        }
    }
}

/// The glow, moving with whoever is talking, a word for what the agent is doing, and the
/// captions.
///
/// While the agent speaks, the glow moves with its voice in Stream's colors; while the person
/// holding the phone speaks, with theirs in `Self.personColors`, crossfading between the two.
/// Its own view so that the levels, read thirty times a second, redraw only this.
private struct AgentPresence: View {
    let voice: VoiceSession?
    let isStarting: Bool

    /// The colors the glow takes while the person is talking. The agent's are the
    /// appearance's, which is Stream's accent.
    private static let personColors = [Color(hue: 0, saturation: 0.85, brightness: 1)]

    /// How long after the agent was last heard the microphone is still ignored, so what is left
    /// of its voice after echo cancellation never reads as the person.
    private static let echoTail: TimeInterval = 0.3

    @State private var agentMeter = VoiceMeter(audible: 0.04)
    /// Higher, and held a moment, so room noise does not count as the person talking.
    @State private var personMeter = VoiceMeter(audible: 0.1, onset: 0.12)
    /// Who is talking and how loud, or nil while nobody is. This comes from the audio rather
    /// than the session: a reply is written before it is spoken and is done before the agent
    /// stops talking, and the person is transcribed only after they have started.
    @State private var heard: Heard?
    @State private var captions = Captions()

    private enum Speaker {
        case agent, person
    }

    private struct Heard: Equatable {
        var speaker: Speaker
        var level: Double
    }

    var body: some View {
        VStack(spacing: 16) {
            Spacer()
            AIVoiceGlowView(
                state: state,
                colors: heard?.speaker == .person ? Self.personColors : nil,
                audioLevel: heard?.level)
                .frame(maxWidth: 320, maxHeight: 320)
                .aspectRatio(1, contentMode: .fit)
            Text(status)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .contentTransition(.opacity)
                .animation(.easeInOut(duration: 0.2), value: status)
            Spacer()
            CaptionView(caption: captions.caption, isVisible: captions.isVisible)
        }
        .task(id: voice?.call?.cId) {
            guard let call = voice?.call else { return }
            // The microphone is metered as the call's audio filter, which reads each frame
            // after WebRTC's processing and leaves it as it was.
            call.setAudioFilter(personMeter.filter)
            // The agent is the call's other participant. Its track can change when the call
            // resubscribes, so the meter follows whichever one is current.
            var metered: RTCAudioTrack?
            defer {
                metered?.remove(agentMeter.renderer)
                call.setAudioFilter(nil)
            }
            for await participants in call.state.$remoteParticipants.values {
                let track = participants.first(where: \.hasAudio)?.audioTrack
                guard track !== metered else { continue }
                metered?.remove(agentMeter.renderer)
                track?.add(agentMeter.renderer)
                metered = track
            }
        }
        .task(id: voice?.call?.cId) {
            heard = nil
            captions = Captions()
            guard let voice, voice.call != nil else { return }
            var agentLastHeard = -TimeInterval.infinity
            var last = ProcessInfo.processInfo.systemUptime
            while !Task.isCancelled {
                let now = ProcessInfo.processInfo.systemUptime
                let agent = agentMeter.read()
                let person = personMeter.read()
                // The agent first: while its voice plays, the microphone hears it too.
                if let agent {
                    agentLastHeard = now
                    heard = Heard(speaker: .agent, level: agent)
                } else if let person, now - agentLastHeard > Self.echoTail {
                    heard = Heard(speaker: .person, level: person)
                } else {
                    heard = nil
                }
                captions.update(
                    turn: voice.session.turns.last,
                    agentIsAudible: agent != nil,
                    agentIsAnswering: isAnswering(voice.session.state),
                    now: now,
                    elapsed: now - last)
                last = now
                try? await Task.sleep(for: .milliseconds(33))
            }
        }
    }

    private func isAnswering(_ state: Conversation.State) -> Bool {
        switch state {
        case .responding, .working: true
        default: false
        }
    }

    private var state: AIVoiceAgentState {
        guard let voice else { return isStarting ? .connecting : .idle }
        guard voice.call != nil else { return .connecting }
        switch heard?.speaker {
        case .person: return .listening
        case .agent: return .speaking
        case nil: break
        }
        switch voice.session.state {
        case .listening:
            return .listening
        case .responding, .working:
            return .thinking
        case .idle, .ended:
            return .idle
        }
    }

    /// What the glow is showing, in words. The line keeps its height so the glow does not
    /// move as it changes.
    private var status: String {
        guard let voice else { return isStarting ? "Connecting" : " " }
        guard voice.call != nil else { return "Connecting" }
        switch heard?.speaker {
        case .person: return "Listening"
        case .agent: return "Speaking"
        case nil: break
        }
        switch voice.session.state {
        case .listening:
            return "Listening"
        case .responding:
            return "Thinking"
        case .working(let skills):
            return "Working on " + skills.map { $0.replacingOccurrences(of: "_", with: " ") }
                .joined(separator: " and ")
        case .idle, .ended:
            return "Ready"
        }
    }
}

/// A tool's question, asked over the call with Stream's approval card. The tool runs only once
/// it is allowed, and the agent is told when it is not.
private struct ApprovalCard: View {
    let request: ToolApprovalRequest
    let session: AgentSession

    @State private var state = AIToolApprovalState()

    var body: some View {
        AIToolApprovalCard(
            approval: AIToolApproval(
                title: request.approval.title,
                message: request.approval.message,
                reason: request.reason.isEmpty ? nil : request.reason,
                allowTitle: request.approval.allowTitle,
                declineTitle: request.approval.declineTitle),
            state: state
        ) { allowed in
            state = AIToolApprovalState(isSending: true)
            Task {
                do {
                    try await session.decide(request.id, allowed: allowed)
                } catch {
                    state = AIToolApprovalState(failed: true)
                }
            }
        }
    }
}
