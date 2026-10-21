import StreamChatAI
import StreamVideo
import StreamWebRTC
import SwiftUI
import VisionAgentsCore

/// A spoken conversation, with the agent shown as Stream's voice glow and what is said captioned
/// underneath.
///
/// Tapping talk does two things: the router starts a session, which puts the agent on a call,
/// and this device joins it as the user `setUser` named, with the same token. The glow shows
/// what the agent is doing: connecting while the call is joined, thinking while it answers,
/// and moving with whoever is talking. A tool that asks first asks here, over the call, and
/// the agent waits for the answer.
struct VoiceView: View {
    let agent: String
    let style: VoiceStyle

    @State private var voice: VoiceSession?
    @State private var failure: String?
    @State private var isStarting = false

    var body: some View {
        VStack(spacing: 16) {
            AgentPresence(voice: voice, style: style, isStarting: isStarting)

            if let voice {
                if let request = voice.session.approvals.first {
                    ApprovalCard(request: request, session: voice.session)
                        .id(request.id)
                        .padding(.horizontal)
                }
                CallControls(voice: voice)
                    .task { await voice.session.start() }
                    .padding(.bottom)
            } else {
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
                .padding(.bottom)
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
                    agents: Demo.agents, agent: agent, tools: Demo.tools)
            } catch is CancellationError {
            } catch {
                failure = error.localizedDescription
            }
            isStarting = false
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
    let style: VoiceStyle
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
            VoiceStyleView(
                style: style,
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

    /// What the glow is showing, in words. Nothing before a call, but the line keeps its
    /// height so the glow does not move when a call starts.
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

/// How the agent is shown on a call: Stream's voice glow, an orb, or its voice wave.
enum VoiceStyle: CaseIterable, Identifiable {
    case orb, wave

    var id: Self { self }

    var label: String {
        switch self {
        case .orb: "Orb"
        case .wave: "Wave"
        }
    }
}

/// The agent drawn in `style`.
private struct VoiceStyleView: View {
    let style: VoiceStyle
    let state: AIVoiceAgentState
    var colors: [Color]?
    var audioLevel: Double?

    var body: some View {
        switch style {
        case .orb: AIVoiceGlowView(state: state, colors: colors, audioLevel: audioLevel)
        case .wave: AIVoiceWaveView(state: state, colors: colors, audioLevel: audioLevel)
        }
    }
}

/// Settings for the Voice tab, at the top right: the orb and the wave, each moving as it would
/// on a call, to pick how the agent is shown.
struct VoiceStyleSheet: View {
    @Binding var style: VoiceStyle
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Voice style")
                .font(.title2)
            HStack(spacing: 16) {
                ForEach(VoiceStyle.allCases) { option in
                    Button {
                        style = option
                        dismiss()
                    } label: {
                        VStack(spacing: 8) {
                            VoiceStyleView(style: option, state: .speaking)
                                .aspectRatio(1, contentMode: .fit)
                            Label(option.label, systemImage: "checkmark.circle.fill")
                                .labelStyle(StyleLabel(isSelected: option == style))
                                .font(.subheadline.weight(.medium))
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity)
                        .background(.background, in: .rect(cornerRadius: 12))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12)
                                .strokeBorder(
                                    option == style ? Color.accentColor : Color.secondary.opacity(0.3),
                                    lineWidth: option == style ? 2 : 1)
                        }
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(option == style ? .isSelected : [])
                }
            }
        }
        .padding(24)
    }
}

/// A check before the selected style's name, and only its name otherwise.
private struct StyleLabel: LabelStyle {
    let isSelected: Bool

    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 6) {
            if isSelected {
                configuration.icon.foregroundStyle(Color.accentColor)
            }
            configuration.title
        }
    }
}
