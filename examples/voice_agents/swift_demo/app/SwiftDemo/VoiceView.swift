import StreamChatAI
import SwiftUI
import VisionAgentsCore

/// A spoken conversation, with what was said written out as it happens.
///
/// Tapping talk does two things: the router starts a session, which puts the agent on a call,
/// and this device joins it as the user `setUser` named, with the same token. The transcript
/// underneath comes off the session socket rather than out of the call. A tool that asks
/// first asks here, over the call, and the agent waits for the answer.
struct VoiceView: View {
    let agent: String

    @State private var voice: VoiceSession?
    @State private var failure: String?
    @State private var isStarting = false

    var body: some View {
        VStack(spacing: 16) {
            if let voice {
                SpokenTranscript(session: voice.session)
                    .frame(maxHeight: .infinity)
                if let request = voice.session.approvals.first {
                    ApprovalCard(request: request, session: voice.session)
                        .id(request.id)
                        .padding(.horizontal)
                }
                CallControls(voice: voice)
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
                    agents: Demo.agents, agent: agent, tools: Demo.tools)
            } catch is CancellationError {
            } catch {
                failure = error.localizedDescription
            }
            isStarting = false
        }
    }
}

/// What was said on the call, the agent's words streaming in as they are written.
private struct SpokenTranscript: View {
    let session: AgentSession

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
                ForEach(session.turns) { turn in
                    if turn.speaker.isAgent {
                        StreamingMessageView(
                            content: turn.text,
                            isGenerating: turn.id == session.turns.last?.id && session.state == .responding)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        Text(turn.text)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(.tint.opacity(0.15), in: .rect(cornerRadius: 18))
                            .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                }
                if let status {
                    AITypingIndicatorView(text: status)
                }
            }
            .padding()
        }
        .defaultScrollAnchor(.bottom)
    }

    /// What the agent is doing while there is nothing of it to read yet.
    private var status: String? {
        switch session.state {
        case .listening:
            return "Listening"
        case .responding where session.turns.last.map { !$0.speaker.isAgent || $0.text.isEmpty } ?? true:
            return "Thinking"
        case .working(let skills):
            return "Working on " + skills.map { $0.replacingOccurrences(of: "_", with: " ") }
                .joined(separator: " and ")
        default:
            return nil
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
