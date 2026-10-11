import StreamChat
import StreamChatAI
import SwiftUI
import VisionAgentsCore

/// A written conversation.
///
/// Nothing is transcribed and nothing is spoken, so no call is joined. Everything between
/// hearing a question and answering it is the same as on a call: the same instructions, the
/// same knowledge, the same skills and the same tools running on this phone. What is new in
/// writing is that the reply shows its working, with Stream's AI components: the agent's
/// reasoning as it thinks, each tool call as it runs, and a question when a tool asks first.
struct ChatView: View {
    let agent: String

    @State private var chat: AgentChat?
    @State private var failure: String?

    var body: some View {
        Group {
            if let chat {
                AgentConversation(chat: chat)
            } else if let failure {
                ContentUnavailableView(
                    "Could not start", systemImage: "exclamationmark.triangle", description: Text(failure))
            } else {
                ProgressView()
            }
        }
        .task {
            guard chat == nil else { return }
            do {
                // A model that streams its reasoning, which is what fills the reply's thinking
                // panel. The call keeps the agent's own model, which answers sooner.
                var options = SessionOptions(agent: agent)
                options.llm = "xai/grok-4.7"
                options.tools = Demo.tools
                let session = try await Demo.agents.chat(options)
                chat = try await AgentChat.open(session)
            } catch is CancellationError {
                return
            } catch {
                failure = error.localizedDescription
            }
        }
        .onDisappear {
            let closing = chat
            chat = nil
            Task { await closing?.close() }
        }
    }
}

private struct AgentConversation: View {
    let chat: AgentChat

    @StateObject private var composer = AIComposerViewModel()

    private static let starters = [
        "Where is order A-1042?",
        "Can I return order A-1043?",
        "I'd like a refund for order A-1042",
    ]

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(chat.messages, id: \.id) { message in
                        MessageRow(message: message, reasoning: chat.reasoning, approver: approver)
                    }
                    if let indicator = chat.indicator {
                        AITypingIndicatorView(text: indicator)
                    }
                }
                .padding()
            }
            .defaultScrollAnchor(.bottom)
            .overlay {
                if chat.messages.isEmpty {
                    ContentUnavailableView(
                        "Ask about an order", systemImage: "shippingbox",
                        description: Text("Orders A-1042 and A-1043 are on this account."))
                }
            }

            if chat.messages.isEmpty {
                SuggestionsView(suggestions: Self.starters) { message in
                    chat.ask(message.text)
                }
            }
            if let failure = chat.failure {
                Text(failure)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
            }
            AIComposerView(
                viewFactory: Composer(),
                viewModel: composer,
                onMessageSend: { message in chat.ask(message.text) },
                onStopGenerating: { chat.stop() })
        }
        .onChange(of: chat.isAnswering, initial: true) { _, answering in
            composer.isGenerating = answering
        }
    }

    /// Answers the questions of the tools that ask first, as the person signed in here.
    ///
    /// The router addresses a question to the person whose message the call answers, and
    /// Stream's components show it only to them. The answer goes back over the session's
    /// socket, which is also what runs the tool once it is allowed.
    private var approver: AIToolApprover {
        let session = chat.session
        return AIToolApprover(userID: Demo.user.id, clientID: AIClientIdentity.installID) {
            call, allowed in
            try await session.decide(call.id, allowed: allowed)
        }
    }
}

/// One message: the person's in a bubble, the agent's as its steps and then its answer.
private struct MessageRow: View {
    let message: ChatMessage
    let reasoning: LiveReasoning
    let approver: AIToolApprover

    /// Whether the text is shown as still being written, which is what types it out.
    ///
    /// Stream folds a fast reply's live updates into one, so its text can arrive in the same
    /// update that says it is finished. `StreamingMessageView` types only what arrives while it
    /// is told the reply is being written, so it is told that until the text has reached it.
    @State private var writing: Bool

    init(message: ChatMessage, reasoning: LiveReasoning, approver: AIToolApprover) {
        self.message = message
        self.reasoning = reasoning
        self.approver = approver
        _writing = State(initialValue: message.isGenerating)
    }

    var body: some View {
        if message.isAgentReply {
            VStack(alignment: .leading, spacing: 8) {
                let parts = message.parts
                if !parts.isEmpty {
                    AIMessagePartsView(parts: parts) { part in
                        // The step keeps only the opening of its thinking; the whole of it
                        // came with the live updates, while the reply was being written.
                        if let step = part.reasoning {
                            StreamingReasoningView(part: step, text: reasoning[step.id])
                        } else {
                            AIMessagePartView(part: part, approver: approver)
                        }
                    }
                }
                if !message.text.isEmpty {
                    StreamingMessageView(content: message.text, isGenerating: writing)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .task(id: message.isGenerating) {
                guard !message.isGenerating else {
                    writing = true
                    return
                }
                try? await Task.sleep(for: .milliseconds(100))
                writing = false
            }
        } else {
            Text(message.text)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.tint.opacity(0.15), in: .rect(cornerRadius: 18))
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
    }
}

/// The composer without its attachment button: a question goes to the agent as text, and
/// dictating one stays.
private final class Composer: AIComposerViewFactory {
    func makeLeadingComposerView(options: AIComposerLeadingViewOptions) -> some View {
        EmptyView()
    }
}
