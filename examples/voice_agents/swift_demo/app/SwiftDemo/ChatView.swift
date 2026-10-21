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
            // The person's messages in bubbles, the agent's as their steps and then their answer.
            // Each question goes to the top as it is asked, and its reply writes itself out below
            // it without moving what is on screen. While the agent thinks, the list shows its
            // thinking, which the reply's first step takes over once it arrives.
            AIConversationView(
                messages: chat.listMessages,
                isThinking: chat.activity == .thinking,
                approver: approver,
                loadOlderMessages: chat.hasOlderMessages ? { await chat.loadOlderMessages() } : nil,
                viewFactory: MessageViews(chat: chat),
                footer: {
                    if case .working(let text) = chat.activity {
                        AITypingIndicatorView(text: text)
                    }
                },
                emptyContent: {
                    ContentUnavailableView(
                        "Ask about an order", systemImage: "shippingbox",
                        description: Text("Orders A-1042 and A-1043 are on this account."))
                })

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

/// Stream's message views, with what the person can do under each finished reply: copy it,
/// share it, and rate it. Asking for a reply again needs the router to replace one, which it
/// does not yet.
private final class MessageViews: AIConversationViewFactory {
    let chat: AgentChat?

    init(chat: AgentChat?) {
        self.chat = chat
    }

    @ViewBuilder
    func makeReplyActionsView(options: AIReplyActionsViewOptions) -> some View {
        if let chat, let reply = options.message as? AgentMessage {
            ReplyActions(chat: chat, reply: reply)
        }
    }
}

/// The buttons under a reply. The rating is read here, from the chat, so the thumbs change as
/// soon as it does: the list keeps a reply's row while the reply stays the same.
private struct ReplyActions: View {
    let chat: AgentChat
    let reply: AgentMessage

    var body: some View {
        AIReplyActionsView(
            text: reply.text,
            actions: AIReplyActions(
                rating: chat.ratings[reply.id],
                onRate: { rating in chat.rate(reply.message, rating) }))
    }
}

/// The composer without its attachment button: a question goes to the agent as text, and
/// dictating one stays.
private final class Composer: AIComposerViewFactory {
    func makeLeadingComposerView(options: AIComposerLeadingViewOptions) -> some View {
        EmptyView()
    }
}
