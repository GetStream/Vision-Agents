import StreamChat
import StreamChatAI
import SwiftUI
import VisionAgentsCore

/// One conversation: the messages, the composer, and the way into voice mode.
///
/// A new conversation has no session until its first message, so opening the app or tapping
/// New chat leaves nothing in the list. A conversation from the list is followed again by its
/// id, and one that ended carries on when it is next asked something.
struct ChatScreen: View {
    /// The conversation to carry on, or nil for a new one.
    let sessionID: String?
    /// Says which session a new conversation became, once it has one.
    let onStarted: (String) -> Void

    @Environment(GmailAccount.self) private var gmail
    @State private var chat: AgentChat?
    @State private var failure: String?
    /// What the person just asked, shown at once while the conversation is opened and the
    /// router writes it into the channel, which together take a few seconds.
    @State private var asking: String?
    @State private var isVoiceShown = false
    @StateObject private var composer = AIComposerViewModel()

    private static let starters = [
        "What's on my calendar today?",
        "Remind me to call mum at 6pm",
        "Plan a weekend in Lisbon",
        "What should I cook tonight?",
    ]

    private static let thinkHarder = "think_harder"

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    if let chat {
                        ForEach(chat.messages, id: \.id) { message in
                            MessageRow(
                                message: message, reasoning: chat.reasoning,
                                approver: approver(for: chat.session))
                        }
                        if let indicator = chat.indicator {
                            AITypingIndicatorView(text: indicator)
                        }
                    }
                    if let asking, isEmpty {
                        PersonBubble(text: asking)
                        if chat?.indicator == nil {
                            AITypingIndicatorView(text: "Thinking")
                        }
                    }
                }
                .padding()
            }
            .defaultScrollAnchor(.bottom)
            .scrollDismissesKeyboard(.interactively)
            .overlay {
                if isWelcoming {
                    Welcome()
                }
            }

            if isWelcoming {
                SuggestionsView(suggestions: Self.starters) { send($0) }
            }
            if let failure = failure ?? chat?.failure ?? chat?.session.conversation.failure {
                Text(failure)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
            }
            AIComposerView(
                viewFactory: Composer(gmail: gmail),
                viewModel: composer,
                onMessageSend: { send($0) },
                onStopGenerating: { chat?.stop() })
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    Task {
                        guard await open() != nil else { return }
                        isVoiceShown = true
                    }
                } label: {
                    Image(systemName: "waveform")
                }
                .accessibilityLabel("Talk to Juno")
            }
        }
        .fullScreenCover(isPresented: $isVoiceShown) {
            if let chat {
                VoiceMode(chat: chat)
            }
        }
        .onChange(of: chat?.isAnswering ?? false, initial: true) { _, answering in
            composer.isGenerating = answering
        }
        .onAppear {
            // An option's own action is what selects it: the list only runs it.
            let model = composer
            model.chatOptions = [
                ChatOption(
                    id: Self.thinkHarder, title: "Think harder",
                    description: "More time to reason before answering", icon: "brain",
                    shortTitle: "Think"
                ) { [weak model] in
                    model?.selectedChatOption = model?.chatOptions.first {
                        $0.id == Self.thinkHarder
                    }
                    model?.sheetShown = false
                }
            ]
        }
        .task {
            guard sessionID != nil else { return }
            await open()
        }
        .onDisappear {
            let closing = chat
            chat = nil
            Task { await closing?.close() }
        }
    }

    private var isEmpty: Bool {
        chat?.messages.isEmpty ?? true
    }

    private var isWelcoming: Bool {
        isEmpty && asking == nil
    }

    private func send(_ message: MessageData) {
        if isEmpty { asking = message.text }
        Task {
            guard let chat = await open() else {
                asking = nil
                return
            }
            chat.ask(
                message.text, photos: message.attachments,
                thinkHarder: message.chatOption?.id == Self.thinkHarder)
        }
    }

    /// The conversation's chat, starting a session for a new one.
    @discardableResult
    private func open() async -> AgentChat? {
        if let chat { return chat }
        failure = nil
        do {
            let session: AgentSession
            if let sessionID {
                session = try await Assistant.agents.attach(sessionID: sessionID, tools: Assistant.tools)
            } else {
                var options = SessionOptions(agent: Assistant.agentName)
                options.llm = Assistant.chatModel
                options.tools = Assistant.tools
                session = try await Assistant.agents.chat(options)
                onStarted(session.id)
            }
            let opened = try await AgentChat.open(session)
            chat = opened
            return opened
        } catch is CancellationError {
            return nil
        } catch {
            failure = error.localizedDescription
            return nil
        }
    }

    /// Answers the questions of the tools that ask first, as the person signed in here.
    private func approver(for session: AgentSession) -> AIToolApprover {
        AIToolApprover(userID: Assistant.user.id, clientID: AIClientIdentity.installID) {
            call, allowed in
            try await session.decide(call.id, allowed: allowed)
        }
    }
}

/// The composer with the Gmail button beside the message box.
private final class Composer: AIComposerViewFactory {
    let gmail: GmailAccount

    init(gmail: GmailAccount) {
        self.gmail = gmail
    }

    func makeTrailingComposerView(options: AIComposerTrailingViewOptions) -> some View {
        GmailButton(gmail: gmail)
    }
}

/// What a new conversation shows before anything is said.
private struct Welcome: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "sparkles")
                .font(.system(size: 44))
                .foregroundStyle(.tint)
            Text(greeting)
                .font(.title2.weight(.semibold))
            Text("Ask me anything, show me a photo, or tap the waveform to talk.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(32)
    }

    private var greeting: String {
        let hour = Calendar.current.component(.hour, from: .now)
        let part = hour < 12 ? "morning" : hour < 18 ? "afternoon" : "evening"
        return "Good \(part), \(Assistant.user.name)"
    }
}
