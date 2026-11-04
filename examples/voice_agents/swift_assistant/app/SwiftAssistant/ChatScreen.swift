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
    /// Says which session a new conversation became, once it is first written or talked to.
    let onStarted: (String) -> Void

    /// The conversation's session and channel, opened when the conversation first needs them.
    @State private var chat: AgentChat
    /// What the conversation shows and what the person does in it. Replies are rated on this
    /// phone only, since the router takes no ratings yet, and are not written again or edited:
    /// the router keeps the conversation in Chat and cannot rewind it.
    @StateObject private var conversation: AIConversation<AgentMessage>
    /// Why the conversation could not be opened, or nil.
    @State private var failure: String?
    @State private var isVoiceShown = false

    private static let starters = [
        "What's on my calendar today?",
        "Remind me to call mum at 6pm",
        "Plan a weekend in Ohrid",
        "What should I cook tonight?",
    ]

    init(sessionID: String?, onStarted: @escaping (String) -> Void) {
        self.sessionID = sessionID
        self.onStarted = onStarted
        // SwiftUI keeps the first of each for as long as the screen shows, so the conversation
        // kept goes through the chat kept.
        let chat = AgentChat(sessionID: sessionID)
        _chat = State(initialValue: chat)
        _conversation = StateObject(
            wrappedValue: AIConversation(
                transport: chat, composerViewModel: Self.composer(), approver: chat.approver,
                actions: [.rate]))
    }

    var body: some View {
        // The person's messages in bubbles, the agent's as their steps and then their answer,
        // or the UI it wrote. What the person asks shows at once, with its photos, and Stream's
        // copy takes its place. Each question goes to the top as it is asked, and its reply
        // writes itself out below it without moving what is on screen. While the agent thinks,
        // the list shows its thinking, which the reply's first step takes over once it arrives.
        AIConversationView(
            conversation: conversation,
            ui: chat.ui,
            footer: {
                if conversation.status.isAnswering, let text = conversation.activity?.text {
                    AITypingIndicatorView(text: text)
                }
            },
            emptyContent: {
                Welcome()
            })
        .scrollDismissesKeyboard(.interactively)
        .modifier(ComposerBar(bar: composerBar))
        .fullScreenCover(isPresented: $isVoiceShown) {
            if let session = chat.session {
                VoiceMode(chat: chat, session: session)
            }
        }
        .task {
            guard sessionID != nil else { return }
            await open()
        }
        .onReceive(conversation.composerViewModel.$text) { text in
            // Opening a session takes seconds, so a new conversation's opens as its first
            // message is written rather than once it is sent.
            guard sessionID == nil, chat.session == nil, !text.isEmpty else { return }
            Task { await open() }
        }
        .onChange(of: chat.usedSessionID) { _, used in
            // A new conversation joins the list once it is written or talked to.
            guard sessionID == nil, let used else { return }
            onStarted(used)
        }
        .onDisappear {
            Task { await chat.close() }
        }
    }

    /// The starters, why asking failed, and the composer, over the bottom of the conversation.
    private var composerBar: some View {
        VStack(spacing: 0) {
            if isWelcoming {
                SuggestionsView(suggestions: Self.starters) { conversation.send($0) }
            }
            if let failure = failure ?? chat.failure ?? conversation.error?.localizedDescription
                ?? chat.sessionFailure
            {
                HStack(spacing: 8) {
                    Text(failure)
                        .foregroundStyle(.red)
                    if conversation.error != nil {
                        Button("Try again") { conversation.retry() }
                            .fontWeight(.semibold)
                    }
                }
                .font(.caption)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .modifier(Floating(shape: .capsule))
                .padding(.horizontal)
            }
            AIComposerView(conversation: conversation, viewFactory: Composer(onTalk: talk))
        }
    }

    private var isWelcoming: Bool {
        conversation.messages.isEmpty && conversation.pendingMessage == nil
    }

    /// The composer, whose options sheet has Think harder, which asks with a higher
    /// `reasoning_effort`. An option's own action is what selects it: the list only runs it.
    private static func composer() -> AIComposerViewModel {
        let model = AIComposerViewModel()
        model.chatOptions = [
            ChatOption(
                id: AgentChat.thinkHarder, title: "Think harder",
                description: "More time to reason before answering", icon: "brain",
                shortTitle: "Think"
            ) { [weak model] in
                model?.selectedChatOption = model?.chatOptions.first { $0.id == AgentChat.thinkHarder }
                model?.sheetShown = false
            }
        ]
        return model
    }

    /// Switches to voice mode on this conversation, starting a session for a new one.
    private func talk() {
        Task {
            if await open(use: true) != nil {
                isVoiceShown = true
            }
        }
    }

    /// The conversation's session, opened if it is not yet, and marked used when `use`. Says
    /// why when it cannot be opened.
    @discardableResult
    private func open(use: Bool = false) async -> AgentSession? {
        failure = nil
        do {
            return use ? try await chat.use() : try await chat.open()
        } catch is CancellationError {
            return nil
        } catch {
            failure = error.localizedDescription
            return nil
        }
    }
}

/// The composer with the way into voice mode beside the message box. Its button and its
/// field float over the conversation, so they are glass rather than filled.
private final class Composer: AIComposerViewFactory {
    let onTalk: () -> Void

    init(onTalk: @escaping () -> Void) {
        self.onTalk = onTalk
    }

    func makeLeadingComposerView(options: AIComposerLeadingViewOptions) -> some View {
        AddAttachmentsButton(onTap: options.onTap)
            .modifier(Floating(shape: .circle, isInteractive: true))
    }

    func makeComposerInputView(options: AIComposerInputViewOptions) -> some View {
        AIComposerInputView(
            viewModel: options.viewModel, speechHandler: options.speechHandler,
            onMessageSend: options.onMessageSend, onStopGenerating: options.onStopGenerating
        )
        .modifier(Floating(shape: .rect(cornerRadius: AIAppearance.shared.tokens.layout.radius3xl), isInteractive: true))
    }

    func makeTrailingComposerView(options: AIComposerTrailingViewOptions) -> some View {
        Button(action: onTalk) {
            Image(systemName: "waveform")
                .fontWeight(.semibold)
        }
        .buttonStyle(.borderedProminent)
        .buttonBorderShape(.circle)
        .accessibilityLabel("Talk to Juno")
    }
}

/// Floats the composer over the conversation, which scrolls on beneath it, while the list keeps
/// the latest message clear of it. Where the system has it, the conversation blurs away under
/// the composer as it does under the navigation bar, softly rather than at a hairline, so the
/// composer does not read as a bar of its own.
private struct ComposerBar<Bar: View>: ViewModifier {
    let bar: Bar

    func body(content: Content) -> some View {
        if #available(iOS 26, *) {
            content
                .safeAreaBar(edge: .bottom, spacing: 0) { bar }
                .scrollEdgeEffectStyle(.soft, for: .bottom)
        } else {
            content.safeAreaInset(edge: .bottom, spacing: 0) { bar }
        }
    }
}

/// What floats over the conversation lets it show through: glass where the system has it,
/// and a material before.
private struct Floating<S: Shape>: ViewModifier {
    let shape: S
    var isInteractive = false

    func body(content: Content) -> some View {
        if #available(iOS 26, *) {
            content.glassEffect(.regular.interactive(isInteractive), in: shape)
        } else {
            content.background(.regularMaterial, in: shape)
        }
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
