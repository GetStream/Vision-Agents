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
    /// What the person just asked, shown at once with its photos until Stream's copy of their
    /// message arrives: opening a conversation, uploading its photos and writing the message
    /// take seconds together.
    @State private var asking: Asked?
    /// The photos sent from this screen, by the message Stream keeps them on, so they are
    /// shown from the phone rather than after a download from Stream's CDN.
    @State private var sentPhotos: [String: [UIImage]] = [:]
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
                                message: message, photos: sentPhotos[message.id] ?? [],
                                reasoning: chat.reasoning, approver: approver(for: chat.session))
                        }
                    }
                    if let asking {
                        VStack(alignment: .trailing, spacing: 6) {
                            SentPhotos(photos: asking.photos)
                            PersonBubble(text: asking.text)
                        }
                        .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                    if let indicator {
                        AITypingIndicatorView(text: indicator)
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
                viewFactory: Composer(onTalk: talk),
                viewModel: composer,
                onMessageSend: { send($0) },
                onStopGenerating: { chat?.stop() })
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                GmailButton(gmail: gmail)
            }
        }
        .fullScreenCover(isPresented: $isVoiceShown) {
            if let chat {
                VoiceMode(chat: chat)
            }
        }
        .onChange(of: chat?.messages.count ?? 0) {
            // The person's first message after the ones there were when they asked is Stream's
            // copy of the question, which takes its place.
            guard let asked = asking,
                let arrived = chat?.messages.filter({ !$0.isAgentReply }).dropFirst(asked.after).first
            else { return }
            if !asked.photos.isEmpty {
                sentPhotos[arrived.id] = asked.photos
            }
            asking = nil
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

    /// What the agent is doing, shown under the last message. A question on its way shows
    /// "Thinking" before the session says anything.
    private var indicator: String? {
        chat?.indicator ?? (asking != nil ? "Thinking" : nil)
    }

    private func send(_ message: MessageData) {
        asking = Asked(
            text: message.text, photos: Photos.thumbnails(for: message.attachments),
            after: chat?.messages.count(where: { !$0.isAgentReply }) ?? 0)
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

    /// Switches to voice mode on this conversation, starting a session for a new one.
    private func talk() {
        Task {
            guard await open() != nil else { return }
            isVoiceShown = true
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

/// A question on its way, as the person sees it before Stream's copy of it arrives.
private struct Asked {
    let text: String
    let photos: [UIImage]
    /// How many of the person's messages the conversation held when it was asked: the next
    /// one is this question.
    let after: Int
}

/// The composer with the way into voice mode beside the message box.
private final class Composer: AIComposerViewFactory {
    let onTalk: () -> Void

    init(onTalk: @escaping () -> Void) {
        self.onTalk = onTalk
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
