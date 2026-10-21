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
    /// Says which session a new conversation became, once it is first written or talked to.
    let onStarted: (String) -> Void

    @Environment(GmailAccount.self) private var gmail
    @State private var chat: AgentChat?
    @State private var failure: String?
    /// What the person just asked, shown at once with its photos until Stream's copy of their
    /// message arrives and takes its place: opening a conversation, uploading its photos and
    /// writing the message take seconds together.
    @State private var asking: Asked?
    @State private var isVoiceShown = false
    /// The session being opened, so asking while it opens waits for it rather than opening a
    /// second one.
    @State private var opening: Task<AgentChat?, Never>?
    /// Whether the list has been told about this new conversation.
    @State private var isAnnounced = false
    /// A new conversation's session, opened while the person wrote their first message and not
    /// used yet. Leaving without using it deletes it, so it is not left in the list.
    @State private var unused: String?
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
            // The person's messages in bubbles, the agent's as their steps and then their answer.
            // What the person asks shows at once, and Stream's copy takes its place, photos and
            // all, which keeps them from loading again. Each question goes to the top as it is
            // asked, and its reply writes itself out below it without moving what is on screen.
            // While the agent thinks, the list shows its thinking, which the reply's first step
            // takes over once it arrives.
            AIConversationView(
                messages: chat?.listMessages ?? [],
                pendingMessage: asking.map { asked in
                    AIPendingMessage(id: asked.id, text: asked.text, attachments: asked.photos.map { .image($0) })
                },
                isThinking: activity == .thinking,
                approver: chat.map { approver(for: $0.session) },
                loadOlderMessages: chat?.hasOlderMessages == true ? { await chat?.loadOlderMessages() } : nil,
                viewFactory: MessageViews(chat: chat),
                footer: {
                    if case .working(let text) = activity {
                        AITypingIndicatorView(text: text)
                    }
                },
                emptyContent: {
                    Welcome()
                })
                .scrollDismissesKeyboard(.interactively)

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
            if let asked = asking, copy(of: asked) != nil {
                asking = nil
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
        .onChange(of: composer.text) { _, text in
            // Opening a session takes seconds, so a new conversation's opens as its first
            // message is written rather than once it is sent.
            guard sessionID == nil, chat == nil, opening == nil, !text.isEmpty else { return }
            Task {
                if let chat = await open(), !isAnnounced {
                    unused = chat.session.id
                }
            }
        }
        .onDisappear {
            let closing = chat
            let abandoned = unused
            chat = nil
            Task {
                await closing?.close()
                if let abandoned {
                    try? await Assistant.sessions.delete(abandoned)
                }
            }
        }
    }

    /// Stream's copy of a question on its way, once it has arrived: the person's first message
    /// after the ones there were when they asked.
    private func copy(of asked: Asked) -> ChatMessage? {
        chat?.messages.filter { !$0.isAgentReply }.dropFirst(asked.after).first
    }

    private var isEmpty: Bool {
        chat?.messages.isEmpty ?? true
    }

    private var isWelcoming: Bool {
        isEmpty && asking == nil
    }

    /// What the agent is doing, shown under the last message. A question on its way is
    /// thought about before the session says anything.
    private var activity: AgentChat.Activity? {
        chat?.activity ?? (asking != nil ? .thinking : nil)
    }

    private func send(_ message: MessageData) {
        let asked = Asked(
            text: message.text, photos: Photos.thumbnails(for: message.attachments),
            after: chat?.messages.count(where: { !$0.isAgentReply }) ?? 0)
        asking = asked
        Task {
            guard let chat = await open() else {
                asking = nil
                return
            }
            markUsed(chat)
            chat.ask(
                message.text, photos: message.attachments,
                thinkHarder: message.chatOption?.id == Self.thinkHarder)
        }
    }

    /// Switches to voice mode on this conversation, starting a session for a new one.
    private func talk() {
        Task {
            guard let chat = await open() else { return }
            markUsed(chat)
            isVoiceShown = true
        }
    }

    /// Tells the list about a new conversation once it is used, written or talked to.
    private func markUsed(_ chat: AgentChat) {
        unused = nil
        guard sessionID == nil, !isAnnounced else { return }
        isAnnounced = true
        onStarted(chat.session.id)
    }

    /// The conversation's chat, starting a session for a new one.
    @discardableResult
    private func open() async -> AgentChat? {
        if let chat { return chat }
        if let opening { return await opening.value }
        let task = Task { await start() }
        opening = task
        let opened = await task.value
        opening = nil
        return opened
    }

    /// Carries on the conversation's session, or starts one for a new conversation, and opens
    /// its channel.
    private func start() async -> AgentChat? {
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
    /// Its row in the list, which Stream's copy of it takes over.
    let id = UUID().uuidString
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
