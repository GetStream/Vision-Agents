import Foundation
import Combine
import Observation
import StreamChat
import StreamChatA2UI
import StreamChatAI
import StreamChatOpenUI
import VisionAgentsCore

/// How Stream's `AIConversation` reaches Juno: the session that answers the conversation, and
/// the channel the router keeps it in.
///
/// The router writes everything here. Asking goes through the session, which records the
/// question in the channel and answers it there: the reply's text, and its steps (each round
/// of reasoning and each tool call) as attachments. The session's socket is what runs this
/// phone's tools and carries the person's answers to the ones that ask first, on a call too.
///
/// A new conversation has no session until it is first written or talked to, which opens one.
/// The router keeps a conversation in Chat and cannot rewind it, so a reply is not written
/// again and a message is not edited: asking for either asks the question again.
@MainActor
@Observable
final class AgentChat: AIConversationTransport {
    /// The session, once the conversation has one.
    private(set) var session: AgentSession?

    /// The session once something was asked in it or said on its call. A new conversation
    /// joins the list then.
    private(set) var usedSessionID: String?

    /// The session's tool calls waiting for the person. The session is an `ObservableObject`,
    /// which Observation does not see into, so views follow them through here.
    private(set) var approvals: [ToolApprovalRequest] = []

    /// What the session last said went wrong, followed the same way.
    private(set) var sessionFailure: String?

    /// Why telling the agent what the person did in a reply's UI failed, or nil.
    private(set) var failure: String?

    /// The UI the agent writes for some replies, in A2UI or OpenUI, whichever it was told.
    typealias UI = AIUIHost<AIUIRendererPair<A2UIRenderer, OpenUIRenderer>>

    /// Shows the UI the agent writes for some replies, such as an interactive trip plan, with the
    /// phone's own components, and sends what the person does in it back.
    @ObservationIgnored private(set) var ui: UI?

    /// The composer option that asks with a higher reasoning effort.
    static let thinkHarder = "think_harder"

    /// The conversation to carry on, or nil for a new one.
    @ObservationIgnored private let sessionID: String?
    @ObservationIgnored private var feed: AIConversationFeed<AgentMessage>?
    @ObservationIgnored private var channel: ChatChannelController?
    @ObservationIgnored private var events: EventsController?
    /// Each reply's thinking in full, from its live updates. A stored step only keeps its
    /// opening.
    @ObservationIgnored private var reasoning = LiveReasoning()
    /// What the conversation was last told the agent is doing.
    @ObservationIgnored private var activity: AIAgentActivity?
    @ObservationIgnored private var following: Set<AnyCancellable> = []
    /// The session being opened, so asking while it opens waits for it rather than opening a
    /// second one.
    @ObservationIgnored private var opening: Task<AgentSession, any Error>?

    init(sessionID: String?) {
        self.sessionID = sessionID
        ui = AIUIHost(components: TripComponents.all) {
            A2UIRenderer()
            OpenUIRenderer()
        } send: { [weak self] action in
            try await self?.answer(action)
        }
    }

    /// Answers the questions of the tools that ask first, and the sign-ins the agent's
    /// connectors wait for, as the person signed in here. A sign-in opens in a web
    /// authentication sheet, which the router hands back to this app once the provider has
    /// answered; "Not now" ends the call without it.
    var approver: AIToolApprover {
        AIToolApprover(
            userID: Assistant.user.id, clientID: AIClientIdentity.installID,
            authorizer: AIToolAuthorizer(callbackScheme: Assistant.callbackScheme) { [weak self] call in
                try await self?.session?.declineLogin(call.id)
            }
        ) { [weak self] call, allowed in
            try await self?.session?.decide(call.id, allowed: allowed)
        }
    }

    func connect(_ feed: AIConversationFeed<AgentMessage>) {
        self.feed = feed
        publish()
    }

    /// Asks, with the photos the composer holds and, for Think harder, more time to reason.
    func send(_ request: AIConversationRequest<AgentMessage>) async throws {
        let session = try await use()
        let images = await Photos.sources(for: request.attachments)
        _ = try await session.responses.create(
            request.text, images: images,
            reasoningEffort: request.chatOption?.id == Self.thinkHarder ? "high" : nil)
    }

    /// Stops the reply in flight. What was written so far is kept.
    func stop() async throws {
        try await session?.interrupt()
    }

    /// Loads the messages before the oldest one loaded.
    func loadOlderMessages() async throws {
        guard let channel else { return }
        try await withCheckedThrowingContinuation { (done: CheckedContinuation<Void, any Error>) in
            channel.loadPreviousMessages { error in
                if let error { done.resume(throwing: error) } else { done.resume() }
            }
        }
        publish()
    }

    /// The conversation's session, opened the first time it is asked for: the one it carries on,
    /// or a new one for a new conversation.
    @discardableResult
    func open() async throws -> AgentSession {
        if let session { return session }
        if let opening { return try await opening.value }
        let task = Task { try await start() }
        opening = task
        defer { opening = nil }
        return try await task.value
    }

    /// The session, open and written or talked to.
    @discardableResult
    func use() async throws -> AgentSession {
        let session = try await open()
        usedSessionID = session.id
        return session
    }

    /// Closes the session. A new conversation's that was opened while its first message was
    /// written, and never used, is deleted, so it is not left in the list.
    func close() async {
        if let opening {
            _ = try? await opening.value
        }
        guard let session else { return }
        await session.close()
        if sessionID == nil, usedSessionID == nil {
            try? await Assistant.sessions.delete(session.id)
        }
    }

    /// Carries on the conversation's session, or starts one for a new conversation, and opens
    /// its channel.
    private func start() async throws -> AgentSession {
        let session: AgentSession
        if let sessionID {
            session = try await Assistant.agents.attach(sessionID: sessionID, tools: Assistant.tools)
        } else {
            var options = SessionOptions(agent: Assistant.agentName)
            options.llm = Assistant.chatModel
            options.tools = Assistant.tools
            options.returnURL = Assistant.returnURL
            session = try await Assistant.agents.chat(options)
        }
        let channel = try await session.chat()
        try await withCheckedThrowingContinuation { (done: CheckedContinuation<Void, any Error>) in
            channel.synchronize { error in
                if let error { done.resume(throwing: error) } else { done.resume() }
            }
        }
        follow(session, channel: channel)
        await session.start()
        self.session = session
        publish()
        return session
    }

    private func follow(_ session: AgentSession, channel: ChatChannelController) {
        self.channel = channel
        // The client's events rather than the channel's list: the list can fold several live
        // updates into one, and each update carries its own window of the reply's thinking.
        let events = channel.client.eventsController()
        self.events = events
        channel.delegate = self
        events.delegate = self
        // Published from the main actor, which is where the session changes.
        session.$conversation
            .sink { [weak self] conversation in MainActor.assumeIsolated { self?.sessionChanged(conversation) } }
            .store(in: &following)
        session.$approvals
            .sink { [weak self] approvals in MainActor.assumeIsolated { self?.approvals = approvals } }
            .store(in: &following)
    }

    /// Tells the conversation what the channel holds: its messages, each reply with the whole
    /// of its thinking, and whether there are older ones.
    private func publish() {
        guard let channel else { return }
        feed?.update(messages: channel.messages.reversed().map { AgentMessage($0, reasoning: reasoning) })
        feed?.update(hasOlderMessages: channel.state == .remoteDataFetched && !channel.hasLoadedAllPreviousMessages)
    }

    /// Tells the conversation what the agent is doing, which keeps it waiting for a reply that
    /// takes long, such as one that runs a skill.
    private func sessionChanged(_ conversation: Conversation) {
        sessionFailure = conversation.failure
        let activity: AIAgentActivity? = switch conversation.state {
        case .responding:
            AIAgentActivity(state: .thinking)
        case .working(let skills):
            AIAgentActivity(
                state: .checkingExternalSources,
                text: "Working on " + skills.map { $0.replacingOccurrences(of: "_", with: " ") }
                    .joined(separator: " and "))
        default:
            nil
        }
        guard activity != self.activity else { return }
        self.activity = activity
        feed?.update(activity: activity)
    }

    /// Tells the agent what the person did in the UI of a reply, such as tapping a place on its
    /// map: their message says it in words, and carries the action for the agent. A request of
    /// the conversation carries no action, so this asks the session itself, and the conversation
    /// follows the question and its reply in the channel.
    private func answer(_ action: AIUIAction) async throws {
        failure = nil
        do {
            _ = try await use().responses.create(action.label, uiAction: action.json)
        } catch is CancellationError {
        } catch {
            failure = error.localizedDescription
            throw error
        }
    }
}

extension AgentChat: ChatChannelControllerDelegate {
    func channelController(
        _ channelController: ChatChannelController,
        didUpdateMessages changes: [ListChange<ChatMessage>]
    ) {
        publish()
    }
}

extension AgentChat: EventsControllerDelegate {
    func eventsController(_ controller: EventsController, didReceiveEvent event: Event) {
        guard let event = event as? MessageUpdatedEvent, event.cid == channel?.cid else { return }
        let before = reasoning
        reasoning.read(event.message)
        if reasoning != before {
            publish()
        }
    }
}

extension ChatMessage {
    /// A reply the agent wrote, which Stream's AI components render.
    var isAgentReply: Bool {
        extraData["ai_generated"]?.boolValue == true
    }

    /// Whether the agent is still writing it.
    var isGenerating: Bool {
        extraData["generating"]?.boolValue == true
    }

    /// The reply's steps, in the order they happened.
    var parts: [AIMessagePart] {
        AIMessagePart.parts(from: allAttachments.map { ($0.type.rawValue, $0.payload) })
    }
}

/// A message of the conversation as Stream's AI components show it.
struct AgentMessage: AIMessage {
    let message: ChatMessage
    let parts: [AIMessagePart]
    private let reasoning: LiveReasoning

    init(_ message: ChatMessage, reasoning: LiveReasoning) {
        self.message = message
        self.reasoning = reasoning
        parts = message.parts
    }

    var id: String { message.id }
    var role: AIMessageRole { message.isAgentReply ? .assistant : .user }
    var text: String { message.text }
    var isGenerating: Bool { message.isGenerating }
    /// The UI the agent wrote for the reply, which shows in place of its text.
    var uiDocument: AIUIDocument? { AIUIDocument(customData: message.extraData) }
    var attachments: [AIMessageAttachment] {
        message.imageAttachments.map { .image($0.imageURL, name: $0.title) }
            + message.fileAttachments.map {
                .file($0.assetURL, name: $0.title, size: $0.file.size, mimeType: $0.file.mimeType)
            }
    }

    /// A step keeps only the opening of its thinking; the whole of it came with the live
    /// updates, while the reply was being written.
    func reasoningText(for step: AIReasoningPart) -> String? {
        reasoning[message: message.id, step: step.id]
    }
}
