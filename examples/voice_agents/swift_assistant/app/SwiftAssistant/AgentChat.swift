import Foundation
import Combine
import Observation
import StreamChat
import StreamChatAI
import VisionAgentsCore

/// A written conversation as Stream's AI components want it: the channel it is kept in, the
/// whole of each reply's thinking, and what the agent is doing before it writes.
///
/// The router writes everything here. Asking goes through the session, which records the
/// question in the channel and answers it there: the reply's text, and its steps (each round
/// of reasoning and each tool call) as attachments. The session's socket is what runs this
/// phone's tools and carries the person's answers to the ones that ask first, on a call too.
@MainActor
@Observable
final class AgentChat {
    let session: AgentSession

    /// The session's conversation as it last was. The session is an `ObservableObject`, which
    /// Observation does not see into, so views follow it through here.
    private(set) var conversation: Conversation

    /// The session's tool calls waiting for the person, followed the same way.
    private(set) var approvals: [ToolApprovalRequest] = []

    /// The channel's messages, oldest first.
    private(set) var messages: [ChatMessage] = []

    /// Each reply's thinking in full, from its live updates. A stored step only keeps its
    /// opening.
    private(set) var reasoning = LiveReasoning()

    /// Why asking failed, or nil.
    private(set) var failure: String?

    /// Whether the channel holds messages older than those loaded.
    private(set) var hasOlderMessages = false

    /// How the person rated replies on this phone, by message. Kept for the session only: the
    /// router takes no ratings yet.
    private(set) var ratings: [String: AIMessageRating] = [:]

    /// The newest message when something was last asked here, until it fails or is stopped:
    /// what arrives after it is the question and then its reply.
    private var askedAfter: String?

    @ObservationIgnored private let channel: ChatChannelController
    @ObservationIgnored private let events: EventsController
    @ObservationIgnored private var following: Set<AnyCancellable> = []

    /// Opens the session's channel and starts following the session.
    static func open(_ session: AgentSession) async throws -> AgentChat {
        let channel = try await session.chat()
        let chat = AgentChat(session: session, channel: channel)
        try await withCheckedThrowingContinuation { (done: CheckedContinuation<Void, any Error>) in
            channel.synchronize { error in
                if let error { done.resume(throwing: error) } else { done.resume() }
            }
        }
        await session.start()
        chat.hasOlderMessages = !channel.hasLoadedAllPreviousMessages
        return chat
    }

    private init(session: AgentSession, channel: ChatChannelController) {
        self.session = session
        self.channel = channel
        conversation = session.conversation
        approvals = session.approvals
        // The client's events rather than the channel's list: the list can fold several live
        // updates into one, and each update carries its own window of the reply's thinking.
        events = channel.client.eventsController()
        messages = channel.messages.reversed()
        channel.delegate = self
        events.delegate = self
        // Published from the main actor, which is where the session changes.
        session.$conversation
            .sink { [weak self] conversation in MainActor.assumeIsolated { self?.conversation = conversation } }
            .store(in: &following)
        session.$approvals
            .sink { [weak self] approvals in MainActor.assumeIsolated { self?.approvals = approvals } }
            .store(in: &following)
    }

    /// What the agent is doing while its reply shows nothing yet.
    enum Activity: Equatable {
        /// Thinking about the question.
        case thinking
        /// Running skills, such as "Working on trip planning".
        case working(String)
    }

    /// What the agent is doing while its reply has nothing to show, or nil.
    var activity: Activity? {
        switch conversation.state {
        case .working(let skills):
            return .working(
                "Working on " + skills.map { $0.replacingOccurrences(of: "_", with: " ") }
                    .joined(separator: " and "))
        case .responding where !replyHasBegun:
            return .thinking
        default:
            // The session can pass through idle between the question and the first of its
            // reply, which would hide the activity for a moment.
            return isAwaitingReply ? .thinking : nil
        }
    }

    /// The messages as Stream's AI components show them, each reply with the whole of its
    /// thinking.
    var listMessages: [AgentMessage] {
        messages.map { AgentMessage($0, reasoning: reasoning) }
    }

    /// Loads the messages before the oldest one loaded.
    func loadOlderMessages() async {
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            channel.loadPreviousMessages { _ in done.resume() }
        }
        hasOlderMessages = !channel.hasLoadedAllPreviousMessages
    }

    /// Rates a reply, or takes the rating back with nil.
    func rate(_ reply: ChatMessage, _ rating: AIMessageRating?) {
        ratings[reply.id] = rating
    }

    /// Whether a reply is on its way, from the first thought to the last word.
    var isAnswering: Bool {
        switch conversation.state {
        case .responding, .working:
            return true
        default:
            return messages.last(where: \.isAgentReply)?.isGenerating == true
        }
    }

    /// Whether the reply to the latest question shows anything yet: a step or some text.
    private var replyHasBegun: Bool {
        guard let reply = messages.last, reply.isAgentReply else { return false }
        return !reply.isEmptyReply
    }

    /// Whether the question asked here is in the channel and its reply shows nothing yet.
    private var isAwaitingReply: Bool {
        guard let askedAfter, let last = messages.last, last.id != askedAfter else { return false }
        return !last.isAgentReply || (last.isGenerating && !replyHasBegun)
    }

    /// Asks, with the photos the composer holds and, when `thinkHarder`, more time to reason.
    func ask(_ text: String, photos: [URL] = [], thinkHarder: Bool = false) {
        failure = nil
        askedAfter = messages.last?.id ?? ""
        Task {
            do {
                let images = await Photos.sources(for: photos)
                _ = try await session.responses.create(
                    text, images: images, reasoningEffort: thinkHarder ? "high" : nil)
            } catch is CancellationError {
            } catch {
                failure = error.localizedDescription
                askedAfter = nil
            }
        }
    }

    /// Stops the reply in flight. What was written so far is kept.
    func stop() {
        askedAfter = nil
        Task { try? await session.interrupt() }
    }

    func close() async {
        await session.close()
    }
}

extension AgentChat: ChatChannelControllerDelegate {
    func channelController(
        _ channelController: ChatChannelController,
        didUpdateMessages changes: [ListChange<ChatMessage>]
    ) {
        messages = channelController.messages.reversed()
    }
}

extension AgentChat: EventsControllerDelegate {
    func eventsController(_ controller: EventsController, didReceiveEvent event: Event) {
        guard let event = event as? MessageUpdatedEvent, event.cid == channel.cid else { return }
        reasoning.read(event.message)
    }
}

extension ChatMessage {
    /// A reply the agent wrote, which Stream's AI components render.
    var isAgentReply: Bool {
        extraData["ai_generated"]?.boolValue == true
    }

    /// Whether it is a reply with nothing to show yet: no text and no steps.
    var isEmptyReply: Bool {
        isAgentReply && text.isEmpty && parts.isEmpty
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
