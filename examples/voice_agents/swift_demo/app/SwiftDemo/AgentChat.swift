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
/// phone's tools and carries the person's answers to the ones that ask first.
@MainActor
@Observable
final class AgentChat {
    let session: AgentSession

    /// The channel's messages, oldest first.
    private(set) var messages: [ChatMessage] = []

    /// Each reply's thinking in full, from its live updates. A stored step only keeps its
    /// opening.
    private(set) var reasoning = LiveReasoning()

    /// Why asking failed, or nil.
    private(set) var failure: String?

    @ObservationIgnored private let channel: ChatChannelController
    @ObservationIgnored private let events: EventsController

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
        return chat
    }

    private init(session: AgentSession, channel: ChatChannelController) {
        self.session = session
        self.channel = channel
        // The client's events rather than the channel's list: the list can fold several live
        // updates into one, and each update carries its own window of the reply's thinking.
        events = channel.client.eventsController()
        messages = channel.messages.reversed()
        channel.delegate = self
        events.delegate = self
    }

    /// What the agent is doing while its reply has nothing to show, such as "Thinking", or nil.
    ///
    /// From the session's socket rather than Stream's AI indicator events: StreamChat 5.13 cannot
    /// read those as Stream delivers them, with their state under `custom`, and drops every one
    /// but the clear.
    var indicator: String? {
        switch session.state {
        case .working(let skills):
            return "Working on " + skills.map { $0.replacingOccurrences(of: "_", with: " ") }
                .joined(separator: " and ")
        case .responding where !replyHasBegun:
            return "Thinking"
        default:
            return nil
        }
    }

    /// Whether a reply is on its way, from the first thought to the last word.
    var isAnswering: Bool {
        switch session.state {
        case .responding, .working:
            return true
        default:
            return messages.last(where: \.isAgentReply)?.isGenerating == true
        }
    }

    /// Whether the reply to the latest question shows anything yet: a step or some text.
    private var replyHasBegun: Bool {
        guard let reply = messages.last, reply.isAgentReply else { return false }
        return !reply.text.isEmpty || !reply.parts.isEmpty
    }

    func ask(_ text: String) {
        failure = nil
        Task {
            do {
                _ = try await session.responses.create(text)
            } catch is CancellationError {
            } catch {
                failure = error.localizedDescription
            }
        }
    }

    /// Stops the reply in flight. What was written so far is kept.
    func stop() {
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

    /// Whether the agent is still writing it.
    var isGenerating: Bool {
        extraData["generating"]?.boolValue == true
    }

    /// The reply's steps, in the order they happened.
    var parts: [AIMessagePart] {
        AIMessagePart.parts(from: allAttachments.map { ($0.type.rawValue, $0.payload) })
    }
}
