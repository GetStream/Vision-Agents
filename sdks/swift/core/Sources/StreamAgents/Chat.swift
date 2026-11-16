import Foundation
import StreamChat
import StreamCore

private let kind = "chat"

extension StreamAgents {
    /// Hands over the app's own Stream Chat client, so every conversation opens on it rather
    /// than on a second one. Connecting it, as the user `setUser` named, stays the app's job,
    /// and `disconnect` leaves it alone.
    public func use(_ client: ChatClient) {
        backend.give(kind, client)
    }
}

extension AgentSession {
    /// The Stream Chat channel this conversation is kept in.
    ///
    /// It opens on the client handed over with `use`, or else on one built here and connected
    /// as the user `setUser` named, with the agents' key, which every session then shares.
    public func chat() async throws -> ChatChannelController {
        guard !session.conversationID.isEmpty else {
            throw AgentsError.configuration(
                "this session keeps no transcript, so there is no channel to read: only a text "
                    + "session has one, and an incognito session never does")
        }
        let cid = try ChannelId(cid: session.conversationID)
        let backend = backend
        let client = try await backend.shared(
            kind,
            open: { try await connect(backend, $0) },
            disconnect: { client in
                await withCheckedContinuation { done in client.disconnect { done.resume() } }
            })
        if let connected = client.currentUserId, let user = backend.user, connected != user.id {
            throw AgentsError.configuration(
                "the chat client is connected as \(connected) but setUser named \(user.id): "
                    + "connect it as the same user, or the conversation is somebody else's")
        }
        return client.channelController(for: cid)
    }
}

/// A client connected as the agents' user, asking them for a fresh token when Stream says the
/// one it has expired.
private func connect(_ backend: Backend, _ credentials: StreamCredentials) async throws -> ChatClient {
    let client = ChatClient(config: ChatClientConfig(apiKeyString: credentials.apiKey))
    let unused = AllocatedUnfairLock<String?>(credentials.token)
    let user = credentials.user
    try await client.connectUser(
        userInfo: UserInfo(id: user.id, name: user.originalName, imageURL: user.imageURL),
        tokenProvider: { done in
            if let token = unused.withLock({ held in defer { held = nil }; return held }) {
                done(Result { try Token(rawValue: token) })
                return
            }
            Task {
                do {
                    done(.success(try Token(rawValue: try await backend.streamCredentials(refresh: true).token)))
                } catch {
                    done(.failure(error))
                }
            }
        })
    return client
}

/// A reply's thinking, put back together from its live updates.
///
/// The router stores only the opening of each round of reasoning, as its step's `preview`.
/// The whole of it reaches people watching the reply live: each update carries the next
/// window of the round being streamed, in the message's `reasoning` field. Read every update
/// into one of these, from the client's `MessageUpdatedEvent`s rather than the channel's
/// message list, which can fold several updates into one, and show a step by its reply and its
/// id: every reply numbers its steps from `r1`.
public struct LiveReasoning: Sendable, Hashable {
    private var steps: [Key: Step] = [:]

    private struct Key: Sendable, Hashable {
        var message: String
        var step: String
    }

    private struct Step: Sendable, Hashable {
        var text = ""
        /// Where `text` ends in the round's whole thinking, in Unicode scalars.
        var end = 0
    }

    public init() {}

    /// The thinking of reasoning step `stepID` of reply `messageID` so far, or nil before any
    /// of it arrived.
    public subscript(message messageID: String, step stepID: String) -> String? {
        steps[Key(message: messageID, step: stepID)]?.text
    }

    /// Reads the window a live update carries, if it carries one.
    public mutating func read(_ message: ChatMessage) {
        guard case .dictionary(let window)? = message.extraData["reasoning"],
            case .string(let id)? = window["id"],
            case .number(let offset)? = window["offset"],
            case .string(let text)? = window["text"]
        else { return }
        add(text, at: Int(offset), to: id, of: message.id)
    }

    /// Adds what `text`, which starts `offset` scalars into the thinking of step `id` of reply
    /// `message`, has that is not held yet. A window that repeats recent thinking, for somebody
    /// who joined midway, overlaps what is held; one that starts past it means updates were
    /// missed, and is kept after a paragraph break rather than run on.
    mutating func add(_ text: String, at offset: Int, to id: String, of message: String) {
        let key = Key(message: message, step: id)
        var step = steps[key] ?? Step()
        let scalars = text.unicodeScalars
        let end = offset + scalars.count
        guard end > step.end else { return }
        if offset > step.end, !step.text.isEmpty {
            step.text += "\n\n"
        }
        step.text.unicodeScalars.append(contentsOf: scalars.dropFirst(max(0, step.end - offset)))
        step.end = end
        steps[key] = step
    }
}

/// A reply's interactive UI, shown as the session's socket streams it rather than as its
/// message in Stream Chat catches up.
///
/// The router sends a turn's UI on the session's socket the moment the model writes it
/// (`ui_delta`, then `ui_completed`), and writes it on the reply's message, as `ai_ui`, on a
/// slower tick of its own. Fold the session's conversation in with `observe(_:)` as it changes,
/// and read each reply's custom data through `overlay(_:)`: a reply Chat is still writing, whose
/// own `ai_ui` is not complete yet, shows the socket's, until the message settles and takes over.
public struct LiveUI: Sendable, Hashable {
    private var byTurn: [String: TurnUI] = [:]
    /// The UI of the turn the agent is on: the conversation's newest turn's, when it is the
    /// agent's.
    public private(set) var latest: TurnUI?

    public init() {}

    /// The UI the socket streamed for agent turn `turnID`, or nil.
    public subscript(turnID: String) -> TurnUI? {
        byTurn[turnID]
    }

    /// Takes in the UI of `conversation`'s agent turns, and says whether what it shows changed.
    /// The person's question and the agent's new turn clear `latest`, so a new reply never shows
    /// the card of the one before it.
    @discardableResult
    public mutating func observe(_ conversation: Conversation) -> Bool {
        var changed = false
        for turn in conversation.turns where turn.speaker.isAgent {
            guard let ui = turn.ui, byTurn[turn.id] != ui else { continue }
            byTurn[turn.id] = ui
            changed = true
        }
        let newest = conversation.turns.last
        let latest = newest?.speaker.isAgent == true ? newest?.ui : nil
        if latest != self.latest {
            self.latest = latest
            changed = true
        }
        return changed
    }

    /// The custom data of each agent reply in `messages` that shows the socket's UI, by message
    /// id: its own, with `ai_ui` replaced. A message missing from it shows its own custom data.
    ///
    /// A reply Chat is still writing (`generating`), whose own `ai_ui` is not complete, shows its
    /// turn's UI, matched by the `turn_id` the router writes on it. Before its turn id is in Chat,
    /// the newest reply being written shows `latest`. Once Chat says a reply is written, what
    /// Chat stored shows. A UI with no statement finished yet leaves the reply its text.
    public func overlay(_ messages: [ChatMessage]) -> [MessageId: [String: RawJSON]] {
        overlay(messages.map { (id: $0.id, custom: $0.extraData) })
    }

    /// `overlay(_:)` over each message's id and custom data, oldest first.
    func overlay(_ messages: [(id: String, custom: [String: RawJSON])]) -> [String: [String: RawJSON]] {
        guard !byTurn.isEmpty else { return [:] }
        let generating = messages.last { Self.isAgentReply($0.custom) && $0.custom["generating"]?.boolValue == true }?.id
        var overlaid: [String: [String: RawJSON]] = [:]
        for message in messages where Self.isAgentReply(message.custom) && Self.showsLiveUI(message.custom) {
            let ui: TurnUI? =
                if let turnID = Self.turnID(of: message.custom) {
                    byTurn[turnID]
                } else {
                    message.id == generating ? latest : nil
                }
            guard let ui, !ui.content.isEmpty else { continue }
            var custom = message.custom
            custom["ai_ui"] = ui.customData
            overlaid[message.id] = custom
        }
        return overlaid
    }

    private static func isAgentReply(_ custom: [String: RawJSON]) -> Bool {
        custom["ai_generated"]?.boolValue == true
            || custom["support_runtime"]?.dictionaryValue?["role"]?.stringValue == "assistant"
    }

    /// Whether a reply shows the socket's UI rather than what Chat stored: while Chat says it is
    /// being written and its UI isn't complete there.
    private static func showsLiveUI(_ custom: [String: RawJSON]) -> Bool {
        custom["generating"]?.boolValue == true
            && custom["ai_ui"]?.dictionaryValue?["status"]?.stringValue != "complete"
    }

    private static func turnID(of custom: [String: RawJSON]) -> String? {
        let turnID = custom["turn_id"]?.stringValue
            ?? custom["support_runtime"]?.dictionaryValue?["turn_id"]?.stringValue
        return turnID?.isEmpty == false ? turnID : nil
    }
}
