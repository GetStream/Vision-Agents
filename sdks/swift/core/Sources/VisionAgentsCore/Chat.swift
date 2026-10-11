import Foundation
import StreamChat
import os

private let kind = "chat"

extension VisionAgents {
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
    let unused = OSAllocatedUnfairLock<String?>(initialState: credentials.token)
    let user = credentials.user
    try await client.connectUser(
        userInfo: UserInfo(
            id: user.id, name: user.name.isEmpty ? nil : user.name, imageURL: URL(string: user.image)),
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
/// message list, which can fold several updates into one, and show a step by its id.
public struct LiveReasoning: Sendable, Hashable {
    private var steps: [String: Step] = [:]

    private struct Step: Sendable, Hashable {
        var text = ""
        /// Where `text` ends in the round's whole thinking, in Unicode scalars.
        var end = 0
    }

    public init() {}

    /// The thinking of reasoning step `stepID` so far, or nil before any of it arrived.
    public subscript(stepID: String) -> String? {
        steps[stepID]?.text
    }

    /// Reads the window a live update carries, if it carries one.
    public mutating func read(_ message: ChatMessage) {
        guard case .dictionary(let window)? = message.extraData["reasoning"],
            case .string(let id)? = window["id"],
            case .number(let offset)? = window["offset"],
            case .string(let text)? = window["text"]
        else { return }
        add(text, at: Int(offset), to: id)
    }

    /// Adds what `text`, which starts `offset` scalars into step `id`'s thinking, has that is
    /// not held yet. A window that repeats recent thinking, for somebody who joined midway,
    /// overlaps what is held; one that starts past it means updates were missed, and is kept
    /// after a paragraph break rather than run on.
    mutating func add(_ text: String, at offset: Int, to id: String) {
        var step = steps[id] ?? Step()
        let scalars = text.unicodeScalars
        let end = offset + scalars.count
        guard end > step.end else { return }
        if offset > step.end, !step.text.isEmpty {
            step.text += "\n\n"
        }
        step.text.unicodeScalars.append(contentsOf: scalars.dropFirst(max(0, step.end - offset)))
        step.end = end
        steps[id] = step
    }
}
