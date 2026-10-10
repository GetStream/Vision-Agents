import Foundation
import StreamChat
@_spi(Stream) import VisionAgentsCore
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
