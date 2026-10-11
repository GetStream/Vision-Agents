import Foundation
import Testing

@testable import VisionAgentsCore

@Suite struct StreamCredentialsTests {
    @Test func theTokenIsAskedForOnceAndAgainOnlyOnARefresh() async throws {
        let asked = Tally()
        let backend = Backend(apiKey: "key")
        backend.setUser(User(id: "john", name: "John")) { "token-\(await asked.add())" }

        let first = try await backend.streamCredentials()
        let again = try await backend.streamCredentials()
        let refreshed = try await backend.streamCredentials(refresh: true)

        #expect(first == StreamCredentials(apiKey: "key", user: User(id: "john", name: "John"), token: "token-1"))
        #expect(again.token == "token-1")
        #expect(refreshed.token == "token-2")
    }

    @Test func twoClientsRefreshingAtOnceFetchOneToken() async throws {
        let asked = Tally()
        let backend = Backend(apiKey: "key")
        backend.setUser(User(id: "john")) {
            let count = await asked.add()
            // Long enough that the second refresh arrives while the first is still fetching.
            try await Task.sleep(for: .milliseconds(50))
            return "token-\(count)"
        }
        _ = try await backend.streamCredentials()

        async let chat = backend.streamCredentials(refresh: true)
        async let video = backend.streamCredentials(refresh: true)

        #expect(try await [chat.token, video.token] == ["token-2", "token-2"])
        #expect(await asked.count == 2)
    }

    @Test func refusesWithoutAStreamKeyOrAUser() async throws {
        let local = Backend(url: URL(string: "http://localhost:8080")!, customerID: "acme")
        local.setUser(User(id: "john"), token: "token")
        await #expect(throws: AgentsError.self) { try await local.streamCredentials() }

        await #expect(throws: AgentsError.self) { try await Backend(apiKey: "key").streamCredentials() }
    }

    @Test func twoSessionsAskingAtOnceShareOneClient() async throws {
        let opened = Tally()
        let backend = signedIn("john")

        async let first = backend.shared("chat", open: { credentials in
            _ = await opened.add()
            try await Task.sleep(for: .milliseconds(50))
            return Peer(credentials.user.id)
        }, disconnect: { _ in })
        async let second = backend.shared("chat", open: { credentials in
            _ = await opened.add()
            return Peer(credentials.user.id)
        }, disconnect: { _ in })

        #expect(try await first === second)
        #expect(await opened.count == 1)
    }

    @Test func aClientThatFailedToOpenIsOpenedAgain() async throws {
        let opened = Tally()
        let backend = signedIn("john")
        let open: @Sendable (StreamCredentials) async throws -> Peer = { credentials in
            if await opened.add() == 1 { throw AgentsError.unreadable("offline") }
            return Peer(credentials.user.id)
        }

        await #expect(throws: AgentsError.self) {
            try await backend.shared("chat", open: open, disconnect: { _ in })
        }
        let peer = try await backend.shared("chat", open: open, disconnect: { _ in })

        #expect(peer.user == "john")
        #expect(await opened.count == 2)
    }

    @Test func anotherUserGetsAClientOfTheirOwn() async throws {
        let backend = signedIn("john")
        let open: @Sendable (StreamCredentials) async throws -> Peer = { Peer($0.user.id) }

        let john = try await backend.shared("chat", open: open, disconnect: { _ in })
        backend.setUser(User(id: "jane"), token: "token-for-jane")
        let jane = try await backend.shared("chat", open: open, disconnect: { _ in })

        #expect(john.user == "john")
        #expect(jane.user == "jane")
    }

    @Test func theAppsOwnClientIsUsedKeptAndLeftConnected() async throws {
        let opened = Tally()
        let disconnected = Tally()
        let backend = signedIn("john")
        let mine = Peer("john")
        backend.give("chat", mine)
        let open: @Sendable (StreamCredentials) async throws -> Peer = { credentials in
            _ = await opened.add()
            return Peer(credentials.user.id)
        }
        let disconnect: @Sendable (Peer) async -> Void = { _ in _ = await disconnected.add() }

        let first = try await backend.shared("chat", open: open, disconnect: disconnect)
        backend.setUser(User(id: "john"), token: "another-token")
        let second = try await backend.shared("chat", open: open, disconnect: disconnect)
        await VisionAgents(backend: backend).disconnect()

        #expect(first === mine)
        #expect(second === mine)
        #expect(await opened.count == 0)
        #expect(await disconnected.count == 0)
    }

    @Test func disconnectingClosesWhatWasBuiltAndTheNextSessionOpensAnew() async throws {
        let opened = Tally()
        let disconnected = Tally()
        let backend = signedIn("john")
        let open: @Sendable (StreamCredentials) async throws -> Peer = { credentials in
            _ = await opened.add()
            return Peer(credentials.user.id)
        }
        let disconnect: @Sendable (Peer) async -> Void = { _ in _ = await disconnected.add() }

        let first = try await backend.shared("chat", open: open, disconnect: disconnect)
        await VisionAgents(backend: backend).disconnect()
        let second = try await backend.shared("chat", open: open, disconnect: disconnect)

        #expect(first !== second)
        #expect(await disconnected.count == 1)
        #expect(await opened.count == 2)
    }

    private func signedIn(_ user: String) -> Backend {
        let backend = Backend(apiKey: "key")
        backend.setUser(User(id: user), token: "token-for-\(user)")
        return backend
    }
}

/// Stands in for a Stream client: something with an identity of its own, connected as a user.
private final class Peer: Sendable {
    let user: String

    init(_ user: String) {
        self.user = user
    }
}

/// Counts what it was asked to, from whichever task asks.
private actor Tally {
    private(set) var count = 0

    func add() -> Int {
        count += 1
        return count
    }
}
