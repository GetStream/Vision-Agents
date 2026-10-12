import Foundation
import Testing

@testable import VisionAgentsCore

@Suite struct GuestTests {
    let store = FileGuestStore(
        url: FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathComponent("guest.json"))

    static func minted(_ body: String) -> SessionServer.Answer {
        SessionServer.Answer(status: 201, headers: ["Content-Type": "application/json"], body: body)
    }

    @Test func aRememberedGuestIsTheSameGuestOnTheNextLaunch() async throws {
        let server = try SessionServer(
            answer: Self.minted(#"{"id":"guest-1","token":"tok","name":"Guest","expires_at":"2999-01-01T00:00:00Z"}"#))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let first = try await agents.guestUser(GuestOptions(name: "Guest"), store: store)
        let again = try await agents.guestUser(GuestOptions(name: "Guest"), store: store)

        #expect(first == again)
        #expect(first.id == "guest-1")
        let request = try await server.request()
        #expect(request.path == "/v1/agents/guests")
        #expect(request.body == ["name": .string("Guest")])
    }

    @Test func aGuestWhoseTokenExpiredComesBackAsTheSameGuestWithAFreshOne() async throws {
        try store.write(#"{"id":"guest-1","token":"old","expires_at":"2020-01-01T00:00:00Z"}"#)
        let server = try SessionServer(
            answer: Self.minted(#"{"id":"guest-1","token":"new","expires_at":"2999-01-01T00:00:00Z"}"#))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let guest = try await agents.guestUser(store: store)

        #expect(try await server.request().body == ["id": .string("guest-1")])
        #expect(guest.token == "new")
    }

    @Test func notMeMintsSomebodyNewAndForgettingLeavesNothing() async throws {
        try store.write(#"{"id":"guest-1","token":"tok"}"#)
        let server = try SessionServer(answer: Self.minted(#"{"id":"guest-2","token":"t"}"#))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let guest = try await agents.guestUser(GuestOptions(fresh: true), store: store)
        agents.setUser(guest)

        #expect(try await server.request().body == [:])
        #expect(agents.backend.user?.id == "guest-2")
        try agents.forgetGuest(store: store)
        #expect(store.read() == nil)
        #expect(agents.backend.user == nil)
    }

    @Test func aStoreHoldingRubbishIsNotFatal() async throws {
        try store.write("{truncated")
        let server = try SessionServer(answer: Self.minted(#"{"id":"guest-1","token":"t"}"#))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let guest = try await agents.guestUser(store: store)

        #expect(guest.id == "guest-1")
        #expect(try await server.request().body == [:])
    }
}
