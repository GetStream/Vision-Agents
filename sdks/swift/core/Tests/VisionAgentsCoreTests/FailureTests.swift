import Foundation
import Testing

@testable import VisionAgentsCore

/// What a refusal says, read off a real HTTP peer: the router's envelope, or whatever else was in
/// front of it.
@Suite struct FailureTests {
    /// A status `getSession` documents, the 500 every operation declares, and one it does not
    /// name read the same way.
    @Test(arguments: [
        (404, "not_found", "session_not_found", "no session s1"),
        (500, "internal", "internal_error", "something went wrong"),
        (429, "rate_limited", "rate_limited", "slow down"),
    ])
    func aFailureCarriesTheEnvelopeAndTheRequestID(
        status: Int, type: String, code: String, message: String
    ) async throws {
        let docURL = "https://getstream.io/agents/docs/api/errors/#\(code)"
        let server = try SessionServer(
            answer: .init(
                status: status,
                headers: ["Content-Type": "application/json", "X-Request-Id": "req-1"],
                body: #"{"error":{"message":"\#(message)","type":"\#(type)","code":"\#(code)","doc_url":"\#(docURL)"}}"#))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let failure = try await refusal { _ = try await agents.sessions.get("s1") }

        #expect(
            failure
                == HTTPFailure(
                    status: status, message: message, type: type, code: code, docURL: docURL,
                    requestID: "req-1"))
    }

    /// A proxy's page, an older router's string, and something that is not JSON at all.
    @Test(arguments: [
        (502, "text/html", "<html>bad gateway</html>\n"),
        (401, "application/json", #"{"error":"token expired"}"#),
        (400, "application/json", "  not json "),
    ])
    func aBodyThatIsNotTheEnvelopeIsTheMessage(
        status: Int, contentType: String, body: String
    ) async throws {
        let server = try SessionServer(
            answer: .init(
                status: status, headers: ["Content-Type": contentType, "X-Request-Id": "req-2"],
                body: body))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let failure = try await refusal { _ = try await agents.sessions.get("s1") }

        #expect(
            failure
                == HTTPFailure(
                    status: status, message: body.trimmingCharacters(in: .whitespacesAndNewlines),
                    requestID: "req-2"))
    }

    @Test func anEmptyBodyIsNamedByItsStatus() async throws {
        let server = try SessionServer(answer: .init(status: 503, headers: [:], body: ""))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let failure = try await refusal { try await agents.sessions.delete("s1") }

        #expect(
            failure == HTTPFailure(status: 503, message: HTTPURLResponse.localizedString(forStatusCode: 503)))
    }

    @Test func aSuccessThatDoesNotReadIsUnreadableRatherThanUnreached() async throws {
        let server = try SessionServer(
            answer: .init(status: 200, headers: ["Content-Type": "application/json"], body: "{}"))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let error = try await #require(throws: AgentsError.self) { try await agents.sessions.get("s1") }

        guard case .unreadable = error else {
            Issue.record("\(error) is not unreadable")
            return
        }
    }

    /// URLSession hands over a refused upgrade's status and headers but never its body.
    @Test func aRefusedSocketSaysItsStatusAndRequestID() async throws {
        let server = try SessionServer(
            answer: .init(
                status: 401,
                headers: ["Content-Type": "application/json", "X-Request-Id": "req-3"],
                body: #"{"error":{"message":"token expired","type":"authentication","code":"unauthenticated","doc_url":"https://getstream.io/agents/docs/api/errors/#unauthenticated"}}"#))
        defer { server.listener.cancel() }
        let backend = Backend(url: try await server.url(), customerID: "acme")
        let socket = SessionSocket(url: backend.socketURL(path: "/v1/agents/sessions/s1/events"), headers: [:])
        let events = await socket.open()

        let failure = try await refusal { for try await _ in events {} }
        await socket.close()

        #expect(
            failure
                == HTTPFailure(
                    status: 401, message: HTTPURLResponse.localizedString(forStatusCode: 401),
                    requestID: "req-3"))
    }
}

/// The HTTP failure `work` was refused with.
private func refusal(_ work: () async throws -> Void) async throws -> HTTPFailure {
    let error = try await #require(throws: AgentsError.self) { try await work() }
    guard case .http(let failure) = error else {
        Issue.record("\(error) is not an HTTP failure")
        throw error
    }
    return failure
}
