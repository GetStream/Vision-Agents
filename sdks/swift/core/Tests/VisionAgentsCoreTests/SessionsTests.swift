import Foundation
import Network
import Testing

@testable import VisionAgentsCore

@Suite struct SessionsTests {
    @Test func anUpdateSendsOnlyWhatChangedAndReturnsTheSession() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let session = try await agents.agent("myagent").sessions.update("s1", title: "Renamed")

        #expect(session.id == "s1")
        #expect(session.title == "Renamed")
        let request = try await server.request()
        #expect(request.method == "PATCH")
        #expect(request.path == "/v1/agents/sessions/s1")
        #expect(request.body == ["title": .string("Renamed")])
    }

    @Test func labelsAreSentWholeAndReadBack() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")
        let custom: [String: JSONValue] = ["tier": .string("gold"), "seats": .number(3)]

        let session = try await agents.sessions.update("s1", description: "Billing", custom: custom)

        #expect(session.description == "Billing")
        #expect(session.custom == custom)
        let request = try await server.request()
        #expect(request.body == ["description": .string("Billing"), "custom": .object(custom)])
    }
}

/// A real HTTP peer that answers one request with a session carrying what it was sent, or with
/// the answer it was given.
struct SessionServer {
    struct Request: Sendable {
        let method: String
        let path: String
        let body: [String: JSONValue]
    }

    /// What to answer instead of a session.
    struct Answer: Sendable {
        let status: Int
        let headers: [String: String]
        let body: String
    }

    let listener: NWListener
    private let addresses: AsyncThrowingStream<URL, any Error>
    private let requests: AsyncThrowingStream<Request, any Error>

    init(answer: Answer? = nil) throws {
        let queue = DispatchQueue(label: "session-server")
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        let listener = try NWListener(using: parameters)
        self.listener = listener
        let (addresses, address) = AsyncThrowingStream<URL, any Error>.makeStream()
        let (requests, request) = AsyncThrowingStream<Request, any Error>.makeStream()
        self.addresses = addresses
        self.requests = requests
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                if let port = listener.port {
                    address.yield(URL(string: "http://127.0.0.1:\(port.rawValue)")!)
                    address.finish()
                }
            case .failed(let error):
                address.finish(throwing: error)
                request.finish(throwing: error)
            default: break
            }
        }
        listener.newConnectionHandler = { connection in
            connection.start(queue: queue)
            Self.read(connection, Data(), answer, request)
        }
        queue.asyncAfter(deadline: .now() + 5) {
            let error = AgentsError.unreadable("test server timed out")
            address.finish(throwing: error)
            request.finish(throwing: error)
        }
        listener.start(queue: queue)
    }

    func url() async throws -> URL {
        try #require(try await addresses.first(where: { @Sendable _ in true }))
    }

    func request() async throws -> Request {
        try #require(try await requests.first(where: { @Sendable _ in true }))
    }

    private static func read(
        _ connection: NWConnection, _ buffer: Data, _ answer: Answer?,
        _ request: AsyncThrowingStream<Request, any Error>.Continuation
    ) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, done, error in
            var buffer = buffer
            if let data { buffer.append(data) }
            if let end = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buffer[..<end.lowerBound], as: UTF8.self)
                    .components(separatedBy: "\r\n")
                let length =
                    head.first { $0.lowercased().hasPrefix("content-length:") }
                    .flatMap { Int($0.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces)) } ?? 0
                let body = buffer[end.upperBound...]
                if body.count >= length {
                    let fields = (try? JSONDecoder().decode([String: JSONValue].self, from: body)) ?? [:]
                    let line = head[0].split(separator: " ").map(String.init)
                    let path = String(line[1].split(separator: "?")[0])
                    respond(connection, answer ?? session(path: path, fields: fields))
                    request.yield(Request(method: line[0], path: path, body: fields))
                    request.finish()
                    return
                }
            }
            if let error {
                request.finish(throwing: error)
            } else if !done {
                read(connection, buffer, answer, request)
            }
        }
    }

    private static func session(path: String, fields: [String: JSONValue]) -> Answer {
        var session: [String: JSONValue] = [
            "id": .string(String(path.split(separator: "/").last ?? "")),
            "agent_id": .string("a1"), "call_id": .string(""), "call_type": .string("default"),
            "created_at": .string("2026-10-02T17:00:00Z"), "modality": .string("text"),
            "state": .string("live"), "user_id": .string("jlahey"),
        ]
        for key in ["title", "description", "custom"] {
            if let value = fields[key] { session[key] = value }
        }
        let body = (try? JSONEncoder().encode(session)) ?? Data()
        return Answer(
            status: 200, headers: ["Content-Type": "application/json"],
            body: String(decoding: body, as: UTF8.self))
    }

    private static func respond(_ connection: NWConnection, _ answer: Answer) {
        let body = Data(answer.body.utf8)
        var head = "HTTP/1.1 \(answer.status) \(HTTPURLResponse.localizedString(forStatusCode: answer.status))\r\n"
        for (name, value) in answer.headers.sorted(by: { $0.key < $1.key }) {
            head += "\(name): \(value)\r\n"
        }
        head += "Content-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        connection.send(
            content: Data(head.utf8) + body,
            completion: .contentProcessed { _ in connection.cancel() })
    }
}
