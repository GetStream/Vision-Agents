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
        let custom: [String: RawJSON] = ["tier": .string("gold"), "seats": .number(3)]

        let session = try await agents.sessions.update("s1", description: "Billing", custom: custom)

        #expect(session.description == "Billing")
        #expect(session.custom == custom)
        let request = try await server.request()
        #expect(request.body == ["description": .string("Billing"), "custom": .dictionary(custom)])
    }

    @Test func aDeleteAnsweredWithNoContentSucceeds() async throws {
        let server = try SessionServer(answer: .init(status: 204, headers: [:], body: ""))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        try await agents.sessions.delete("s1")

        let request = try await server.request()
        #expect(request.method == "DELETE")
        #expect(request.path == "/v1/agents/sessions/s1")
    }

    @Test func aVoiceSessionAsksToStartVoiceAndIsOnItsOwnCall() async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: "s1"))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")
        var options = SessionOptions(agent: "myagent")
        options.startVoice = true

        let session = try await agents.createSession(options)

        #expect(session.callID == "s1")
        let request = try await server.request()
        #expect(request.path == "/v1/agents/sessions")
        #expect(request.body["start_voice"] == .bool(true))
        #expect(request.body["text"] == nil)
    }

    @Test func aSessionHeldInWritingSendsNoVoice() async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: ""))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")
        var options = SessionOptions(agent: "myagent")
        options.greeting = Greeting("Hello.", mode: .variation)

        _ = try await agents.createSession(options)

        let body = try await server.request().body
        #expect(body["start_voice"] == nil)
        #expect(body["greeting"] == .dictionary(["text": .string("Hello."), "mode": .string("variation")]))
    }

    @Test func aSessionSaysWhereItsSignInsGoBackTo() async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: ""))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")
        var options = SessionOptions(agent: "myagent")
        options.returnURL = URL(string: "com.example.app:/connected")

        _ = try await agents.createSession(options)

        #expect(try await server.request().body["return_url"] == .string("com.example.app:/connected"))
    }

    @Test func aSessionOpensWithItsLabelsModelsAndTheSignedInUsersMemory() async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: ""))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")
        agents.setUser(User(id: "jlahey"), token: "token")
        var options = SessionOptions(agent: "myagent")
        options.custom = ["ticket": .string("T-7")]
        options.modelOverwrites = ModelOverwrites(thinking: .low, temperature: 0, maxOutputTokens: 300)
        options.memory = .init(filter: ["topic": "billing"])
        options.timezone = "Europe/Amsterdam"

        _ = try await agents.createSession(options)

        let body = try await server.request().body
        #expect(body["timezone"] == .string("Europe/Amsterdam"))
        #expect(body["custom"] == .dictionary(["ticket": .string("T-7")]))
        #expect(
            body["model_overwrites"]
                == .dictionary(["thinking": .string("low"), "temperature": .number(0), "max_output_tokens": .number(300)]))
        #expect(
            body["memory"]
                == .dictionary(["user_id": .string("jlahey"), "filter": .dictionary(["topic": .string("billing")])]))
    }

    @Test func aSessionWithoutMemoryOrLabelsSendsNeither() async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: ""))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")
        agents.setUser(User(id: "jlahey"), token: "token")

        _ = try await agents.createSession(SessionOptions(agent: "myagent"))

        let body = try await server.request().body
        #expect(body["memory"] == nil)
        #expect(body["custom"] == nil)
        #expect(body["model_overwrites"] == nil)
    }

    @Test func aForkNamesTheAgentAndWhatItChanges() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")
        var options = ForkOptions(responseID: "r2")
        options.agent = "reviewer"
        options.custom = ["branch": .string("b")]
        options.modelOverwrites = ModelOverwrites(llm: "anthropic/claude-sonnet-5-5")
        options.incognito = true

        _ = try await agents.fork(sessionID: "s1", options)

        let request = try await server.request()
        #expect(request.path == "/v1/agents/sessions/s1/fork")
        #expect(
            request.body == [
                "agent": .string("reviewer"), "response_id": .string("r2"),
                "custom": .dictionary(["branch": .string("b")]),
                "model_overwrites": .dictionary(["llm": .string("anthropic/claude-sonnet-5-5")]),
                "incognito": .bool(true),
            ])
    }

    @Test func aQueryNarrowsByWhenLabelsAndOrder() async throws {
        let server = try SessionServer(
            answer: .init(
                status: 200, headers: ["Content-Type": "application/json"],
                body: #"{"items":[],"has_more":false}"#))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")
        var query = SessionQuery()
        query.createdSince = Date(timeIntervalSince1970: 1_790_000_000)
        query.custom = ["ticket": "T-7"]
        query.sort = .updatedAt

        _ = try await agents.agent("myagent").sessions.search("refund", query)

        let body = try await server.request().body
        #expect(
            body["filter"]
                == .dictionary([
                    "agent": .string("myagent"), "created_at": .dictionary(["$gte": .string("2026-09-21T14:13:20.000Z")]),
                    "custom": .dictionary(["ticket": .string("T-7")]), "text": .dictionary(["$q": .string("refund")]),
                ]))
        #expect(body["sort"] == .array([.dictionary(["field": .string("updated_at")])]))
    }

    @Test func everyItemIsReadInOrderAndNoMoreIsAskedForAfterTheLastPage() async throws {
        let server = try SessionServer(
            answer: .init(
                status: 200, headers: ["Content-Type": "application/json"],
                body: #"""
                    {"items":[
                      {"response_id":"r1","ordinal":0,"kind":"said","text":"Hi","at":"2026-10-27T08:00:00Z"},
                      {"response_id":"r1","ordinal":1,"kind":"answer","text":"Hello","at":"2026-10-27T08:00:01Z"}
                    ],"has_more":false}
                    """#))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        var read: [String] = []
        for try await item in agents.sessions.responses("s1").unwind(responseID: "r1") {
            read.append(item.text)
        }

        #expect(read == ["Hi", "Hello"])
        let request = try await server.request()
        #expect(request.path == "/v1/agents/sessions/s1/responses/items")
        #expect(request.query.contains("response_id=r1"))
        #expect(request.query.contains("limit=200"))
    }
}

@Suite struct ScheduledTasksTests {
    @Test func aTaskThatHasNotRunReadsWithItsNextOccurrences() async throws {
        let server = try SessionServer(
            answer: .init(
                status: 200, headers: ["Content-Type": "application/json"],
                body: #"""
                    {"id":"t1","user_id":"jlahey","config_id":"c1","session_id":"","title":"Stretch",
                     "instructions":"Remind them to stretch","schedule":{"at":"2026-10-28T09:00:00",
                     "repeat":"FREQ=DAILY","timezone":"Europe/Amsterdam","description":"Every day at 09:00 (Europe/Amsterdam)"},
                     "deliver_to":["conversation"],"state":"active","next_run_at":"2026-10-28T09:00:00+01:00",
                     "upcoming":["2026-10-28T09:00:00+01:00","2026-10-29T09:00:00+01:00"],"last_run_at":null,
                     "consecutive_failures":0,"created_by":"agent","created_at":"2026-10-27T08:00:00Z",
                     "updated_at":"2026-10-27T08:00:00Z"}
                    """#))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let task = try await agents.scheduledTasks.get("t1")

        #expect(task.state == .active)
        #expect(task.schedule.repeat == "FREQ=DAILY")
        #expect(task.schedule.description == "Every day at 09:00 (Europe/Amsterdam)")
        #expect(task.nextRunAt == Date(timeIntervalSince1970: 1_793_174_400))
        #expect(task.upcoming.count == 2)
        #expect(task.lastRunAt == nil)
        #expect(task.createdBy == .agent)
        #expect(try await server.request().path == "/v1/agents/scheduled-tasks/t1")
    }

    @Test func aSnoozeMovesOnlyTheNextOccurrence() async throws {
        let server = try SessionServer(answer: .init(status: 500, headers: [:], body: ""))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        _ = try? await agents.scheduledTasks.update("t1", next: .in("PT15M"))

        let request = try await server.request()
        #expect(request.method == "PATCH")
        #expect(request.path == "/v1/agents/scheduled-tasks/t1")
        #expect(request.body == ["next": .dictionary(["in": .string("PT15M")])])
    }

    @Test func aRunNotYetFinishedReadsWithoutAnEnd() async throws {
        let server = try SessionServer(
            answer: .init(
                status: 200, headers: ["Content-Type": "application/json"],
                body: #"""
                    {"items":[{"id":"run1","task_id":"t1","scheduled_for":"2026-10-28T09:00:00+01:00",
                     "trigger":"schedule","state":"running","delivered_to":"","session_id":"s1","attempts":1,
                     "started_at":"2026-10-28T08:00:01Z","finished_at":null,"error":null}],"has_more":false}
                    """#))
        defer { server.listener.cancel() }
        let agents = VisionAgents(url: try await server.url(), customerID: "acme")

        let runs = try await agents.scheduledTasks.runs("t1")

        let run = try #require(runs.items.first)
        #expect(run.state == .running)
        #expect(run.scheduledFor == Date(timeIntervalSince1970: 1_793_174_400))
        #expect(run.finishedAt == nil)
        #expect(run.error == "")
        #expect(try await server.request().path == "/v1/agents/scheduled-tasks/t1/runs")
    }
}

/// A real HTTP peer that answers one request with a session carrying what it was sent, or with
/// the answer it was given.
struct SessionServer {
    struct Request: Sendable {
        let method: String
        let path: String
        let query: String
        /// By lowercased name.
        let headers: [String: String]
        let body: [String: RawJSON]
    }

    /// What to answer instead of a session.
    struct Answer: Sendable {
        let status: Int
        let headers: [String: String]
        let body: String

        /// The session `s1`, on the call named `callID` or held in writing when it is empty.
        static func session(status: Int, callID: String) -> Answer {
            Answer(
                status: status, headers: ["Content-Type": "application/json"],
                body: #"{"id":"s1","agent_id":"a1","call_id":"\#(callID)","call_type":"agent","created_at":"2026-10-02T17:00:00Z","modality":"text","state":"live","user_id":"jlahey"}"#)
        }
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
        queue.asyncAfter(deadline: .now() + 30) {
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
                    let fields = (try? JSONDecoder().decode([String: RawJSON].self, from: body)) ?? [:]
                    let line = head[0].split(separator: " ").map(String.init)
                    let target = line[1].split(separator: "?", maxSplits: 1).map(String.init)
                    let path = target[0]
                    var headers: [String: String] = [:]
                    for field in head.dropFirst() {
                        let pair = field.split(separator: ":", maxSplits: 1)
                        guard pair.count == 2 else { continue }
                        headers[pair[0].lowercased()] = pair[1].trimmingCharacters(in: .whitespaces)
                    }
                    respond(connection, answer ?? session(path: path, fields: fields))
                    request.yield(
                        Request(
                            method: line[0], path: path, query: target.count > 1 ? target[1] : "",
                            headers: headers, body: fields))
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

    private static func session(path: String, fields: [String: RawJSON]) -> Answer {
        var session: [String: RawJSON] = [
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
