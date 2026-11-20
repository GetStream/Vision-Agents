import CryptoKit
import Foundation
import Network
import Testing

@testable import StreamAgents

@Suite struct SessionsTests {
    @Test func anUpdateSendsOnlyWhatChangedAndReturnsTheSession() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

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
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
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
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

        try await agents.sessions.delete("s1")

        let request = try await server.request()
        #expect(request.method == "DELETE")
        #expect(request.path == "/v1/agents/sessions/s1")
    }

    @Test func aVoiceSessionAsksToStartVoiceAndIsOnItsOwnCall() async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: "s1"))
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
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
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
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
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
        var options = SessionOptions(agent: "myagent")
        options.returnURL = URL(string: "com.example.app:/connected")

        _ = try await agents.createSession(options)

        #expect(try await server.request().body["return_url"] == .string("com.example.app:/connected"))
    }

    @Test func aSessionOpensWithItsLabelsModelsAndTheSignedInUsersMemory() async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: ""))
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
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
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
        agents.setUser(User(id: "jlahey"), token: "token")

        _ = try await agents.createSession(SessionOptions(agent: "myagent"))

        let body = try await server.request().body
        #expect(body["memory"] == nil)
        #expect(body["custom"] == nil)
        #expect(body["model_overwrites"] == nil)
    }

    @Test(arguments: ["Asia/Tokyo", "America/Argentina/Buenos_Aires", "UTC", "GMT"])
    func aSessionIsOpenedInTheDevicesZone(zone: String) async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: ""))
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

        _ = try await agents.createSession(SessionOptions(agent: "myagent", deviceZone: zone))

        #expect(try await server.request().body["timezone"] == .string(zone))
    }

    @Test func theDevicesZoneIsTheSystemsOwn() {
        #expect(SessionOptions().timezone == SessionOptions.ianaZone(TimeZone.current.identifier))
        #expect(SessionOptions(agent: "myagent").timezone == SessionOptions().timezone)
    }

    @Test(arguments: ["GMT+0100", "", "Local"])
    func aDeviceZoneWithNoIANANameIsNotSent(zone: String) async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: ""))
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

        _ = try await agents.createSession(SessionOptions(agent: "myagent", deviceZone: zone))

        #expect(try await server.request().body["timezone"] == nil)
    }

    // The router is what judges a zone the app names, so even one with no IANA name goes as it is.
    @Test(arguments: ["Europe/Amsterdam", "GMT+0100"])
    func theAppsOwnZoneIsSentOverTheDevices(zone: String) async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: ""))
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
        var options = SessionOptions(agent: "myagent", deviceZone: "Asia/Tokyo")
        options.timezone = zone

        _ = try await agents.createSession(options)

        #expect(try await server.request().body["timezone"] == .string(zone))
    }

    @Test func aSessionToldNoZoneSendsNone() async throws {
        let server = try SessionServer(answer: .session(status: 201, callID: ""))
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
        var options = SessionOptions(agent: "myagent", deviceZone: "Asia/Tokyo")
        options.timezone = nil

        _ = try await agents.createSession(options)

        #expect(try await server.request().body["timezone"] == nil)
    }

    @Test func aForkNamesTheAgentAndWhatItChanges() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
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
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

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

    private static let lookup = AgentTool(name: "local_lookup", description: "Read local state") { _ in
        "local result"
    }

    /// A session opened is watched by the time it returns, so the router's call of one of its
    /// tools is answered with nothing more asked of the caller.
    @MainActor
    @Test func aSessionOpenedAnswersItsToolsWithoutBeingStarted() async throws {
        let server = try RouterServer()
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

        let session = try await agents.agent("myagent").sessions.create(SessionOptions(tools: [Self.lookup]))
        defer { Task { await session.close() } }

        #expect(session.isConnected)
        var requests = server.requests.makeAsyncIterator()
        let created = try #require(try await requests.next())
        #expect(created.path == "/v1/agents/sessions")
        #expect(created.body["agent"] == .string("myagent"))
        #expect(created.body["tools"]?.arrayValue?.first?["name"] == .string("local_lookup"))
        let watched = try #require(try await requests.next())
        #expect(watched.method == "GET")
        #expect(watched.path == "/v1/agents/sessions/s1/events")
        let answer = try JSONDecoder().decode(
            [String: RawJSON].self, from: try #require(try await server.commands.first { @Sendable _ in true }))
        #expect(answer["type"] == .string("tool_result"))
        #expect(answer["tool_call_id"] == .string("c1"))
        #expect(answer["output"] == .string("local result"))
    }

    /// A session opened with `watch: false` has no socket and declares no tools, so what needs
    /// the socket says how to do without it, and closing it stops it over HTTP.
    @MainActor
    @Test func aSessionOpenedUnwatchedHasNoSocketAndSaysWhatToDoInstead() async throws {
        let server = try RouterServer()
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")
        var options = SessionOptions()
        options.tools = [Self.lookup]
        options.watch = false

        let session = try await agents.agent("myagent").sessions.create(options)
        await session.start()

        #expect(!session.isConnected)
        let said = await #expect(throws: AgentsError.self) { try await session.say("One moment.") }
        let interrupted = await #expect(throws: AgentsError.self) { try await session.interrupt() }
        for error in [said, interrupted] {
            guard case .configuration(let message) = error else {
                Issue.record("\(String(describing: error)) does not say how to do without the socket")
                continue
            }
            #expect(message.contains("without watch: false"))
            #expect(message.contains("responses.create"))
        }
        await session.close()
        var requests = server.requests.makeAsyncIterator()
        let created = try #require(try await requests.next())
        #expect(created.path == "/v1/agents/sessions")
        #expect(created.body["tools"]?.arrayValue?.isEmpty ?? true, "no tool is declared that nothing here runs")
        let stopped = try #require(try await requests.next())
        #expect(stopped.method == "POST")
        #expect(stopped.path == "/v1/agents/sessions/s1/stop")
    }

    /// `chat` opens a session whose socket waits for `start`, which then answers its tools.
    @MainActor
    @Test func aChatAnswersItsToolsOnceStarted() async throws {
        let server = try RouterServer()
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

        let session = try await agents.chat(agent: "myagent", tools: [Self.lookup])
        defer { Task { await session.close() } }
        #expect(!session.isConnected)
        await session.start()

        let answer = try JSONDecoder().decode(
            [String: RawJSON].self, from: try #require(try await server.commands.first { @Sendable _ in true }))
        #expect(answer["tool_call_id"] == .string("c1"))
        #expect(answer["output"] == .string("local result"))
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
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

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
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

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
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

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

        /// No answer at all: the connection closes once the request has arrived, as though it
        /// dropped on the way back.
        static let dropped = Answer(status: 0, headers: [:], body: "")

        /// A gateway's answer from in front of the router, which says nothing of whether the
        /// router did what it was asked.
        static let badGateway = Answer(
            status: 502, headers: ["Content-Type": "text/html"], body: "<html><body>502 Bad Gateway</body></html>")

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
        try self.init { _, _, _ in answer }
    }

    /// A server answering each request by its method, path and body, or with a session for
    /// a nil answer.
    init(answering: @escaping @Sendable (String, String, [String: RawJSON]) -> Answer?) throws {
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
            Self.read(connection, Data(), answering, request)
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
        _ connection: NWConnection, _ buffer: Data,
        _ answering: @escaping @Sendable (String, String, [String: RawJSON]) -> Answer?,
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
                    respond(connection, answering(line[0], path, fields) ?? session(path: path, fields: fields))
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
                read(connection, buffer, answering, request)
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
        if fields["pinned"] == .bool(true) { session["pinned_at"] = .string("2026-11-11T09:30:00Z") }
        let body = (try? JSONEncoder().encode(session)) ?? Data()
        return Answer(
            status: 200, headers: ["Content-Type": "application/json"],
            body: String(decoding: body, as: UTF8.self))
    }

    private static func respond(_ connection: NWConnection, _ answer: Answer) {
        guard answer.status != 0 else { return connection.cancel() }
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

/// Both halves of a router on one port, as a session uses them: HTTP, answered with the session
/// `s1`, and the session's socket, upgraded here by hand, which asks for one call of
/// `local_lookup` once it is open and hands over every frame sent back.
private struct RouterServer {
    struct Request: Sendable {
        let method: String
        let path: String
        let body: [String: RawJSON]
    }

    let listener: NWListener
    /// Every request in the order it arrived, the socket's upgrade included.
    let requests: AsyncThrowingStream<Request, any Error>
    let commands: AsyncThrowingStream<Data, any Error>
    private let addresses: AsyncThrowingStream<URL, any Error>

    init() throws {
        let queue = DispatchQueue(label: "router-server")
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        let listener = try NWListener(using: parameters)
        self.listener = listener
        let (addresses, address) = AsyncThrowingStream<URL, any Error>.makeStream()
        let (requests, request) = AsyncThrowingStream<Request, any Error>.makeStream()
        let (commands, command) = AsyncThrowingStream<Data, any Error>.makeStream()
        self.addresses = addresses
        self.requests = requests
        self.commands = commands
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
                command.finish(throwing: error)
            default: break
            }
        }
        listener.newConnectionHandler = { connection in
            connection.start(queue: queue)
            Self.read(connection, Data(), request, command)
        }
        queue.asyncAfter(deadline: .now() + 30) {
            let error = AgentsError.unreadable("test server timed out")
            address.finish(throwing: error)
            request.finish(throwing: error)
            command.finish(throwing: error)
        }
        listener.start(queue: queue)
    }

    func url() async throws -> URL {
        try #require(try await addresses.first(where: { @Sendable _ in true }))
    }

    private static func read(
        _ connection: NWConnection, _ buffer: Data,
        _ request: AsyncThrowingStream<Request, any Error>.Continuation,
        _ command: AsyncThrowingStream<Data, any Error>.Continuation
    ) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, done, error in
            var buffer = buffer
            if let data { buffer.append(data) }
            guard let end = buffer.range(of: Data("\r\n\r\n".utf8)) else {
                if error == nil, !done { read(connection, buffer, request, command) }
                return
            }
            let head = String(decoding: buffer[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
            var headers: [String: String] = [:]
            for field in head.dropFirst() {
                let pair = field.split(separator: ":", maxSplits: 1)
                guard pair.count == 2 else { continue }
                headers[pair[0].lowercased()] = pair[1].trimmingCharacters(in: .whitespaces)
            }
            let length = Int(headers["content-length"] ?? "") ?? 0
            let body = Data(buffer[end.upperBound...])
            guard body.count >= length else {
                if error == nil, !done { read(connection, buffer, request, command) }
                return
            }
            let line = head[0].split(separator: " ").map(String.init)
            let path = String(line[1].split(separator: "?", maxSplits: 1)[0])
            request.yield(
                Request(
                    method: line[0], path: path,
                    body: (try? JSONDecoder().decode([String: RawJSON].self, from: body)) ?? [:]))
            if let key = headers["sec-websocket-key"] {
                upgrade(connection, key: key, command)
            } else {
                respond(connection, path: path)
            }
        }
    }

    private static func respond(_ connection: NWConnection, path: String) {
        let body =
            path == "/v1/agents/sessions"
            ? #"{"id":"s1","agent_id":"a1","call_id":"","call_type":"agent","created_at":"2026-10-02T17:00:00Z","modality":"text","state":"live","user_id":"jlahey"}"#
            : ""
        let head =
            body.isEmpty
            ? "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n"
            : "HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n"
        connection.send(
            content: Data((head + body).utf8), completion: .contentProcessed { _ in connection.cancel() })
    }

    private static func upgrade(
        _ connection: NWConnection, key: String, _ command: AsyncThrowingStream<Data, any Error>.Continuation
    ) {
        let accept = Data(Insecure.SHA1.hash(data: Data((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").utf8)))
            .base64EncodedString()
        let head =
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            + "Sec-WebSocket-Accept: \(accept)\r\n\r\n"
        let call = #"{"type":"tool_call","id":"c1","name":"local_lookup","arguments":"{}"}"#
        connection.send(content: Data(head.utf8) + frame(0x1, Data(call.utf8)), completion: .idempotent)
        listen(connection, Data(), command)
    }

    /// Reads the frames the session sends, answering its pings.
    private static func listen(
        _ connection: NWConnection, _ buffer: Data, _ command: AsyncThrowingStream<Data, any Error>.Continuation
    ) {
        var buffer = buffer
        while let next = nextFrame(&buffer) {
            let (opcode, payload) = next
            switch opcode {
            case 0x1, 0x2: command.yield(payload)
            case 0x9: connection.send(content: frame(0xA, payload), completion: .idempotent)
            case 0x8: return connection.cancel()
            default: break
            }
        }
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, done, error in
            guard let data, error == nil else { return }
            listen(connection, buffer + data, command)
            if done { connection.cancel() }
        }
    }

    /// A server's frame, which is never masked.
    private static func frame(_ opcode: UInt8, _ payload: Data) -> Data {
        var data = Data([0x80 | opcode])
        if payload.count < 126 {
            data.append(UInt8(payload.count))
        } else {
            data.append(contentsOf: [126, UInt8(payload.count >> 8), UInt8(payload.count & 0xff)])
        }
        return data + payload
    }

    /// Takes the first whole frame off `buffer`, unmasking it, or nil while it has none.
    private static func nextFrame(_ buffer: inout Data) -> (UInt8, Data)? {
        let bytes = [UInt8](buffer)
        guard bytes.count >= 2 else { return nil }
        var length = Int(bytes[1] & 0x7f)
        var offset = 2
        if length == 126 {
            guard bytes.count >= 4 else { return nil }
            length = Int(bytes[2]) << 8 | Int(bytes[3])
            offset = 4
        } else if length == 127 {
            guard bytes.count >= 10 else { return nil }
            length = bytes[2..<10].reduce(0) { $0 << 8 | Int($1) }
            offset = 10
        }
        let masked = bytes[1] & 0x80 != 0
        guard bytes.count >= offset + (masked ? 4 : 0) + length else { return nil }
        let mask = masked ? Array(bytes[offset..<offset + 4]) : [0, 0, 0, 0]
        if masked { offset += 4 }
        let payload = Data((0..<length).map { bytes[offset + $0] ^ mask[$0 % 4] })
        buffer = Data(bytes[(offset + length)...])
        return (bytes[0] & 0x0f, payload)
    }
}
