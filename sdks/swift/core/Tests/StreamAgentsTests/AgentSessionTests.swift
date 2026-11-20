import Foundation
import Network
import Testing
import os

@testable import StreamAgents

@MainActor
@Suite struct AgentSessionTests {
  /// The router refused the tools, so it offers none of them and nothing here answers them.
  @Test func aToolTheRouterRefusedIsNotAnswered() async throws {
    let server = try SessionServer(
      answer: .init(
        status: 409, headers: ["Content-Type": "application/json"],
        body: #"{"error":{"message":"taken","type":"conflict","code":"conflict","doc_url":""}}"#))
    defer { server.listener.cancel() }
    let session = AgentSession(
      backend: Backend(url: try await server.url(), customerID: "test"), session: Self.session, tools: [])

    await #expect(throws: AgentsError.self) { try await session.add(tools: [Self.reserveTable]) }

    #expect(session.tools["reserve_table"] == nil)
  }

  /// Whether a request that never came back added the tools is unknown, and a call of one that
  /// the router did add has to be answered.
  @Test func aToolWhoseAddingWasNeverAnsweredIsStillAnswered() async throws {
    let server = try SessionServer()
    let url = try await server.url()
    server.listener.cancel()
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"), session: Self.session, tools: [])

    await #expect(throws: AgentsError.self) { try await session.add(tools: [Self.reserveTable]) }

    #expect(session.tools["reserve_table"] != nil)
  }

  /// The router never replaces a tool the session has, so the handler answering that tool's
  /// calls stays the one it was added with, even when the request's answer is lost.
  @Test func aToolAddedUnderANameTheSessionHasKeepsItsHandler() async throws {
    let server = try SessionServer()
    let url = try await server.url()
    server.listener.cancel()
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"), session: Self.session,
      tools: [Self.reserveTable])
    let other = AgentTool(name: "reserve_table", description: "Reserve a table elsewhere") { _ in
      "Reserved elsewhere."
    }

    await #expect(throws: AgentsError.self) { try await session.add(tools: [other]) }

    #expect(session.tools["reserve_table"]?.description == "Reserve a table")
  }

  /// An add whose answer was lost may or may not have reached the router, so another tool
  /// under its name settles it first rather than leaving the first one answering calls the
  /// router now makes of the second.
  @Test func aToolWhoseAddingWasNeverAnsweredIsSettledBeforeAnotherTakesItsName() async throws {
    let server = try SessionServer { method, _, body in
      let described = body["tools"]?.arrayValue?.first?["description"]?.stringValue
      switch (method, described) {
      case ("POST", "Reserve a table"):
        return .init(status: 200, headers: ["Content-Type": "application/json"], body: "not json")
      case ("DELETE", _):
        return .init(
          status: 404, headers: ["Content-Type": "application/json"],
          body: #"{"error":{"message":"no tool","type":"not_found","code":"not_found","doc_url":""}}"#)
      default:
        return .init(status: 204, headers: [:], body: "")
      }
    }
    defer { server.listener.cancel() }
    let session = AgentSession(
      backend: Backend(url: try await server.url(), customerID: "test"), session: Self.session, tools: [])
    await #expect(throws: AgentsError.self) { try await session.add(tools: [Self.reserveTable]) }
    let other = AgentTool(name: "reserve_table", description: "Reserve a table elsewhere") { _ in
      "Reserved elsewhere."
    }

    try await session.add(tools: [other])

    #expect(session.tools["reserve_table"]?.description == "Reserve a table elsewhere")
  }

  /// A gateway in front of the router can answer a 5xx after the router added the tools, so
  /// they stay answered, and another tool under the name withdraws the first before it is added
  /// rather than being refused as one the session already has.
  @Test func aToolWhoseAddingAGatewayFailedIsStillAnsweredAndSettledBeforeAnotherTakesItsName()
    async throws
  {
    let router = OSAllocatedUnfairLock(initialState: (offered: Set<String>(), adds: 0))
    let server = try SessionServer { method, path, body in
      router.withLock { router in
        if method == "DELETE" {
          return router.offered.remove(String(path.split(separator: "/").last ?? "")) == nil
            ? .init(
              status: 404, headers: ["Content-Type": "application/json"],
              body: #"{"error":{"message":"no tool","type":"not_found","code":"not_found","doc_url":""}}"#)
            : .init(status: 204, headers: [:], body: "")
        }
        let names = Set(body["tools"]?.arrayValue?.compactMap { $0["name"]?.stringValue } ?? [])
        guard router.offered.isDisjoint(with: names) else {
          return .init(
            status: 409, headers: ["Content-Type": "application/json"],
            body: #"{"error":{"message":"taken","type":"conflict","code":"conflict","doc_url":""}}"#)
        }
        router.offered.formUnion(names)
        router.adds += 1
        return router.adds == 1 ? .badGateway : .init(status: 204, headers: [:], body: "")
      }
    }
    defer { server.listener.cancel() }
    let session = AgentSession(
      backend: Backend(url: try await server.url(), customerID: "test"), session: Self.session, tools: [])
    await #expect(throws: AgentsError.self) { try await session.add(tools: [Self.reserveTable]) }
    #expect(session.tools["reserve_table"]?.description == "Reserve a table", "the router offers it")
    let other = AgentTool(name: "reserve_table", description: "Reserve a table elsewhere") { _ in
      "Reserved elsewhere."
    }

    try await session.add(tools: [other])

    #expect(session.tools["reserve_table"]?.description == "Reserve a table elsewhere")
  }

  /// The router may have withdrawn a tool whose removal was never answered, or answered by a
  /// gateway's error, so asking again settles it when the router has none by that name, rather
  /// than failing.
  @Test(arguments: [SessionServer.Answer.dropped, .badGateway])
  func aToolWhoseRemovalWasNeverAnsweredIsRemovedByAskingAgain(lost: SessionServer.Answer) async throws {
    let server = try Self.losingTheFirstRemoval(lost)
    defer { server.listener.cancel() }
    let session = AgentSession(
      backend: Backend(url: try await server.url(), customerID: "test"), session: Self.session,
      tools: [Self.reserveTable])
    await #expect(throws: AgentsError.self) { try await session.remove(toolNamed: "reserve_table") }
    #expect(session.tools["reserve_table"] != nil, "the router may still offer it")

    try await session.remove(toolNamed: "reserve_table")

    #expect(session.tools["reserve_table"] == nil)
  }

  /// A removal whose answer was lost, or was a gateway's error, may or may not have reached the
  /// router, so another tool under its name settles it first rather than leaving the first one
  /// answering calls the router now makes of the second.
  @Test(arguments: [SessionServer.Answer.dropped, .badGateway])
  func aToolWhoseRemovalWasNeverAnsweredIsSettledBeforeAnotherTakesItsName(lost: SessionServer.Answer)
    async throws
  {
    let server = try Self.losingTheFirstRemoval(lost)
    defer { server.listener.cancel() }
    let session = AgentSession(
      backend: Backend(url: try await server.url(), customerID: "test"), session: Self.session,
      tools: [Self.reserveTable])
    await #expect(throws: AgentsError.self) { try await session.remove(toolNamed: "reserve_table") }
    let other = AgentTool(name: "reserve_table", description: "Reserve a table elsewhere") { _ in
      "Reserved elsewhere."
    }

    try await session.add(tools: [other])

    #expect(session.tools["reserve_table"]?.description == "Reserve a table elsewhere")
  }

  /// A router that withdraws the tool on the first removal but whose answer comes back as
  /// `lost`, so every removal after it finds no tool by that name.
  private static func losingTheFirstRemoval(_ lost: SessionServer.Answer) throws -> SessionServer {
    let removals = OSAllocatedUnfairLock(initialState: 0)
    return try SessionServer { method, _, _ in
      guard method == "DELETE" else { return .init(status: 204, headers: [:], body: "") }
      let first = removals.withLock {
        $0 += 1
        return $0 == 1
      }
      return first
        ? lost
        : .init(
          status: 404, headers: ["Content-Type": "application/json"],
          body: #"{"error":{"message":"no tool","type":"not_found","code":"not_found","doc_url":""}}"#)
    }
  }

  private static let session = Session(
    .init(
      agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
      modality: .text, state: .live, text: true, userId: "test"))

  private static let reserveTable = AgentTool(name: "reserve_table", description: "Reserve a table") { _ in
    "Reserved."
  }

  @Test(arguments: ["get_video_frames", "server_lookup"], [false, true])
  func onlyLocallyOwnedToolsAreAnswered(remoteTool: String, localFailure: Bool) async throws {
    let server = try ToolEventServer(remoteTool: remoteTool)
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let tool = AgentTool(name: "local_lookup", description: "Read local state") { _ in
      // The remote owner may be slower than the observing phone. Give an erroneous
      // response to the unowned request time to reach the server first.
      try await Task.sleep(for: .milliseconds(50))
      if localFailure { throw AgentsError.unreadable("local lookup failed") }
      return "local result"
    }
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"),
      session: Session(
        .init(
          agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
          modality: .text, state: .live, text: true, userId: "test")),
      tools: [tool])
    await session.start()
    do {
      let data = try #require(try await server.commands.first(where: { @Sendable _ in true }))
      let command = try JSONDecoder().decode([String: RawJSON].self, from: data)
      #expect(command["tool_call_id"]?.stringValue == "local")
      #expect(command["type"]?.stringValue == "tool_result")
      if localFailure {
        #expect(command["error"]?.stringValue?.contains("local lookup failed") == true)
      } else {
        #expect(command["output"]?.stringValue == "local result")
        #expect(command["error"]?.stringValue == "")
      }
      await session.close()
    } catch {
      await session.close()
      throw error
    }
  }

  @Test func aToolsPicturesGoBackBesideItsWords() async throws {
    let server = try ToolEventServer(remoteTool: "server_lookup")
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let tool = AgentTool(name: "local_lookup", description: "Photograph the room") { _ in
      ToolOutput(text: "the kitchen", images: [ImageSource(url: "data:image/png;base64,iVBORw0KGgo=")])
    }
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"),
      session: Session(
        .init(
          agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
          modality: .text, state: .live, text: true, userId: "test")),
      tools: [tool])
    await session.start()
    do {
      let data = try #require(try await server.commands.first(where: { @Sendable _ in true }))
      let command = try JSONDecoder().decode([String: RawJSON].self, from: data)
      #expect(
        command["output"]
          == .array([
            .dictionary(["type": .string("text"), "text": .string("the kitchen")]),
            .dictionary([
              "type": .string("image_url"),
              "image_url": .dictionary(["url": .string("data:image/png;base64,iVBORw0KGgo=")]),
            ]),
          ]))
      await session.close()
    } catch {
      await session.close()
      throw error
    }
  }

  private struct Booking: Decodable {
    let location: String
    let guests: Int
  }

  /// A typed tool runs with the model's arguments decoded, and arguments missing a field reach
  /// the model as the tool failing, naming the field, rather than running it with a blank.
  @Test(arguments: [
    (#"{"location":"Boulder","guests":4}"#, "Booked Boulder for 4", ""),
    (#"{"guests":4}"#, nil, #""location""#),
    (#"{"location":"Boulder","guests":"four"}"#, nil, #""guests""#),
  ])
  func aTypedToolGetsItsArgumentsDecoded(arguments: String, output: String?, error: String) async throws {
    let server = try ToolEventServer(remoteTool: "server_lookup", arguments: arguments)
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let tool = AgentTool(name: "local_lookup", description: "Book a table") { (booking: Booking) in
      "Booked \(booking.location) for \(booking.guests)"
    }
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"), session: Self.session, tools: [tool])
    await session.start()
    do {
      let data = try #require(try await server.commands.first(where: { @Sendable _ in true }))
      let command = try JSONDecoder().decode([String: RawJSON].self, from: data)
      #expect(command["type"]?.stringValue == "tool_result")
      if let output {
        #expect(command["output"]?.stringValue == output)
        #expect(command["error"]?.stringValue == "")
      } else {
        #expect(command["output"]?.stringValue == "")
        #expect(command["error"]?.stringValue?.contains(error) == true, "\(command["error"] ?? .nil)")
      }
      await session.close()
    } catch {
      await session.close()
      throw error
    }
  }

  @Test(arguments: [true, false])
  func aCameraOnACallAnswersTheRequestToLook(recent: Bool) async throws {
    let server = try ToolEventServer(remoteTool: "get_video_frames")
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let frames = CameraFrames()
    frames.record(Data([0xff, 0xd8, 0xff]), at: recent ? Date() : Date().addingTimeInterval(-60))
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"),
      session: Session(
        .init(
          agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
          modality: .voice, state: .live, text: false, userId: "test")),
      tools: [])
    session.cameraFrames = frames
    await session.start()
    do {
      let data = try #require(try await server.commands.first(where: { @Sendable _ in true }))
      let command = try JSONDecoder().decode([String: RawJSON].self, from: data)
      #expect(command["tool_call_id"]?.stringValue == "remote")
      if recent {
        let parts = try #require(command["output"]?.arrayValue)
        #expect(parts.count == 2, "what the frame is, then the frame")
        #expect(parts.first?["text"]?.stringValue?.contains("captured_at_ms") == true)
        #expect(parts.last?["image_url"]?["url"]?.stringValue == "data:image/jpeg;base64,/9j/")
      } else {
        #expect(command["output"]?.stringValue?.contains("point it at what they mean again") == true)
      }
      await session.close()
    } catch {
      await session.close()
      throw error
    }
  }

  @Test(arguments: [true, false])
  func aToolThatAsksFirstWaitsForThePerson(allowed: Bool) async throws {
    let server = try ApprovalServer()
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let refund = AgentTool(
      name: "issue_refund", description: "Refund an order",
      approval: .init(title: "Refund order A-1042?", reasonArgument: "reason")
    ) { arguments in
      "Refunded \(arguments["order_id"]?.stringValue ?? "")"
    }
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"),
      session: Session(
        .init(
          agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
          modality: .text, state: .live, text: true, userId: "test")),
      tools: [refund])
    await session.start()
    defer { Task { await session.close() } }

    let deadline = ContinuousClock.now + .seconds(30)
    while session.approvals.isEmpty, ContinuousClock.now < deadline {
      try await Task.sleep(for: .milliseconds(10))
    }
    let waiting = try #require(session.approvals.first)
    #expect(waiting.approval.title == "Refund order A-1042?")
    #expect(waiting.reason == "it arrived unopened")

    try await session.decide("c1", allowed: allowed, summary: allowed ? "" : "Kept the order")

    #expect(session.approvals.isEmpty)
    var commands = server.commands.makeAsyncIterator()
    let answer = try JSONDecoder().decode(
      [String: RawJSON].self, from: try #require(try await commands.next()))
    #expect(answer["type"]?.stringValue == "tool_approval")
    #expect(answer["tool_call_id"]?.stringValue == "c1")
    #expect(answer["allowed"]?.boolValue == allowed)
    #expect(answer["request_id"]?.stringValue == "m1")
    #expect(answer["turn_id"]?.stringValue == "t1")
    let result = try JSONDecoder().decode(
      [String: RawJSON].self, from: try #require(try await commands.next()))
    #expect(result["type"]?.stringValue == "tool_result")
    #expect(result["tool_call_id"]?.stringValue == "c1")
    if allowed {
      #expect(result["output"]?.stringValue == "Refunded A-1042")
    } else {
      #expect(answer["summary"]?.stringValue == "Kept the order")
      #expect(result["output"]?.stringValue == "")
      #expect(result["error"]?.stringValue?.isEmpty == false)
    }
  }

  @Test func aDeclinedLoginNamesOnlyTheCallItWas() async throws {
    let server = try ApprovalServer()
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"),
      session: Session(
        .init(
          agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
          modality: .text, state: .live, text: true, userId: "test")),
      tools: [])
    await session.start()
    defer { Task { await session.close() } }

    try await session.declineLogin("call-linear")

    var commands = server.commands.makeAsyncIterator()
    let answer = try JSONDecoder().decode(
      [String: RawJSON].self, from: try #require(try await commands.next()))
    #expect(
      answer == [
        "type": .string("tool_approval"), "tool_call_id": .string("call-linear"), "allowed": .bool(false),
      ])
  }

  @Test func everyReaderOfTheEventsGetsEachEventAndTheStreamEndsWithTheSession() async throws {
    let server = try ApprovalServer()
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"),
      session: Session(
        .init(
          agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
          modality: .text, state: .live, text: true, userId: "test")),
      tools: [])
    var first = session.events().makeAsyncIterator()
    var second = session.events().makeAsyncIterator()

    await session.start()
    let seen = [await first.next(), await second.next()]
    await session.close()

    #expect(seen.map { $0?.string("id") } == ["c1", "c1"])
    #expect(await first.next() == nil)
    var late = session.events().makeAsyncIterator()
    #expect(await late.next() == nil)
  }

  @Test func sayingSomethingOverTheAgentStopsItFirst() async throws {
    let server = try ApprovalServer()
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"),
      session: Session(
        .init(
          agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
          modality: .voice, state: .live, text: false, userId: "test")),
      tools: [])
    await session.start()
    defer { Task { await session.close() } }

    try await session.say("One moment.", interrupt: true)

    var commands = server.commands.makeAsyncIterator()
    let sent = try [#require(try await commands.next()), #require(try await commands.next())].map {
      try JSONDecoder().decode([String: RawJSON].self, from: $0)
    }
    #expect(sent == [["type": .string("interrupt")], ["type": .string("say"), "text": .string("One moment.")]])
  }

  @Test func anUpdateRefreshesTheSessionItHolds() async throws {
    let server = try SessionServer()
    defer { server.listener.cancel() }
    let session = AgentSession(
      backend: Backend(url: try await server.url(), customerID: "test"),
      session: Session(
        .init(
          agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
          modality: .text, state: .live, text: true, title: "Before", userId: "test")),
      tools: [])

    try await session.update(title: "After")

    #expect(session.session.title == "After")
    #expect(try await server.request().body == ["title": .string("After")])
  }

  @Test(arguments: [("POST", "s1"), ("DELETE", "")])
  func startingAndStoppingVoiceMovesTheSessionOntoAndOffItsCall(method: String, callID: String)
    async throws
  {
    let server = try SessionServer(answer: .session(status: 200, callID: callID))
    defer { server.listener.cancel() }
    let session = AgentSession(
      backend: Backend(url: try await server.url(), customerID: "test"),
      session: Session(
        .init(
          agentId: "a1", callId: method == "POST" ? "" : "s1", callType: "agent",
          createdAt: Date(), id: "s1", modality: .text, state: .live, userId: "jlahey")),
      tools: [])

    if method == "POST" {
      try await session.startVoice()
    } else {
      try await session.stopVoice()
    }

    #expect(session.session.callID == callID)
    let request = try await server.request()
    #expect(request.method == method)
    #expect(request.path == "/v1/agents/sessions/s1/voice")
  }
}

/// A real WebSocket peer broadcasting one remote tool and one locally owned tool.
private struct ToolEventServer {
  let listener: NWListener
  let addresses: AsyncThrowingStream<URL, any Error>
  let commands: AsyncThrowingStream<Data, any Error>

  init(remoteTool: String, arguments: String = "{}") throws {
    let quoted = String(decoding: try JSONEncoder().encode(arguments), as: UTF8.self)
    let queue = DispatchQueue(label: "tool-event-server")
    let webSocket = NWProtocolWebSocket.Options()
    webSocket.autoReplyPing = true
    webSocket.setClientRequestHandler(queue) { _, _ in
      .init(status: .accept, subprotocol: nil)
    }
    let parameters = NWParameters.tcp
    parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
    parameters.defaultProtocolStack.applicationProtocols.insert(webSocket, at: 0)
    let listener = try NWListener(using: parameters)
    self.listener = listener
    let (addresses, address) = AsyncThrowingStream<URL, any Error>.makeStream()
    let (commands, command) = AsyncThrowingStream<Data, any Error>.makeStream()
    self.addresses = addresses
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
        command.finish(throwing: error)
      default: break
      }
    }
    listener.newConnectionHandler = { connection in
      connection.stateUpdateHandler = { state in
        guard case .ready = state else { return }
        for (id, name) in [("remote", remoteTool), ("local", "local_lookup")] {
          let frame = #"{"type":"tool_call","id":"\#(id)","name":"\#(name)","arguments":\#(quoted)}"#
          let context = NWConnection.ContentContext(
            identifier: id,
            metadata: [NWProtocolWebSocket.Metadata(opcode: .text)])
          connection.send(
            content: Data(frame.utf8), contentContext: context,
            completion: .contentProcessed { error in
              if let error { command.finish(throwing: error) }
            })
        }
        connection.receiveMessage { data, _, _, error in
          if let data { command.yield(data) }
          command.finish(throwing: error)
          connection.cancel()
        }
      }
      connection.start(queue: queue)
      queue.asyncAfter(deadline: .now() + 30) { connection.cancel() }
    }
    queue.asyncAfter(deadline: .now() + 30) {
      let error = AgentsError.unreadable("test socket timed out")
      address.finish(throwing: error)
      command.finish(throwing: error)
    }
    listener.start(queue: queue)
  }
}

/// A real WebSocket peer that asks for one call of a tool that asks first, from a durable
/// command, and hands over every frame the session sends back.
private struct ApprovalServer {
  let listener: NWListener
  let addresses: AsyncThrowingStream<URL, any Error>
  let commands: AsyncThrowingStream<Data, any Error>

  init() throws {
    let queue = DispatchQueue(label: "approval-server")
    let webSocket = NWProtocolWebSocket.Options()
    webSocket.autoReplyPing = true
    webSocket.setClientRequestHandler(queue) { _, _ in
      .init(status: .accept, subprotocol: nil)
    }
    let parameters = NWParameters.tcp
    parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
    parameters.defaultProtocolStack.applicationProtocols.insert(webSocket, at: 0)
    let listener = try NWListener(using: parameters)
    self.listener = listener
    let (addresses, address) = AsyncThrowingStream<URL, any Error>.makeStream()
    let (commands, command) = AsyncThrowingStream<Data, any Error>.makeStream()
    self.addresses = addresses
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
        command.finish(throwing: error)
      default: break
      }
    }
    @Sendable func receive(_ connection: NWConnection) {
      connection.receiveMessage { data, _, _, error in
        if let error {
          command.finish(throwing: error)
          return
        }
        if let data { command.yield(data) }
        receive(connection)
      }
    }
    listener.newConnectionHandler = { connection in
      connection.stateUpdateHandler = { state in
        guard case .ready = state else { return }
        let frame =
          #"{"type":"tool_call","id":"c1","name":"issue_refund","arguments":"{\"order_id\":\"A-1042\",\"reason\":\"it arrived unopened\"}","request_id":"m1","turn_id":"t1"}"#
        connection.send(
          content: Data(frame.utf8),
          contentContext: NWConnection.ContentContext(
            identifier: "call", metadata: [NWProtocolWebSocket.Metadata(opcode: .text)]),
          completion: .contentProcessed { error in
            if let error { command.finish(throwing: error) }
          })
        receive(connection)
      }
      connection.start(queue: queue)
      queue.asyncAfter(deadline: .now() + 30) { connection.cancel() }
    }
    queue.asyncAfter(deadline: .now() + 30) {
      let error = AgentsError.unreadable("test socket timed out")
      address.finish(throwing: error)
      command.finish(throwing: error)
    }
    listener.start(queue: queue)
  }
}
