import Foundation
import Network
import Testing

@testable import VisionAgentsCore

@MainActor
@Suite struct AgentSessionTests {
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
      let command = try JSONDecoder().decode([String: JSONValue].self, from: data)
      #expect(command["tool_call_id"]?.stringValue == "local")
      #expect(command["type"]?.stringValue == "tool_result")
      if localFailure {
        #expect(command["error"]?.stringValue.contains("local lookup failed") == true)
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

    let deadline = ContinuousClock.now + .seconds(5)
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
      [String: JSONValue].self, from: try #require(try await commands.next()))
    #expect(answer["type"]?.stringValue == "tool_approval")
    #expect(answer["tool_call_id"]?.stringValue == "c1")
    #expect(answer["allowed"]?.boolValue == allowed)
    #expect(answer["request_id"]?.stringValue == "m1")
    #expect(answer["turn_id"]?.stringValue == "t1")
    let result = try JSONDecoder().decode(
      [String: JSONValue].self, from: try #require(try await commands.next()))
    #expect(result["type"]?.stringValue == "tool_result")
    #expect(result["tool_call_id"]?.stringValue == "c1")
    if allowed {
      #expect(result["output"]?.stringValue == "Refunded A-1042")
    } else {
      #expect(answer["summary"]?.stringValue == "Kept the order")
      #expect(result["output"]?.stringValue == "")
      #expect(result["error"]?.stringValue.isEmpty == false)
    }
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

  init(remoteTool: String) throws {
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
          let frame = #"{"type":"tool_call","id":"\#(id)","name":"\#(name)","arguments":"{}"}"#
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
      queue.asyncAfter(deadline: .now() + 5) { connection.cancel() }
    }
    queue.asyncAfter(deadline: .now() + 5) {
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
      queue.asyncAfter(deadline: .now() + 5) { connection.cancel() }
    }
    queue.asyncAfter(deadline: .now() + 5) {
      let error = AgentsError.unreadable("test socket timed out")
      address.finish(throwing: error)
      command.finish(throwing: error)
    }
    listener.start(queue: queue)
  }
}
