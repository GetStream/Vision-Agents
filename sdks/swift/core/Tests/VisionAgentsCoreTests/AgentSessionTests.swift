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
