import Combine
import Foundation
import Network
import Testing

@testable import StreamAgents

@MainActor
@Suite struct ToolCallStateTests {
  private static let session = Session(
    .init(
      agentId: "test", callId: "", callType: "agent", createdAt: Date(), id: "test",
      modality: .text, state: .live, text: true, userId: "test"))

  private static func state(_ type: String, _ status: String, _ extra: String = "") -> String {
    #"{"type":"\#(type)","tool_call_id":"call-1","tool":"search_flights","turn_id":"t1","status":"\#(status)","started_at":"2026-11-18T12:00:00Z","duration_ms":300\#(extra)}"#
  }

  /// What a handler heard, kept off the stack because a main-actor closure may not mutate a
  /// local it captured.
  @MainActor private final class Heard {
    var progress: [String] = []
    var finished = 0
    var cancellable: AnyCancellable?
  }

  @Test func theSessionKeepsEachCallsLastStateAndAHandlerHearsItsKindUntilCancelled() async throws {
    let server = try FrameServer(frames: [
      Self.state("tool_call.started", "in_progress"),
      Self.state("tool_call.progress", "in_progress", #","progress":"Searching for flights""#),
      Self.state("tool_call.progress", "in_progress", #","progress":"Comparing prices""#),
      Self.state("tool_call.finished", "failed", #","error":"the airline is down""#),
    ])
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let session = AgentSession(backend: Backend(url: url, customerID: "test"), session: Self.session, tools: [])
    let heard = Heard()
    heard.cancellable = session.on(.toolCallProgress) { event in
      heard.progress.append(event.string("progress"))
      heard.cancellable?.cancel()
    }
    let finished = session.on(.toolCallFinished) { _ in heard.finished += 1 }
    await session.start()
    defer { Task { await session.close() } }

    let deadline = ContinuousClock.now + .seconds(30)
    while session.toolCalls["call-1"]?.status.isFinished != true, ContinuousClock.now < deadline {
      try await Task.sleep(for: .milliseconds(10))
    }

    let call = try #require(session.toolCalls["call-1"])
    #expect(call.status == .failed)
    #expect(call.error == "the airline is down")
    #expect(call.progress == "", "the finished frame carries the whole state")
    #expect(call.durationMS == 300)
    #expect(heard.progress == ["Searching for flights"])
    #expect(heard.finished == 1)
    finished.cancel()
  }

  @Test func aCallForAnotherWatcherIsNotRunAndOneAlreadyApprovedIsNotAskedAgain() async throws {
    let server = try FrameServer(frames: [
      #"{"type":"tool_call","id":"elsewhere","name":"lookup","arguments":"{}","run":false}"#,
      #"{"type":"tool_call","id":"approved","name":"issue_refund","arguments":"{}","approved":true}"#,
      #"{"type":"tool_call","id":"here","name":"lookup","arguments":"{}"}"#,
    ])
    defer { server.listener.cancel() }
    let url = try #require(try await server.addresses.first(where: { @Sendable _ in true }))
    let lookup = AgentTool(name: "lookup", description: "Look it up") { _ in "found" }
    let refund = AgentTool(
      name: "issue_refund", description: "Refund an order", approval: .init(title: "Refund it?")
    ) { _ in "refunded" }
    let session = AgentSession(
      backend: Backend(url: url, customerID: "test"), session: Self.session, tools: [lookup, refund])
    await session.start()
    defer { Task { await session.close() } }

    var answered: Set<String> = []
    for try await data in server.commands {
      let command = try JSONDecoder().decode([String: RawJSON].self, from: data)
      answered.insert(command["tool_call_id"]?.stringValue ?? "")
      if answered.isSuperset(of: ["here", "approved"]) { break }
    }

    #expect(answered == ["here", "approved"])
    #expect(session.approvals.isEmpty)
  }
}

/// A real WebSocket peer that sends its frames once the session connects and hands over every
/// frame the session sends back.
private struct FrameServer {
  let listener: NWListener
  let addresses: AsyncThrowingStream<URL, any Error>
  let commands: AsyncThrowingStream<Data, any Error>

  init(frames: [String]) throws {
    let queue = DispatchQueue(label: "frame-server")
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
      @Sendable func receive() {
        connection.receiveMessage { data, _, _, error in
          if let data { command.yield(data) }
          if let error {
            command.finish(throwing: error)
          } else {
            receive()
          }
        }
      }
      connection.stateUpdateHandler = { state in
        guard case .ready = state else { return }
        for (index, frame) in frames.enumerated() {
          let context = NWConnection.ContentContext(
            identifier: "frame-\(index)",
            metadata: [NWProtocolWebSocket.Metadata(opcode: .text)])
          connection.send(
            content: Data(frame.utf8), contentContext: context,
            completion: .contentProcessed { error in
              if let error { command.finish(throwing: error) }
            })
        }
        receive()
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
