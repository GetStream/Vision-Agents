import Foundation

/// The socket carrying one conversation.
///
/// This is hand-written because OpenAPI stops at the upgrade. It is an actor so that the read
/// loop and the sends cannot race over the task, and so that closing twice is harmless.
///
/// `URLSessionWebSocketTask` answers the router's pings itself, so there is no keepalive here.
/// What it does not do is notice a task that was torn down while the app was in the
/// background: that arrives as a read failure, which ends the event stream.
/// How big one frame may be. A transcript event carries a sentence, and a tool call carries the
/// arguments a model wrote, so a megabyte is generous.
private let maximumFrameSize = 1 << 20

public actor SessionSocket {
    private let url: URL
    private let headers: [String: String]
    private let urlSession: URLSession
    private var task: URLSessionWebSocketTask?
    private var reader: Task<Void, Never>?

    public init(url: URL, headers: [String: String], urlSession: URLSession = .shared) {
        self.url = url
        self.headers = headers
        self.urlSession = urlSession
    }

    /// Opens the socket and starts reading.
    ///
    /// The stream is returned rather than exposed as a property so that it cannot be attached
    /// to twice, and so there is no window between opening and attaching in which an event
    /// could be dropped: reading does not begin until the stream exists.
    public func open() -> AsyncThrowingStream<AgentEvent, any Error> {
        var request = URLRequest(url: url)
        for (name, value) in headers {
            request.setValue(value, forHTTPHeaderField: name)
        }

        let task = urlSession.webSocketTask(with: request)
        // A conversation's frames are a few kilobytes at most. Without a ceiling, a socket
        // that has gone wrong at the other end can make this process buffer without bound.
        task.maximumMessageSize = maximumFrameSize
        self.task = task
        task.resume()

        // Unbounded because dropping an event would lose a tool call or the end of a reply.
        // The producer is a network socket, so the queue is bounded in practice by what the
        // router sends.
        let (stream, continuation) = AsyncThrowingStream.makeStream(
            of: AgentEvent.self, bufferingPolicy: .unbounded)

        reader = Task { [weak self] in
            await self?.read(task: task, into: continuation)
        }

        return stream
    }

    /// Sends one command. Fails if the socket is not open.
    public func send(_ command: Command) async throws {
        guard let task else {
            throw AgentsError.socketClosed(code: 0, reason: "the socket is not open")
        }
        let encoded = try JSONEncoder().encode(command)
        do {
            try await task.send(.data(encoded))
        } catch {
            throw AgentsError.transport(error)
        }
    }

    /// Closes the socket. Safe to call more than once.
    public func close() {
        reader?.cancel()
        reader = nil
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
    }

    private func read(
        task: URLSessionWebSocketTask,
        into continuation: AsyncThrowingStream<AgentEvent, any Error>.Continuation
    ) async {
        let decoder = JSONDecoder()
        while !Task.isCancelled {
            let message: URLSessionWebSocketTask.Message
            do {
                message = try await task.receive()
            } catch {
                // A cancelled read is a close we asked for, not a failure to report.
                if Task.isCancelled {
                    continuation.finish()
                } else {
                    continuation.finish(
                        throwing: AgentsError.socketClosed(
                            code: task.closeCode.rawValue,
                            reason: (error as NSError).localizedDescription))
                }
                return
            }

            let payload: Data
            switch message {
            case .data(let data):
                payload = data
            case .string(let text):
                payload = Data(text.utf8)
            @unknown default:
                continue
            }

            // A frame this SDK cannot read is skipped rather than ending the conversation,
            // which is what the router does with a command it cannot read.
            if let event = try? decoder.decode(AgentEvent.self, from: payload) {
                continuation.yield(event)
            }
        }
        continuation.finish()
    }
}
