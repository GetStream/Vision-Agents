import Foundation
import StreamCore

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
    /// Whether the router took the last `open`'s upgrade, so the conversation's frames reach it.
    private(set) var isOpen = false

    public init(url: URL, headers: [String: String], urlSession: URLSession = .shared) {
        self.url = url
        self.headers = headers
        self.urlSession = urlSession
    }

    /// Opens the socket and starts reading, returning once the upgrade is answered, either way.
    ///
    /// The stream is returned rather than exposed as a property so that it cannot be attached
    /// to twice, and so there is no window between opening and attaching in which an event
    /// could be dropped: reading does not begin until the stream exists. `headers` are added
    /// to the ones this socket was made with, for credentials only known once it opens. A
    /// refused upgrade ends the stream with what the router answered.
    public func open(headers extra: [String: String] = [:]) async -> AsyncThrowingStream<AgentEvent, any Error> {
        var request = URLRequest(url: url)
        for (name, value) in headers.merging(extra, uniquingKeysWith: { _, new in new }) {
            request.setValue(value, forHTTPHeaderField: name)
        }

        let task = urlSession.webSocketTask(with: request)
        // A conversation's frames are a few kilobytes at most. Without a ceiling, a socket
        // that has gone wrong at the other end can make this process buffer without bound.
        task.maximumMessageSize = maximumFrameSize
        let upgrade = Upgrade()
        task.delegate = upgrade
        self.task = task
        isOpen = false
        task.resume()

        // Unbounded because dropping an event would lose a tool call or the end of a reply.
        // The producer is a network socket, so the queue is bounded in practice by what the
        // router sends.
        let (stream, continuation) = AsyncThrowingStream.makeStream(
            of: AgentEvent.self, bufferingPolicy: .unbounded)

        reader = Task { [weak self] in
            await self?.read(task: task, into: continuation)
        }

        let opened = await upgrade.answered()
        if self.task === task {
            isOpen = opened
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
        isOpen = false
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
                } else if let refused = task.response as? HTTPURLResponse, refused.statusCode != 101 {
                    // URLSession hands over a refused upgrade's status and headers, never its
                    // body, so the router's envelope cannot be read here.
                    continuation.finish(
                        throwing: AgentsError.http(
                            HTTPFailure(
                                status: refused.statusCode,
                                requestID: refused.value(forHTTPHeaderField: "X-Request-Id") ?? "",
                                body: Data())))
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

/// Tells whoever opened a socket when its upgrade was answered: taken, or the task ended first.
/// The task's own delegate, so it works whatever session the app hands over.
private final class Upgrade: NSObject, URLSessionWebSocketDelegate, Sendable {
    private struct State {
        var opened: Bool?
        var waiting: CheckedContinuation<Bool, Never>?
    }

    private let state = AllocatedUnfairLock(State())

    /// Whether the upgrade was taken, once it was answered.
    func answered() async -> Bool {
        await withCheckedContinuation { continuation in
            let opened = state.withLock { state -> Bool? in
                if state.opened == nil { state.waiting = continuation }
                return state.opened
            }
            if let opened { continuation.resume(returning: opened) }
        }
    }

    func urlSession(
        _ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?
    ) {
        answer(true)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        answer(false)
    }

    private func answer(_ opened: Bool) {
        let waiting = state.withLock { state -> CheckedContinuation<Bool, Never>? in
            guard state.opened == nil else { return nil }
            state.opened = opened
            defer { state.waiting = nil }
            return state.waiting
        }
        waiting?.resume(returning: opened)
    }
}
