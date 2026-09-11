import Foundation

/// How big one frame may be. A transcript carries a sentence and an audio frame a fraction of
/// a second of speech, so a megabyte is generous.
private let maximumRoutedFrameSize = 1 << 20

/// One frame on a modality socket.
///
/// The fields are kept as they arrived so that a frame added to the router after this SDK
/// shipped still reaches the caller, with `kind` nil and `type` naming it.
public struct RoutedFrame: Sendable, Hashable {
    /// The router's own name for this frame, always present.
    public let type: String
    /// The frame's fields, as the router sends them.
    public let fields: [String: JSONValue]

    /// This frame as one of the kinds the SDK knows, or nil for one it does not.
    public var kind: Kind? { Kind(rawValue: type) }

    public init(type: String, fields: [String: JSONValue] = [:]) {
        self.type = type
        self.fields = fields
    }

    /// The frames the modality sockets document. Anything else is still delivered.
    public enum Kind: String, Sendable, CaseIterable {
        case started
        case transcript
        case speechStarted = "speech_started"
        case speechStopped = "speech_stopped"
        case synthesisComplete = "synthesis_complete"
        case delta
        case complete
        case interrupted
        case inputTranscript = "input_transcript"
        case outputTranscript = "output_transcript"
        case responseStarted = "response_started"
        case responseComplete = "response_complete"
        case toolCall = "tool_call"
        case toolCancel = "tool_cancel"
        case sessionExpiring = "session_expiring"
        case error
        case closed
    }

    public subscript(key: String) -> JSONValue { fields[key] ?? .null }

    /// What was heard, spoken or answered, depending on the frame.
    public var text: String { self["text"].stringValue }
    /// Whether a transcript is the final version of what was heard.
    public var isFinal: Bool { self["final"].boolValue }
    /// Who said it, when diarization was asked for.
    public var speaker: String { self["speaker"].stringValue }
    public var language: String { self["language"].stringValue }
    /// What went wrong, on an error frame.
    public var errorText: String { self["error"].stringValue }
}

extension RoutedFrame: Decodable {
    public init(from decoder: Decoder) throws {
        let fields = try decoder.singleValueContainer().decode([String: JSONValue].self)
        guard case .string(let type)? = fields["type"] else {
            throw DecodingError.dataCorruptedError(
                in: try decoder.singleValueContainer(),
                debugDescription: "a modality frame needs a type")
        }
        self.init(type: type, fields: fields.filter { $0.key != "type" })
    }
}

/// One piece of speech, as the provider produced it.
public struct SpokenAudio: Sendable, Hashable {
    /// Signed 16-bit little-endian PCM.
    public let samples: Data
    public let sampleRate: Int
    public let channels: Int
    /// Which reply the audio belongs to and where in it this chunk falls, on a conversation
    /// socket. A client drops audio whose generation a `response_complete` frame has already
    /// reported interrupted. Nil on a voice socket, whose header does not say.
    public let generation: Int?
    public let index: Int?

    /// The size of the header on a voice socket's audio frame, and on a conversation's.
    static let voiceHeader = 8
    static let conversationHeader = 16

    /// Reads one audio frame, whose header says how to play what follows.
    init?(_ payload: Data, header: Int = SpokenAudio.voiceHeader) {
        guard payload.count > header else { return nil }
        let bytes = [UInt8](payload.prefix(header))
        sampleRate = Int(bytes[0]) | Int(bytes[1]) << 8 | Int(bytes[2]) << 16 | Int(bytes[3]) << 24
        channels = Int(bytes[4]) | Int(bytes[5]) << 8
        if header == SpokenAudio.conversationHeader {
            generation =
                Int(bytes[8]) | Int(bytes[9]) << 8 | Int(bytes[10]) << 16 | Int(bytes[11]) << 24
            index = Int(bytes[12]) | Int(bytes[13]) << 8 | Int(bytes[14]) << 16 | Int(bytes[15]) << 24
        } else {
            generation = nil
            index = nil
        }
        samples = payload.dropFirst(header)
    }
}

/// What arrives on a modality socket: a frame, or speech.
public enum RoutedMessage: Sendable, Hashable {
    case frame(RoutedFrame)
    case audio(SpokenAudio)
}

/// The socket carrying one routed modality.
///
/// Hand-written for the same reason `SessionSocket` is: OpenAPI stops at the upgrade. It is an
/// actor so the read loop and the sends cannot race over the task, and so closing twice is
/// harmless.
actor ModalitySocket {
    private let url: URL
    private let headers: [String: String]
    private let urlSession: URLSession
    /// How long the header on an audio frame is, which depends on the modality: a voice
    /// says only how to play the audio, a conversation also says which reply it belongs to.
    private let audioHeader: Int
    private var task: URLSessionWebSocketTask?
    private var reader: Task<Void, Never>?

    init(
        url: URL,
        headers: [String: String],
        urlSession: URLSession = .shared,
        audioHeader: Int = SpokenAudio.voiceHeader
    ) {
        self.url = url
        self.headers = headers
        self.urlSession = urlSession
        self.audioHeader = audioHeader
    }

    /// Opens the socket and starts reading.
    ///
    /// The stream is returned rather than exposed as a property so it cannot be attached to
    /// twice, and so there is no window between opening and attaching in which a frame could
    /// be dropped.
    func open() -> AsyncThrowingStream<RoutedMessage, any Error> {
        var request = URLRequest(url: url)
        for (name, value) in headers {
            request.setValue(value, forHTTPHeaderField: name)
        }

        let task = urlSession.webSocketTask(with: request)
        task.maximumMessageSize = maximumRoutedFrameSize
        self.task = task
        task.resume()

        // Unbounded because dropping a frame would lose the end of an utterance or of an
        // answer. The producer is a network socket, so the queue is bounded in practice by
        // what the router sends.
        let (stream, continuation) = AsyncThrowingStream.makeStream(
            of: RoutedMessage.self, bufferingPolicy: .unbounded)

        reader = Task { [weak self] in
            await self?.read(task: task, into: continuation)
        }

        return stream
    }

    /// Sends one JSON frame.
    func send(_ frame: [String: JSONValue]) async throws {
        guard let task else {
            throw AgentsError.socketClosed(code: 0, reason: "the socket is not open")
        }
        let encoded = try JSONEncoder().encode(JSONValue.object(frame))
        do {
            try await task.send(.data(encoded))
        } catch {
            throw AgentsError.transport(error)
        }
    }

    /// Sends audio to be transcribed.
    func send(audio: Data) async throws {
        guard let task else {
            throw AgentsError.socketClosed(code: 0, reason: "the socket is not open")
        }
        do {
            try await task.send(.data(audio))
        } catch {
            throw AgentsError.transport(error)
        }
    }

    /// Closes the socket. Safe to call more than once.
    func close() {
        reader?.cancel()
        reader = nil
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
    }

    private func read(
        task: URLSessionWebSocketTask,
        into continuation: AsyncThrowingStream<RoutedMessage, any Error>.Continuation
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

            switch message {
            case .data(let payload):
                // Speech arrives as a binary frame and everything else as JSON, so what a
                // payload is is decided by whether it reads as a frame.
                if let frame = try? decoder.decode(RoutedFrame.self, from: payload) {
                    continuation.yield(.frame(frame))
                } else if let audio = SpokenAudio(payload, header: audioHeader) {
                    continuation.yield(.audio(audio))
                }
            case .string(let text):
                if let frame = try? decoder.decode(RoutedFrame.self, from: Data(text.utf8)) {
                    continuation.yield(.frame(frame))
                }
            @unknown default:
                continue
            }
        }
        continuation.finish()
    }
}
