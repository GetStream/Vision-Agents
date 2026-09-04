import Foundation
import OpenAPIRuntime

/// How often a recording job is asked whether it is done. Transcription runs faster than real
/// time, so asking every second costs nothing next to the work itself.
private let pollInterval = Duration.seconds(1)

/// Everything the router routes, configured once.
///
/// Each of the three streaming modalities has a `realtime()` session and a `recording()` job,
/// and search has neither, because a question and its answer are one round trip. Everything
/// the named config holds is a default that a per-call option overrides.
///
///     let router = Router(url: url, customerID: "acme", config: "healthcare")
///
///     var wanted = TranscriptionOptions()
///     wanted.diarize = true
///     let transcript = try await router.stt.recording(.url(recording), options: wanted)
///
/// Configuring what a router config is belongs to a backend: the router refuses the writes
/// from a device, so this reads configs and routes through them.
public struct Router: Sendable {
    public let backend: Backend

    /// A stored router config, by the name it was stored under or by its id. Without one,
    /// every call says what it wants for itself.
    public let config: String

    /// Cost labels carried onto everything routed here, on top of the config's own.
    public var tags: [String: String]

    public init(
        url: URL,
        customerID: String,
        config: String = "",
        tags: [String: String] = [:],
        urlSession: URLSession = .shared
    ) {
        backend = Backend(url: url, customerID: customerID, urlSession: urlSession)
        self.config = config
        self.tags = tags
    }

    public init(backend: Backend, config: String = "", tags: [String: String] = [:]) {
        self.backend = backend
        self.config = config
        self.tags = tags
    }

    /// Transcription, live or from a recording.
    public var stt: Transcribing { Transcribing(router: self) }

    /// A voice, live or recorded.
    public var tts: Speaking { Speaking(router: self) }

    /// The model that answers.
    public var llm: Answering { Answering(router: self) }

    /// Answers one question out of what is true now.
    public func search(
        _ query: String,
        options: SearchOptions = SearchOptions()
    ) async throws -> SearchAnswer {
        let body = Components.Schemas.SearchRequest(
            configId: named(),
            query: query,
            options: options.schema,
            tags: labels().map { .init(additionalProperties: $0) })

        let output = try await call { try await $0.search(body: .json(body)) }
        switch output {
        case .ok(let response):
            return SearchAnswer(try response.body.json)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// The router configs this customer holds, newest first.
    public func configs() async throws -> [RouterConfig] {
        let output = try await call { try await $0.listRouterConfigs(.init()) }
        switch output {
        case .ok(let response):
            return try response.body.json.map(RouterConfig.init)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// Opens one modality socket and sends the start frame that says what it is for.
    ///
    /// Routing has to be told where to go, so a socket with neither a config nor a target is
    /// refused here rather than opened and closed again by the router.
    func open(modality: String, target: String?) throws -> ModalitySocket {
        guard !config.isEmpty || !(target ?? "").isEmpty else {
            throw AgentsError.http(
                status: 400,
                message: "routing needs a target, either in the options or in a config")
        }
        return ModalitySocket(
            url: backend.socketURL(path: "/v1/\(modality)/stream"),
            headers: backend.headers,
            urlSession: backend.urlSession)
    }

    /// The start frame a modality socket opens with.
    func startFrame(modality: String, block: [String: JSONValue]) -> [String: JSONValue] {
        var frame: [String: JSONValue] = ["type": .string("start")]
        if !config.isEmpty { frame["config_id"] = .string(config) }
        if !tags.isEmpty {
            frame["tags"] = .object(tags.mapValues { .string($0) })
        }
        frame[modality] = .object(block)
        return frame
    }

    /// The config a request is made under, when there is one.
    func named() -> String? { config.isEmpty ? nil : config }

    /// The cost labels a request carries, when there are any.
    func labels() -> [String: String]? { tags.isEmpty ? nil : tags }

    /// Runs one request, reporting a transport failure as one and leaving cancellation alone.
    func call<T>(_ body: (Client) async throws -> T) async throws -> T {
        do {
            return try await body(backend.client())
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AgentsError {
            throw error
        } catch let error as ClientError {
            if error.underlyingError is CancellationError { throw CancellationError() }
            throw AgentsError.transport(error.underlyingError)
        } catch {
            throw AgentsError.transport(error)
        }
    }
}

/// Transcription, live or from a recording.
public struct Transcribing: Sendable {
    let router: Router

    /// Opens a transcription socket, configured and ready for audio.
    public func realtime(_ options: TranscriptionOptions = TranscriptionOptions()) throws
        -> Transcriber
    {
        let socket = try router.open(modality: "stt", target: options.target)
        return Transcriber(
            socket: socket,
            start: router.startFrame(modality: "stt", block: options.frame))
    }

    /// Transcribes a whole recording and returns the transcript.
    ///
    /// This is the non-realtime form, served by the batch half of a vendor rather than the
    /// streaming one, which is both cheaper and more accurate. It waits for the job unless a
    /// `callback` is given, in which case it returns as soon as the job is accepted and the
    /// router calls back instead.
    public func recording(
        _ source: Recording,
        options: TranscriptionOptions = TranscriptionOptions(),
        callback: String? = nil
    ) async throws -> Transcription {
        let body = Components.Schemas.TranscriptionRequest(
            configId: router.named(),
            source: source.schema,
            options: options.schema,
            callback: callback,
            tags: router.labels().map { .init(additionalProperties: $0) })

        let output = try await router.call {
            try await $0.transcribeRecording(body: .json(body))
        }
        var job: Transcription
        switch output {
        case .accepted(let response):
            job = Transcription(try response.body.json)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }

        if callback != nil { return job }
        while job.status == .queued || job.status == .running {
            try await Task.sleep(for: pollInterval)
            job = try await transcription(job.id)
        }
        if job.status == .failed {
            throw AgentsError.http(
                status: 502, message: job.error.isEmpty ? "the recording failed" : job.error)
        }
        return job
    }

    /// A transcription job, however far it has got.
    public func transcription(_ id: String) async throws -> Transcription {
        let output = try await router.call {
            try await $0.getTranscription(path: .init(id: id))
        }
        switch output {
        case .ok(let response):
            return Transcription(try response.body.json)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }
}

/// A voice, live or recorded.
public struct Speaking: Sendable {
    let router: Router

    /// Opens a speech socket, configured and ready for text.
    public func realtime(_ options: VoiceOptions = VoiceOptions()) throws -> Voice {
        let socket = try router.open(modality: "tts", target: options.target)
        return Voice(
            socket: socket,
            start: router.startFrame(modality: "tts", block: options.frame))
    }

    /// Speaks a whole text into one file.
    ///
    /// Nothing is listening to an audiobook while it is being made, so this asks for the file
    /// rather than the stream, which is what lets a codec and a bitrate be chosen.
    public func recording(
        _ text: String,
        options: VoiceOptions = VoiceOptions(),
        callback: String? = nil
    ) async throws -> Speech {
        let body = Components.Schemas.SpeechRequest(
            configId: router.named(),
            text: text,
            options: options.schema,
            callback: callback,
            tags: router.labels().map { .init(additionalProperties: $0) })

        let output = try await router.call { try await $0.recordSpeech(body: .json(body)) }
        var job: Speech
        switch output {
        case .accepted(let response):
            job = Speech(try response.body.json)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }

        if callback != nil { return job }
        while job.status == .queued || job.status == .running {
            try await Task.sleep(for: pollInterval)
            job = try await speech(job.id)
        }
        if job.status == .failed {
            throw AgentsError.http(
                status: 502, message: job.error.isEmpty ? "the speech failed" : job.error)
        }
        return job
    }

    /// A speech job, however far it has got.
    public func speech(_ id: String) async throws -> Speech {
        let output = try await router.call { try await $0.getSpeech(path: .init(id: id)) }
        switch output {
        case .ok(let response):
            return Speech(try response.body.json)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }
}

/// The model that answers.
///
/// There is no `recording()` here. A completion is already whole by the time it is returned,
/// and what the socket buys is the answer arriving as it is written.
public struct Answering: Sendable {
    let router: Router

    /// Opens a completions socket, configured and ready for a question.
    public func realtime(_ options: ModelOptions = ModelOptions()) throws -> Model {
        let socket = try router.open(modality: "llm", target: options.target)
        return Model(
            socket: socket,
            start: router.startFrame(modality: "llm", block: options.frame))
    }
}

/// One open transcription socket.
public actor Transcriber {
    private let socket: ModalitySocket
    private let start: [String: JSONValue]

    init(socket: ModalitySocket, start: [String: JSONValue]) {
        self.socket = socket
        self.start = start
    }

    /// Opens the socket and yields what was heard until it closes.
    public func transcripts() async throws -> AsyncThrowingStream<Transcript, any Error> {
        let messages = await socket.open()
        try await socket.send(start)

        return AsyncThrowingStream { continuation in
            let pump = Task {
                do {
                    for try await message in messages {
                        guard case .frame(let frame) = message else { continue }
                        switch frame.kind {
                        case .transcript:
                            continuation.yield(Transcript(frame))
                        case .error:
                            continuation.finish(
                                throwing: AgentsError.http(
                                    status: 502, message: frame.errorText))
                            return
                        case .closed:
                            continuation.finish()
                            return
                        default:
                            continue
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in pump.cancel() }
        }
    }

    /// Hands over 16 kHz mono signed 16-bit PCM to be transcribed.
    public func send(_ pcm: Data) async throws {
        try await socket.send(audio: pcm)
    }

    /// Closes the socket. Safe to call more than once.
    public func close() async {
        await socket.close()
    }
}

/// One thing the transcriber heard.
public struct Transcript: Sendable, Hashable {
    public let text: String
    /// Whether this is the final version of what was heard, rather than a partial one.
    public let isFinal: Bool
    /// Who said it, when diarization was asked for.
    public let speaker: String
    public let language: String
    /// The frame as it arrived, for anything this SDK does not name.
    public let frame: RoutedFrame

    init(_ frame: RoutedFrame) {
        text = frame.text
        isFinal = frame.isFinal
        speaker = frame.speaker
        language = frame.language
        self.frame = frame
    }
}

/// One open speech socket.
public actor Voice {
    private let socket: ModalitySocket
    private let start: [String: JSONValue]

    init(socket: ModalitySocket, start: [String: JSONValue]) {
        self.socket = socket
        self.start = start
    }

    /// Opens the socket and yields speech until it closes.
    ///
    /// One utterance at a time: the audio frames come back bare, so two overlapping ones
    /// would be indistinguishable.
    public func audio() async throws -> AsyncThrowingStream<SpokenAudio, any Error> {
        let messages = await socket.open()
        try await socket.send(start)

        return AsyncThrowingStream { continuation in
            let pump = Task {
                do {
                    for try await message in messages {
                        switch message {
                        case .audio(let audio):
                            continuation.yield(audio)
                        case .frame(let frame):
                            if frame.kind == .error {
                                continuation.finish(
                                    throwing: AgentsError.http(
                                        status: 502, message: frame.errorText))
                                return
                            }
                            if frame.kind == .closed {
                                continuation.finish()
                                return
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in pump.cancel() }
        }
    }

    /// Says this, whose audio arrives on `audio()`.
    public func speak(_ text: String) async throws {
        try await socket.send([
            "type": .string("speak"), "text": .string(text), "final": .bool(true),
        ])
    }

    /// Abandons what is being spoken.
    public func interrupt() async throws {
        try await socket.send(["type": .string("interrupt")])
    }

    /// Closes the socket. Safe to call more than once.
    public func close() async {
        await socket.close()
    }
}

/// What to answer and how.
///
/// The names are the response parameters the router already speaks rather than a second
/// vocabulary for the same things. What the config holds fills in whatever is left here.
public struct Question: Sendable, Hashable {
    /// What the model answers under.
    public var instructions: String?
    /// The conversation so far, oldest first.
    public var messages: [Said] = []
    /// `auto`, `none`, `required`, or the name of a tool it must call.
    public var toolChoice: String?
    public var maxOutputTokens: Int?
    public var temperature: Double?
    public var reasoningEffort: ModelOptions.ReasoningEffort?
    public var format: ModelOptions.AnswerFormat?
    public var verbosity: ModelOptions.Verbosity?
    /// Continues from an earlier answer the provider still holds.
    public var previousResponseID: String?

    public init(_ text: String = "") {
        if !text.isEmpty { messages = [Said(role: "user", content: text)] }
    }

    /// This question as the frame that asks it.
    var frame: [String: JSONValue] {
        var fields: [String: JSONValue] = [
            "type": .string("respond"),
            "messages": .array(
                messages.map {
                    .object(["role": .string($0.role), "content": .string($0.content)])
                }),
        ]
        if let instructions { fields["instructions"] = .string(instructions) }
        if let toolChoice { fields["tool_choice"] = .string(toolChoice) }
        if let maxOutputTokens {
            fields["max_output_tokens"] = .number(Double(maxOutputTokens))
        }
        if let temperature { fields["temperature"] = .number(temperature) }
        if let reasoningEffort {
            fields["reasoning_effort"] = .string(reasoningEffort.rawValue)
        }
        if let format { fields["format"] = .string(format.rawValue) }
        if let verbosity { fields["verbosity"] = .string(verbosity.rawValue) }
        if let previousResponseID {
            fields["previous_response_id"] = .string(previousResponseID)
        }
        return fields
    }
}

/// One turn of the conversation.
public struct Said: Sendable, Hashable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// The reply, arriving as it is written.
public struct Answer: Sendable, Hashable {
    /// The next piece of text, on a delta.
    public let delta: String
    /// The whole answer, on the frame that finishes it.
    public let text: String
    /// Whether this is the frame that finished the answer.
    public let isComplete: Bool
    /// The frame as it arrived, for anything this SDK does not name.
    public let frame: RoutedFrame

    init(_ frame: RoutedFrame) {
        delta = frame.kind == .delta ? frame.text : ""
        text = frame.kind == .complete ? frame.text : ""
        isComplete = frame.kind == .complete
        self.frame = frame
    }
}

/// One open completions socket.
public actor Model {
    private let socket: ModalitySocket
    private let start: [String: JSONValue]

    init(socket: ModalitySocket, start: [String: JSONValue]) {
        self.socket = socket
        self.start = start
    }

    /// Opens the socket and yields the reply as it is written.
    public func answers() async throws -> AsyncThrowingStream<Answer, any Error> {
        let messages = await socket.open()
        try await socket.send(start)

        return AsyncThrowingStream { continuation in
            let pump = Task {
                do {
                    for try await message in messages {
                        guard case .frame(let frame) = message else { continue }
                        switch frame.kind {
                        case .delta, .complete:
                            continuation.yield(Answer(frame))
                        case .error:
                            continuation.finish(
                                throwing: AgentsError.http(
                                    status: 502, message: frame.errorText))
                            return
                        case .closed:
                            continuation.finish()
                            return
                        default:
                            continue
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in pump.cancel() }
        }
    }

    /// Asks something, whose answer arrives on `answers()`.
    public func ask(_ question: Question) async throws {
        try await socket.send(question.frame)
    }

    /// Asks something in the caller's own words.
    public func ask(_ text: String) async throws {
        try await socket.send(Question(text).frame)
    }

    /// Abandons the answer in flight.
    public func interrupt() async throws {
        try await socket.send(["type": .string("interrupt")])
    }

    /// Closes the socket. Safe to call more than once.
    public func close() async {
        await socket.close()
    }
}
