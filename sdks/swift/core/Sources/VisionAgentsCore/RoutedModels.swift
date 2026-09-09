import Foundation

/// How to transcribe, live or from a recording.
///
/// Everything is optional because everything has an answer already: a named router config
/// decides what this does not say, and the router decides what the config does not. A field
/// that only means something on one of the two forms says so.
///
/// A provider that cannot express a term refuses the request rather than dropping it, so
/// asking for something is either honoured or reported.
public struct TranscriptionOptions: Sendable, Hashable {
    /// A provider/model or a capability shortcut: `en-low-latency` live, `en-recorded` for a
    /// recording.
    public var target: String?
    /// ISO codes the candidates must cover.
    public var languages: [String] = []
    /// Let the provider work out the language instead of being told it.
    public var detectLanguage: Bool?
    /// Rate of the PCM sent on the socket. Live only.
    public var sampleRate: Int?
    /// Emit partial transcripts as they firm up. Live only.
    public var interim: Bool?
    /// What decides a turn is over. Live only.
    public var endpointing: Endpointing?
    /// How long a pause ends a turn, for silence endpointing. Live only.
    public var silenceMs: Int?
    /// Label each stretch of speech with who said it.
    public var diarize: Bool?
    /// A hard cap on the speakers diarization may find, not a hint.
    public var maxSpeakers: Int?
    /// Business-specific words the transcriber would otherwise get wrong.
    public var keyterms: [String] = []
    /// Punctuation, capitalisation and smart formatting.
    public var format: Bool?
    /// Remove personally identifying information.
    public var redact: Bool?
    /// Word-level timestamps. Recording only.
    public var words: Bool?
    /// What a finished transcript is rendered as. Recording only.
    public var output: TranscriptOutput?
    /// Summarise the recording, where the provider offers audio intelligence.
    public var summary: Bool?
    /// Extract named entities. Recording only.
    public var entities: Bool?

    public init() {}

    /// What decides a turn is over: a pause, or a model judging the sentence finished.
    public enum Endpointing: String, Sendable {
        case silence
        case semantic
    }

    /// What a finished transcript is rendered as.
    public enum TranscriptOutput: String, Sendable {
        case json
        case srt
        case vtt
    }
}

/// How to speak, live or into a file.
public struct VoiceOptions: Sendable, Hashable {
    /// A provider/model or a capability shortcut.
    public var target: String?
    /// A provider-specific voice id.
    public var voice: String?
    public var languages: [String] = []
    /// Rate of delivery, 1 being the voice's own.
    public var speed: Double?
    /// Loudness, 1 being the voice's own.
    public var volume: Double?
    /// Affect to speak with, for the providers that take one.
    public var emotion: String?
    /// Delivery style, for the providers that name styles rather than emotions.
    public var style: String?
    /// How much the voice may vary between chunks.
    public var stability: Double?
    /// How closely a cloned voice tracks its reference.
    public var similarity: Double?
    /// Codec, sample rate and bitrate as one name: `pcm_16000`, `mp3_44100_128`.
    public var format: String?

    public init() {}
}

/// How to answer. The names are the response parameters the router already speaks rather than
/// a second vocabulary for the same things.
public struct ModelOptions: Sendable, Hashable {
    /// A provider/model or a capability shortcut such as `llm-fast`.
    public var target: String?
    /// What the model answers under, when a request does not say.
    public var instructions: String?
    public var maxOutputTokens: Int?
    public var temperature: Double?
    /// How long the model may think, on the models that think.
    public var reasoningEffort: ReasoningEffort?
    /// Whether the answer is prose or a JSON object.
    public var format: AnswerFormat?
    public var verbosity: Verbosity?
    /// `auto`, `none`, `required`, or the name of a tool the model must call.
    public var toolChoice: String?

    public init() {}

    public enum ReasoningEffort: String, Sendable {
        case minimal, low, medium, high
    }

    public enum AnswerFormat: String, Sendable {
        case text
        case jsonObject = "json_object"
    }

    public enum Verbosity: String, Sendable {
        case low, medium, high
    }
}

/// How to find out today's answers.
public struct SearchOptions: Sendable, Hashable {
    /// A provider/model or a capability shortcut such as `search-fast`.
    public var target: String?
    /// How much work a search is worth: `instant` answers from the index in a few hundred
    /// milliseconds, `deep` reads what it finds and can take tens of seconds.
    public var depth: Depth?
    /// How many hits to return.
    public var results: Int?
    /// Only answer from these domains.
    public var includeDomains: [String] = []
    public var excludeDomains: [String] = []
    /// The kind of source to prefer, for the providers that classify their index.
    public var category: String?
    /// How stale a cached page may be. Zero forces a live crawl.
    public var maxAgeHours: Int?
    /// Country or region to answer from.
    public var location: String?

    public init() {}

    public enum Depth: String, Sendable {
        case instant, fast, standard, deep
    }
}

/// Where the audio to work on comes from.
///
/// A URL is what every vendor's batch API takes and what anything longer than a clip should
/// use; the bytes save a caller with a voice memo from hosting it somewhere first.
public enum Recording: Sendable, Hashable {
    case url(URL)
    case audio(Data)

    /// A local file, read into the bytes the router takes.
    public static func file(_ path: URL) throws -> Recording {
        .audio(try Data(contentsOf: path))
    }
}

extension Recording {
    var schema: Components.Schemas.RecordingSource {
        switch self {
        case .url(let url):
            return .init(url: url.absoluteString)
        case .audio(let data):
            return .init(audio: .init(data))
        }
    }
}

extension TranscriptionOptions {
    /// These options as the router's own option block.
    var schema: Components.Schemas.SttOptions {
        .init(
            target: target,
            languages: languages.isEmpty ? nil : languages,
            detectLanguage: detectLanguage,
            sampleRate: sampleRate,
            interim: interim,
            endpointing: endpointing.flatMap { .init(rawValue: $0.rawValue) },
            silenceMs: silenceMs,
            diarize: diarize,
            maxSpeakers: maxSpeakers,
            keyterms: keyterms.isEmpty ? nil : keyterms,
            format: format,
            redact: redact,
            words: words,
            output: output.flatMap { .init(rawValue: $0.rawValue) },
            summary: summary,
            entities: entities)
    }

    /// These options as the start frame carries them, which is the same block as JSON.
    var frame: [String: JSONValue] { block(schema) }
}

extension VoiceOptions {
    var schema: Components.Schemas.TtsOptions {
        .init(
            target: target,
            voice: voice,
            languages: languages.isEmpty ? nil : languages,
            speed: speed.map(Float.init),
            volume: volume.map(Float.init),
            emotion: emotion,
            style: style,
            stability: stability.map(Float.init),
            similarity: similarity.map(Float.init),
            format: format)
    }

    var frame: [String: JSONValue] { block(schema) }
}

extension ModelOptions {
    var schema: Components.Schemas.LlmOptions {
        .init(
            target: target,
            instructions: instructions,
            maxOutputTokens: maxOutputTokens,
            temperature: temperature.map(Float.init),
            reasoningEffort: reasoningEffort.flatMap { .init(rawValue: $0.rawValue) },
            format: format.flatMap { .init(rawValue: $0.rawValue) },
            verbosity: verbosity.flatMap { .init(rawValue: $0.rawValue) },
            toolChoice: toolChoice)
    }

    var frame: [String: JSONValue] { block(schema) }
}

extension SearchOptions {
    var schema: Components.Schemas.SearchOptions {
        .init(
            target: target,
            depth: depth.flatMap { .init(rawValue: $0.rawValue) },
            results: results,
            includeDomains: includeDomains.isEmpty ? nil : includeDomains,
            excludeDomains: excludeDomains.isEmpty ? nil : excludeDomains,
            category: category,
            maxAgeHours: maxAgeHours,
            location: location)
    }
}

/// One option block as the start frame carries it.
///
/// The block is encoded rather than written out field by field so that a field added to the
/// spec reaches the socket without being listed in a second place.
private func block(_ options: some Encodable) -> [String: JSONValue] {
    guard let encoded = try? JSONEncoder().encode(options),
        let decoded = try? JSONDecoder().decode(JSONValue.self, from: encoded),
        case .object(let fields) = decoded
    else { return [:] }
    return fields
}

/// Where a recording job has got to. A failed job carries the reason, and a completed one
/// carries its result.
public enum RecordingStatus: String, Sendable {
    case queued, running, completed, failed

    init(_ schema: Components.Schemas.RecordingStatus) {
        switch schema {
        case .queued: self = .queued
        case .running: self = .running
        case .completed: self = .completed
        case .failed: self = .failed
        }
    }
}

/// A whole recording transcribed.
public struct Transcription: Sendable, Hashable, Identifiable {
    public let id: String
    public let status: RecordingStatus
    public let provider: String
    public let model: String
    /// What was spoken, whether it was asked for or detected.
    public let language: String
    /// The whole transcript as prose, which is what most callers want.
    public let text: String
    /// The timings, when they were asked for.
    public let words: [TranscriptWord]
    /// The speakers diarization found, in the order they first spoke.
    public let speakers: [String]
    /// The transcript as an SRT or VTT file, when one of those was asked for.
    public let subtitles: String
    public let summary: String
    /// How long the recording was, which is what it was billed on.
    public let audioDurationMs: Int
    /// Why the job failed, if it did.
    public let error: String

    init(_ schema: Components.Schemas.Transcription) {
        id = schema.id
        status = RecordingStatus(schema.status)
        provider = schema.provider ?? ""
        model = schema.model ?? ""
        language = schema.language ?? ""
        text = schema.text ?? ""
        words = (schema.words ?? []).map(TranscriptWord.init)
        speakers = schema.speakers ?? []
        subtitles = schema.subtitles ?? ""
        summary = schema.summary ?? ""
        audioDurationMs = Int(schema.audioDurationMs ?? 0)
        error = schema.error ?? ""
    }
}

/// One word and when it was said.
public struct TranscriptWord: Sendable, Hashable {
    public let text: String
    public let startMs: Int
    public let endMs: Int
    public let confidence: Double
    /// Who said it, when diarization was asked for.
    public let speaker: String

    init(_ schema: Components.Schemas.TranscriptWord) {
        text = schema.text
        startMs = Int(schema.startMs)
        endMs = Int(schema.endMs)
        confidence = Double(schema.confidence ?? 0)
        speaker = schema.speaker ?? ""
    }
}

/// A whole text spoken into one file.
public struct Speech: Sendable, Hashable, Identifiable {
    public let id: String
    public let status: RecordingStatus
    public let provider: String
    public let model: String
    /// What the audio is encoded as, which is what was asked for.
    public let format: String
    /// The audio itself, when it did not come back behind a URL.
    public let audio: Data
    /// Where the finished audio is, on a deployment that stores it.
    public let url: URL?
    public let audioDurationMs: Int
    /// How much text was spoken, which is what it was billed on.
    public let characters: Int
    public let error: String

    init(_ schema: Components.Schemas.Speech) {
        id = schema.id
        status = RecordingStatus(schema.status)
        provider = schema.provider ?? ""
        model = schema.model ?? ""
        format = schema.format ?? ""
        audio = schema.audio.map { Data($0.data) } ?? Data()
        url = schema.url.flatMap(URL.init(string:))
        audioDurationMs = Int(schema.audioDurationMs ?? 0)
        characters = Int(schema.characters ?? 0)
        error = schema.error ?? ""
    }
}

/// What a search found.
public struct SearchAnswer: Sendable, Hashable {
    public let provider: String
    public let model: String
    /// The provider's own summary, where it offers one: a sentence to say rather than a page to
    /// read.
    public let answer: String
    /// The sources behind it, most relevant first.
    public let results: [SearchHit]

    init(_ schema: Components.Schemas.SearchAnswer) {
        provider = schema.provider
        model = schema.model
        answer = schema.answer ?? ""
        results = schema.results.map(SearchHit.init)
    }
}

/// One source a search found.
public struct SearchHit: Sendable, Hashable {
    public let title: String
    public let url: String
    /// The relevant extract, which is what a model reads.
    public let text: String
    public let score: Double

    init(_ schema: Components.Schemas.SearchResult) {
        title = schema.title ?? ""
        url = schema.url
        text = schema.text ?? ""
        score = Double(schema.score ?? 0)
    }
}

/// A stored set of routing options, which a `Router` can be named after.
public struct RouterConfig: Sendable, Hashable, Identifiable {
    public let id: String
    public let name: String
    public let tags: [String: String]
    public let updatedAt: Date

    init(_ schema: Components.Schemas.RouterConfig) {
        id = schema.id
        name = schema.name
        tags = schema.tags?.additionalProperties ?? [:]
        updatedAt = schema.updatedAt
    }
}
