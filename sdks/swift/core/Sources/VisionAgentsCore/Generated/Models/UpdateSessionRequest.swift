import Foundation
import StreamCore

final class UpdateSessionRequestThinking: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let high = UpdateSessionRequestThinking(rawValue: "high")
    static let low = UpdateSessionRequestThinking(rawValue: "low")
    static let medium = UpdateSessionRequestThinking(rawValue: "medium")
    static let minimal = UpdateSessionRequestThinking(rawValue: "minimal")
    static let none = UpdateSessionRequestThinking(rawValue: "none")
}

final class UpdateSessionRequestVerbosity: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let high = UpdateSessionRequestVerbosity(rawValue: "high")
    static let low = UpdateSessionRequestVerbosity(rawValue: "low")
    static let medium = UpdateSessionRequestVerbosity(rawValue: "medium")
}

/// What to change about one session. A field left out is left as it is. Title, description and custom can change on a session that ended, and are all an end user's device may change; everything else needs the session running and a server-side caller.
final class UpdateSessionRequest: Sendable, Codable, JSONEncodable {
    /// Replaces the caller's labels whole. An empty object clears them.
    let custom: [String: RawJSON]?
    let description: String?
    /// The conversation model, a provider/model or a capability shortcut.
    let llm: String?
    let maxOutputTokens: Int?
    /// A speech-to-speech target, which makes the session native. Empty makes it a cascade again.
    let sts: String?
    let stt: String?
    let temperature: Double?
    let thinking: UpdateSessionRequestThinking?
    let title: String?
    let tts: String?
    let verbosity: UpdateSessionRequestVerbosity?
    /// The voice to speak in: a provider's own voice id, or one of your voices by id or by the name you gave it. Prefix it with custom: to mean only the latter: without the prefix a name that is not one of yours is passed through to the provider's library, and with it a name that is not one of yours is refused. A native session (sts) takes the vendor's own voice name instead, such as marin at OpenAI. Empty returns to the provider's default.
    let voice: String?

    init(custom: [String: RawJSON]? = nil, description: String? = nil, llm: String? = nil, maxOutputTokens: Int? = nil, sts: String? = nil, stt: String? = nil, temperature: Double? = nil, thinking: UpdateSessionRequestThinking? = nil, title: String? = nil, tts: String? = nil, verbosity: UpdateSessionRequestVerbosity? = nil, voice: String? = nil) {
        self.custom = custom
        self.description = description
        self.llm = llm
        self.maxOutputTokens = maxOutputTokens
        self.sts = sts
        self.stt = stt
        self.temperature = temperature
        self.thinking = thinking
        self.title = title
        self.tts = tts
        self.verbosity = verbosity
        self.voice = voice
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.custom = try container.decodeIfPresent([String: RawJSON].self, forKey: .custom)
        self.description = try container.decodeIfPresent(String.self, forKey: .description)
        self.llm = try container.decodeIfPresent(String.self, forKey: .llm)
        self.maxOutputTokens = try container.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        self.sts = try container.decodeIfPresent(String.self, forKey: .sts)
        self.stt = try container.decodeIfPresent(String.self, forKey: .stt)
        self.temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        self.thinking = try container.decodeIfPresent(UpdateSessionRequestThinking.self, forKey: .thinking)
        self.title = try container.decodeIfPresent(String.self, forKey: .title)
        self.tts = try container.decodeIfPresent(String.self, forKey: .tts)
        self.verbosity = try container.decodeIfPresent(UpdateSessionRequestVerbosity.self, forKey: .verbosity)
        self.voice = try container.decodeIfPresent(String.self, forKey: .voice)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encodeIfPresent(description, forKey: .description)
        try container.encodeIfPresent(llm, forKey: .llm)
        try container.encodeIfPresent(maxOutputTokens, forKey: .maxOutputTokens)
        try container.encodeIfPresent(sts, forKey: .sts)
        try container.encodeIfPresent(stt, forKey: .stt)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(thinking, forKey: .thinking)
        try container.encodeIfPresent(title, forKey: .title)
        try container.encodeIfPresent(tts, forKey: .tts)
        try container.encodeIfPresent(verbosity, forKey: .verbosity)
        try container.encodeIfPresent(voice, forKey: .voice)
    }

}

extension UpdateSessionRequest: Hashable {
    static func == (lhs: UpdateSessionRequest, rhs: UpdateSessionRequest) -> Bool {
        lhs.custom == rhs.custom &&
        lhs.description == rhs.description &&
        lhs.llm == rhs.llm &&
        lhs.maxOutputTokens == rhs.maxOutputTokens &&
        lhs.sts == rhs.sts &&
        lhs.stt == rhs.stt &&
        lhs.temperature == rhs.temperature &&
        lhs.thinking == rhs.thinking &&
        lhs.title == rhs.title &&
        lhs.tts == rhs.tts &&
        lhs.verbosity == rhs.verbosity &&
        lhs.voice == rhs.voice
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(custom)
        hasher.combine(description)
        hasher.combine(llm)
        hasher.combine(maxOutputTokens)
        hasher.combine(sts)
        hasher.combine(stt)
        hasher.combine(temperature)
        hasher.combine(thinking)
        hasher.combine(title)
        hasher.combine(tts)
        hasher.combine(verbosity)
        hasher.combine(voice)
    }
}
