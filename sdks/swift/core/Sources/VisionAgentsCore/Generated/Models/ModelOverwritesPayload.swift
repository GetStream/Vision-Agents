import Foundation
import StreamCore

final class ModelOverwritesPayloadThinking: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let high = ModelOverwritesPayloadThinking(rawValue: "high")
    static let low = ModelOverwritesPayloadThinking(rawValue: "low")
    static let medium = ModelOverwritesPayloadThinking(rawValue: "medium")
    static let minimal = ModelOverwritesPayloadThinking(rawValue: "minimal")
    static let none = ModelOverwritesPayloadThinking(rawValue: "none")
}

final class ModelOverwritesPayloadVerbosity: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let high = ModelOverwritesPayloadVerbosity(rawValue: "high")
    static let low = ModelOverwritesPayloadVerbosity(rawValue: "low")
    static let medium = ModelOverwritesPayloadVerbosity(rawValue: "medium")
}

/// What to change about the models for one session, over whatever its agent config decided.
/// It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides the config, and a caller reading a session back wants to see what they changed in one place rather than diffed against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because a caller able to rewrite those could make a session impersonate a different agent. A backend sets one session's instructions with the instructions of CreateSessionRequest instead, which a device may not send.
final class ModelOverwritesPayload: Sendable, Codable, JSONEncodable {
    /// A provider/model or a capability shortcut, in place of the config's.
    let llm: String?
    /// Caps the reply, reasoning included. Omitted leaves the provider's default.
    let maxOutputTokens: Int?
    let search: String?
    /// A speech-to-speech target. Naming one here makes the session native even if the config did not, which means no transcriber, model or voice is opened.
    let sts: String?
    let stt: String?
    /// How random the answer is. Omitted leaves the provider's own default, which is not the same as zero: zero is a real request for a deterministic model.
    let temperature: Double?
    /// How hard to reason before answering. It becomes the reasoning effort on the request, which is the vocabulary the providers that support one already speak, and means nothing to a model that does not reason.
    let thinking: ModelOverwritesPayloadThinking?
    let tts: String?
    /// How much detail to give. Dropped for models that do not take it.
    let verbosity: ModelOverwritesPayloadVerbosity?

    init(llm: String? = nil, maxOutputTokens: Int? = nil, search: String? = nil, sts: String? = nil, stt: String? = nil, temperature: Double? = nil, thinking: ModelOverwritesPayloadThinking? = nil, tts: String? = nil, verbosity: ModelOverwritesPayloadVerbosity? = nil) {
        self.llm = llm
        self.maxOutputTokens = maxOutputTokens
        self.search = search
        self.sts = sts
        self.stt = stt
        self.temperature = temperature
        self.thinking = thinking
        self.tts = tts
        self.verbosity = verbosity
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.llm = try container.decodeIfPresent(String.self, forKey: .llm)
        self.maxOutputTokens = try container.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        self.search = try container.decodeIfPresent(String.self, forKey: .search)
        self.sts = try container.decodeIfPresent(String.self, forKey: .sts)
        self.stt = try container.decodeIfPresent(String.self, forKey: .stt)
        self.temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        self.thinking = try container.decodeIfPresent(ModelOverwritesPayloadThinking.self, forKey: .thinking)
        self.tts = try container.decodeIfPresent(String.self, forKey: .tts)
        self.verbosity = try container.decodeIfPresent(ModelOverwritesPayloadVerbosity.self, forKey: .verbosity)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(llm, forKey: .llm)
        try container.encodeIfPresent(maxOutputTokens, forKey: .maxOutputTokens)
        try container.encodeIfPresent(search, forKey: .search)
        try container.encodeIfPresent(sts, forKey: .sts)
        try container.encodeIfPresent(stt, forKey: .stt)
        try container.encodeIfPresent(temperature, forKey: .temperature)
        try container.encodeIfPresent(thinking, forKey: .thinking)
        try container.encodeIfPresent(tts, forKey: .tts)
        try container.encodeIfPresent(verbosity, forKey: .verbosity)
    }

}

extension ModelOverwritesPayload: Hashable {
    static func == (lhs: ModelOverwritesPayload, rhs: ModelOverwritesPayload) -> Bool {
        lhs.llm == rhs.llm &&
        lhs.maxOutputTokens == rhs.maxOutputTokens &&
        lhs.search == rhs.search &&
        lhs.sts == rhs.sts &&
        lhs.stt == rhs.stt &&
        lhs.temperature == rhs.temperature &&
        lhs.thinking == rhs.thinking &&
        lhs.tts == rhs.tts &&
        lhs.verbosity == rhs.verbosity
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(llm)
        hasher.combine(maxOutputTokens)
        hasher.combine(search)
        hasher.combine(sts)
        hasher.combine(stt)
        hasher.combine(temperature)
        hasher.combine(thinking)
        hasher.combine(tts)
        hasher.combine(verbosity)
    }
}
