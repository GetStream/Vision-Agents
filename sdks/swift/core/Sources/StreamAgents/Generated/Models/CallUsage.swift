import Foundation
import StreamCore

/// What the call spent, summed over every request it made. Counted once the call is over, so it is absent while one is still running. Requests that failed are included: a model that read the prompt and then fell over is still billed for it.
final class CallUsage: Sendable, Codable, JSONEncodable {
    /// The part of those prompts a provider served from its own cache.
    let cachedInputTokens: Int
    /// Millionths of a dollar, priced from the providers' configured rates.
    let costMicros: Int
    /// Every prompt the models read, the cached part included.
    let inputTokens: Int
    /// Everything the models generated, reasoning included.
    let outputTokens: Int
    /// How many calls to a model it took, transcription and speech included.
    let requests: Int

    init(cachedInputTokens: Int, costMicros: Int, inputTokens: Int, outputTokens: Int, requests: Int) {
        self.cachedInputTokens = cachedInputTokens
        self.costMicros = costMicros
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.requests = requests
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.cachedInputTokens = try container.decode(Int.self, forKey: .cachedInputTokens)
        self.costMicros = try container.decode(Int.self, forKey: .costMicros)
        self.inputTokens = try container.decode(Int.self, forKey: .inputTokens)
        self.outputTokens = try container.decode(Int.self, forKey: .outputTokens)
        self.requests = try container.decode(Int.self, forKey: .requests)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(cachedInputTokens, forKey: .cachedInputTokens)
        try container.encode(costMicros, forKey: .costMicros)
        try container.encode(inputTokens, forKey: .inputTokens)
        try container.encode(outputTokens, forKey: .outputTokens)
        try container.encode(requests, forKey: .requests)
    }

}

extension CallUsage: Hashable {
    static func == (lhs: CallUsage, rhs: CallUsage) -> Bool {
        lhs.cachedInputTokens == rhs.cachedInputTokens &&
        lhs.costMicros == rhs.costMicros &&
        lhs.inputTokens == rhs.inputTokens &&
        lhs.outputTokens == rhs.outputTokens &&
        lhs.requests == rhs.requests
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(cachedInputTokens)
        hasher.combine(costMicros)
        hasher.combine(inputTokens)
        hasher.combine(outputTokens)
        hasher.combine(requests)
    }
}
