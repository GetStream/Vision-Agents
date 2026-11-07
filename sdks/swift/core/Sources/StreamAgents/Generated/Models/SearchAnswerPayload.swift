import Foundation
import StreamCore

final class SearchAnswerPayload: Sendable, Codable, JSONEncodable {
    /// The provider's own summary, where it offers one. It is what a voice agent wants: a sentence to say rather than a page to read.
    let answer: String?
    let model: String
    let provider: String
    /// The sources behind it, most relevant first.
    let results: [SearchResult]

    init(answer: String? = nil, model: String, provider: String, results: [SearchResult]) {
        self.answer = answer
        self.model = model
        self.provider = provider
        self.results = results
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.answer = try container.decodeIfPresent(String.self, forKey: .answer)
        self.model = try container.decode(String.self, forKey: .model)
        self.provider = try container.decode(String.self, forKey: .provider)
        self.results = try container.decode([SearchResult].self, forKey: .results)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(answer, forKey: .answer)
        try container.encode(model, forKey: .model)
        try container.encode(provider, forKey: .provider)
        try container.encode(results, forKey: .results)
    }

}

extension SearchAnswerPayload: Hashable {
    static func == (lhs: SearchAnswerPayload, rhs: SearchAnswerPayload) -> Bool {
        lhs.answer == rhs.answer &&
        lhs.model == rhs.model &&
        lhs.provider == rhs.provider &&
        lhs.results == rhs.results
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(answer)
        hasher.combine(model)
        hasher.combine(provider)
        hasher.combine(results)
    }
}
