import Foundation
import StreamCore

final class SearchResult: Sendable, Codable, JSONEncodable {
    /// How relevant the provider judged it.
    let score: Double?
    /// The relevant extract, which is what a model reads.
    let text: String?
    let title: String?
    let url: String

    init(score: Double? = nil, text: String? = nil, title: String? = nil, url: String) {
        self.score = score
        self.text = text
        self.title = title
        self.url = url
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.score = try container.decodeIfPresent(Double.self, forKey: .score)
        self.text = try container.decodeIfPresent(String.self, forKey: .text)
        self.title = try container.decodeIfPresent(String.self, forKey: .title)
        self.url = try container.decode(String.self, forKey: .url)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(score, forKey: .score)
        try container.encodeIfPresent(text, forKey: .text)
        try container.encodeIfPresent(title, forKey: .title)
        try container.encode(url, forKey: .url)
    }

}

extension SearchResult: Hashable {
    static func == (lhs: SearchResult, rhs: SearchResult) -> Bool {
        lhs.score == rhs.score &&
        lhs.text == rhs.text &&
        lhs.title == rhs.title &&
        lhs.url == rhs.url
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(score)
        hasher.combine(text)
        hasher.combine(title)
        hasher.combine(url)
    }
}
