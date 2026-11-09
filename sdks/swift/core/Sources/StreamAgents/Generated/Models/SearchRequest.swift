import Foundation
import StreamCore

final class SearchRequest: Sendable, Codable, JSONEncodable {
    /// A stored router config to take the options from. Anything named here as well overrides that one field of it.
    let configId: String?
    /// How this config finds out today's answers.
    let options: SearchOptionsPayload?
    /// The question, in the caller's own words.
    let query: String
    let tags: [String: String]?

    init(configId: String? = nil, options: SearchOptionsPayload? = nil, query: String, tags: [String: String]? = nil) {
        self.configId = configId
        self.options = options
        self.query = query
        self.tags = tags
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.configId = try container.decodeIfPresent(String.self, forKey: .configId)
        self.options = try container.decodeIfPresent(SearchOptionsPayload.self, forKey: .options)
        self.query = try container.decode(String.self, forKey: .query)
        self.tags = try container.decodeIfPresent([String: String].self, forKey: .tags)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(configId, forKey: .configId)
        try container.encodeIfPresent(options, forKey: .options)
        try container.encode(query, forKey: .query)
        try container.encodeIfPresent(tags, forKey: .tags)
    }

}

extension SearchRequest: Hashable {
    static func == (lhs: SearchRequest, rhs: SearchRequest) -> Bool {
        lhs.configId == rhs.configId &&
        lhs.options == rhs.options &&
        lhs.query == rhs.query &&
        lhs.tags == rhs.tags
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(configId)
        hasher.combine(options)
        hasher.combine(query)
        hasher.combine(tags)
    }
}
