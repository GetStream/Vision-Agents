import Foundation
import StreamCore

/// How this config finds out today's answers.
final class SearchOptionsPayload: Sendable, Codable, JSONEncodable {
    /// The kind of source to prefer - news, papers, company, github - for the providers that classify their index.
    let category: String?
    /// What to return alongside each hit.
    let contents: [String]?
    /// How much work a search is worth. instant answers from the index in a few hundred milliseconds; deep crawls and reasons over what it finds and can take tens of seconds. Providers offer different ladders, so each one maps these four onto its own.
    let depth: SearchDepth?
    let excludeDomains: [String]?
    /// Only answer from these domains.
    let includeDomains: [String]?
    /// Country or region to answer from, for queries whose answer depends on where.
    let location: String?
    /// How stale a cached page may be. Zero forces a live crawl, which is slower and costs more.
    let maxAgeHours: Int?
    /// A JSON schema the answer must fit, for the providers that can be asked to structure what they found.
    let outputSchema: [String: RawJSON]?
    /// A priority list of where to try, in the order given, which wins over target and depth when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it stands. A search that fails is asked of the next entry that will have it.
    let providers: [String]?
    /// How many hits to return.
    let results: Int?
    /// A provider/model or a capability shortcut.
    let target: String?

    init(category: String? = nil, contents: [String]? = nil, depth: SearchDepth? = nil, excludeDomains: [String]? = nil, includeDomains: [String]? = nil, location: String? = nil, maxAgeHours: Int? = nil, outputSchema: [String: RawJSON]? = nil, providers: [String]? = nil, results: Int? = nil, target: String? = nil) {
        self.category = category
        self.contents = contents
        self.depth = depth
        self.excludeDomains = excludeDomains
        self.includeDomains = includeDomains
        self.location = location
        self.maxAgeHours = maxAgeHours
        self.outputSchema = outputSchema
        self.providers = providers
        self.results = results
        self.target = target
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.category = try container.decodeIfPresent(String.self, forKey: .category)
        self.contents = try container.decodeIfPresent([String].self, forKey: .contents)
        self.depth = try container.decodeIfPresent(SearchDepth.self, forKey: .depth)
        self.excludeDomains = try container.decodeIfPresent([String].self, forKey: .excludeDomains)
        self.includeDomains = try container.decodeIfPresent([String].self, forKey: .includeDomains)
        self.location = try container.decodeIfPresent(String.self, forKey: .location)
        self.maxAgeHours = try container.decodeIfPresent(Int.self, forKey: .maxAgeHours)
        self.outputSchema = try container.decodeIfPresent([String: RawJSON].self, forKey: .outputSchema)
        self.providers = try container.decodeIfPresent([String].self, forKey: .providers)
        self.results = try container.decodeIfPresent(Int.self, forKey: .results)
        self.target = try container.decodeIfPresent(String.self, forKey: .target)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(category, forKey: .category)
        try container.encodeIfPresent(contents, forKey: .contents)
        try container.encodeIfPresent(depth, forKey: .depth)
        try container.encodeIfPresent(excludeDomains, forKey: .excludeDomains)
        try container.encodeIfPresent(includeDomains, forKey: .includeDomains)
        try container.encodeIfPresent(location, forKey: .location)
        try container.encodeIfPresent(maxAgeHours, forKey: .maxAgeHours)
        try container.encodeIfPresent(outputSchema, forKey: .outputSchema)
        try container.encodeIfPresent(providers, forKey: .providers)
        try container.encodeIfPresent(results, forKey: .results)
        try container.encodeIfPresent(target, forKey: .target)
    }

}

extension SearchOptionsPayload: Hashable {
    static func == (lhs: SearchOptionsPayload, rhs: SearchOptionsPayload) -> Bool {
        lhs.category == rhs.category &&
        lhs.contents == rhs.contents &&
        lhs.depth == rhs.depth &&
        lhs.excludeDomains == rhs.excludeDomains &&
        lhs.includeDomains == rhs.includeDomains &&
        lhs.location == rhs.location &&
        lhs.maxAgeHours == rhs.maxAgeHours &&
        lhs.outputSchema == rhs.outputSchema &&
        lhs.providers == rhs.providers &&
        lhs.results == rhs.results &&
        lhs.target == rhs.target
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(category)
        hasher.combine(contents)
        hasher.combine(depth)
        hasher.combine(excludeDomains)
        hasher.combine(includeDomains)
        hasher.combine(location)
        hasher.combine(maxAgeHours)
        hasher.combine(outputSchema)
        hasher.combine(providers)
        hasher.combine(results)
        hasher.combine(target)
    }
}
