import Foundation

/// How to find out today's answers.
///
/// Everything is optional because everything has an answer already: a named router config
/// decides what this does not say, and the router decides what the config does not.
///
/// A provider that cannot express a term refuses the request rather than dropping it, so
/// asking for something is either honoured or reported.
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
