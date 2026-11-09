import Foundation
import StreamCore

/// How much work a search is worth. instant answers from the index in a few hundred milliseconds; deep crawls and reasons over what it finds and can take tens of seconds. Providers offer different ladders, so each one maps these four onto its own.
final class SearchDepth: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let deep = SearchDepth(rawValue: "deep")
    static let fast = SearchDepth(rawValue: "fast")
    static let instant = SearchDepth(rawValue: "instant")
    static let standard = SearchDepth(rawValue: "standard")
}