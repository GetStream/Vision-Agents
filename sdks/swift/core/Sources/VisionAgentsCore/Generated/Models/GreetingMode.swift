import Foundation
import StreamCore

/// exact says the text word for word. variation has the model say its own variation of it on every call, so callers do not hear the same opening each time. A speech-to-speech model cannot say exact words, so it always says its own rendering of the text.
final class GreetingMode: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let exact = GreetingMode(rawValue: "exact")
    static let variation = GreetingMode(rawValue: "variation")
}