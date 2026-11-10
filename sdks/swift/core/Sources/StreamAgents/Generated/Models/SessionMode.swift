import Foundation
import StreamCore

/// How the session hears and speaks: a transcriber, a conversation model and a voice; one speech-to-speech model; or in writing.
final class SessionMode: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let cascade = SessionMode(rawValue: "cascade")
    static let native = SessionMode(rawValue: "native")
    static let text = SessionMode(rawValue: "text")
}