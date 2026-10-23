import Foundation
import StreamCore

/// What the agent says as it joins, before anyone speaks.
final class GreetingPayload: Sendable, Codable, JSONEncodable {
    /// exact says the text word for word. variation has the model say its own variation of it on every call, so callers do not hear the same opening each time. A speech-to-speech model cannot say exact words, so it always says its own rendering of the text.
    let mode: GreetingMode?
    /// What the agent says on joining. Empty means the agent waits to be spoken to.
    let text: String

    init(mode: GreetingMode? = nil, text: String) {
        self.mode = mode
        self.text = text
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.mode = try container.decodeIfPresent(GreetingMode.self, forKey: .mode)
        self.text = try container.decode(String.self, forKey: .text)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(mode, forKey: .mode)
        try container.encode(text, forKey: .text)
    }

}

extension GreetingPayload: Hashable {
    static func == (lhs: GreetingPayload, rhs: GreetingPayload) -> Bool {
        lhs.mode == rhs.mode &&
        lhs.text == rhs.text
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(mode)
        hasher.combine(text)
    }
}
