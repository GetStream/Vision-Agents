import Foundation
import StreamCore

final class TextMatch: Sendable, Codable, JSONEncodable {
    /// Quoted phrases and bare words both work, and punctuation is taken rather than refused.
    let q: String

    init(q: String) {
        self.q = q
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.q = try container.decode(String.self, forKey: .q)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(q, forKey: .q)
    }

}

extension TextMatch: Hashable {
    static func == (lhs: TextMatch, rhs: TextMatch) -> Bool {
        lhs.q == rhs.q
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(q)
    }
}
