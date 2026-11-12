import Foundation
import StreamCore

final class RewoundTo: Sendable, Codable, JSONEncodable {
    /// The message itself is hidden too.
    let inclusive: Bool
    /// The message the channel was rewound to. Every message after it is hidden.
    let messageId: String

    init(inclusive: Bool, messageId: String) {
        self.inclusive = inclusive
        self.messageId = messageId
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.inclusive = try container.decode(Bool.self, forKey: .inclusive)
        self.messageId = try container.decode(String.self, forKey: .messageId)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(inclusive, forKey: .inclusive)
        try container.encode(messageId, forKey: .messageId)
    }

}

extension RewoundTo: Hashable {
    static func == (lhs: RewoundTo, rhs: RewoundTo) -> Bool {
        lhs.inclusive == rhs.inclusive &&
        lhs.messageId == rhs.messageId
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(inclusive)
        hasher.combine(messageId)
    }
}
