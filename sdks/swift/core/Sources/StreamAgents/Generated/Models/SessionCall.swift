import Foundation
import StreamCore

/// An existing Stream call for the agent to join, by its type and id.
final class SessionCall: Sendable, Codable, JSONEncodable {
    /// The call id.
    let id: String
    /// The call type, such as default.
    let type: String

    init(id: String, type: String) {
        self.id = id
        self.type = type
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.id = try container.decode(String.self, forKey: .id)
        self.type = try container.decode(String.self, forKey: .type)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(id, forKey: .id)
        try container.encode(type, forKey: .type)
    }

}

extension SessionCall: Hashable {
    static func == (lhs: SessionCall, rhs: SessionCall) -> Bool {
        lhs.id == rhs.id &&
        lhs.type == rhs.type
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(type)
    }
}
