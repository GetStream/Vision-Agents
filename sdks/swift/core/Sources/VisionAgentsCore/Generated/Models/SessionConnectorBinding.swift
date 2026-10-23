import Foundation
import StreamCore

/// The connection a session uses for one of its agent config's connector bindings chosen per session. Only a reference: the credential stays sealed on the connection.
final class SessionConnectorBinding: Sendable, Codable, JSONEncodable {
    /// The caller's own connection to the binding's connector.
    let connectionId: String
    /// The binding's alias in the agent config.
    let name: String

    init(connectionId: String, name: String) {
        self.connectionId = connectionId
        self.name = name
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.connectionId = try container.decode(String.self, forKey: .connectionId)
        self.name = try container.decode(String.self, forKey: .name)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(connectionId, forKey: .connectionId)
        try container.encode(name, forKey: .name)
    }

}

extension SessionConnectorBinding: Hashable {
    static func == (lhs: SessionConnectorBinding, rhs: SessionConnectorBinding) -> Bool {
        lhs.connectionId == rhs.connectionId &&
        lhs.name == rhs.name
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(connectionId)
        hasher.combine(name)
    }
}
