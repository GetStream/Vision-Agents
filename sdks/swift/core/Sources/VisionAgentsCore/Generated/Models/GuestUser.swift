import Foundation
import StreamCore

final class GuestUser: Sendable, Codable, JSONEncodable {
    let custom: [String: RawJSON]?
    /// When the token stops working. A guest coming back after it asks for another.
    let expiresAt: Date?
    let id: String
    let name: String?
    /// A Stream user token for this guest, which is what the chat and video SDKs connect with. It carries role guest, so an app that has turned guests off refuses it.
    let token: String

    init(custom: [String: RawJSON]? = nil, expiresAt: Date? = nil, id: String, name: String? = nil, token: String) {
        self.custom = custom
        self.expiresAt = expiresAt
        self.id = id
        self.name = name
        self.token = token
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.custom = try container.decodeIfPresent([String: RawJSON].self, forKey: .custom)
        self.expiresAt = try container.decodeIfPresent(Date.self, forKey: .expiresAt)
        self.id = try container.decode(String.self, forKey: .id)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)
        self.token = try container.decode(String.self, forKey: .token)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encodeIfPresent(expiresAt, forKey: .expiresAt)
        try container.encode(id, forKey: .id)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encode(token, forKey: .token)
    }

}

extension GuestUser: Hashable {
    static func == (lhs: GuestUser, rhs: GuestUser) -> Bool {
        lhs.custom == rhs.custom &&
        lhs.expiresAt == rhs.expiresAt &&
        lhs.id == rhs.id &&
        lhs.name == rhs.name &&
        lhs.token == rhs.token
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(custom)
        hasher.combine(expiresAt)
        hasher.combine(id)
        hasher.combine(name)
        hasher.combine(token)
    }
}
