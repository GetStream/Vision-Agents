import Foundation
import StreamCore

final class GuestUserRequest: Sendable, Codable, JSONEncodable {
    let custom: [String: RawJSON]?
    /// A guest id to reuse, for somebody coming back. Omitted mints a new one. Asking for an id that is already a guest of this customer returns that guest with a fresh token rather than failing, because coming back is the same person.
    let id: String?
    /// What to call them, for a transcript a person reads later.
    let name: String?

    init(custom: [String: RawJSON]? = nil, id: String? = nil, name: String? = nil) {
        self.custom = custom
        self.id = id
        self.name = name
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.custom = try container.decodeIfPresent([String: RawJSON].self, forKey: .custom)
        self.id = try container.decodeIfPresent(String.self, forKey: .id)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encodeIfPresent(id, forKey: .id)
        try container.encodeIfPresent(name, forKey: .name)
    }

}

extension GuestUserRequest: Hashable {
    static func == (lhs: GuestUserRequest, rhs: GuestUserRequest) -> Bool {
        lhs.custom == rhs.custom &&
        lhs.id == rhs.id &&
        lhs.name == rhs.name
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(custom)
        hasher.combine(id)
        hasher.combine(name)
    }
}
