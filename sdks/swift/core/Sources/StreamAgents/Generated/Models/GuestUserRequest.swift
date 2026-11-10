import Foundation
import StreamCore

final class GuestUserRequest: Sendable, Codable, JSONEncodable {
    let custom: [String: RawJSON]?
    /// A guest coming back, to be given a fresh token as the same guest. Only honoured for a caller that proves it is that guest: the app's backend, the guest's own still-valid token, or the refresh_secret they were last handed. Any other caller is minted a new guest, as if no id had been sent, because a guest id is no secret: it is the author of their messages. Omitted mints a new guest.
    let id: String?
    /// What to call a new guest, for a transcript a person reads later. A guest coming back keeps their name unless the app's backend renames them.
    let name: String?
    /// The refresh_secret this guest was last handed, sent with their id once their token has expired. Spent by being used. One that is wrong or already spent mints a new guest instead.
    let refreshSecret: String?

    init(custom: [String: RawJSON]? = nil, id: String? = nil, name: String? = nil, refreshSecret: String? = nil) {
        self.custom = custom
        self.id = id
        self.name = name
        self.refreshSecret = refreshSecret
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.custom = try container.decodeIfPresent([String: RawJSON].self, forKey: .custom)
        self.id = try container.decodeIfPresent(String.self, forKey: .id)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)
        self.refreshSecret = try container.decodeIfPresent(String.self, forKey: .refreshSecret)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encodeIfPresent(id, forKey: .id)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encodeIfPresent(refreshSecret, forKey: .refreshSecret)
    }

}

extension GuestUserRequest: Hashable {
    static func == (lhs: GuestUserRequest, rhs: GuestUserRequest) -> Bool {
        lhs.custom == rhs.custom &&
        lhs.id == rhs.id &&
        lhs.name == rhs.name &&
        lhs.refreshSecret == rhs.refreshSecret
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(custom)
        hasher.combine(id)
        hasher.combine(name)
        hasher.combine(refreshSecret)
    }
}
