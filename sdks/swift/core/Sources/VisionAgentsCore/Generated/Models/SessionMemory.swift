import Foundation
import StreamCore

/// Who the session's memories are about. Without a user id nothing is recalled or stored, which is the case for a call with nobody identified on it.
final class SessionMemory: Sendable, Codable, JSONEncodable {
    /// Separates two deployments sharing one memory account.
    let appId: String?
    /// The caller's own labels, which narrow recall further. They cannot widen it: a filter is applied alongside the user id, never instead of it.
    let filter: [String: String]?
    /// Who the memories belong to. Empty means the customer.
    let userId: String?

    init(appId: String? = nil, filter: [String: String]? = nil, userId: String? = nil) {
        self.appId = appId
        self.filter = filter
        self.userId = userId
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.appId = try container.decodeIfPresent(String.self, forKey: .appId)
        self.filter = try container.decodeIfPresent([String: String].self, forKey: .filter)
        self.userId = try container.decodeIfPresent(String.self, forKey: .userId)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(appId, forKey: .appId)
        try container.encodeIfPresent(filter, forKey: .filter)
        try container.encodeIfPresent(userId, forKey: .userId)
    }

}

extension SessionMemory: Hashable {
    static func == (lhs: SessionMemory, rhs: SessionMemory) -> Bool {
        lhs.appId == rhs.appId &&
        lhs.filter == rhs.filter &&
        lhs.userId == rhs.userId
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(appId)
        hasher.combine(filter)
        hasher.combine(userId)
    }
}
