import Foundation
import StreamCore

final class HistoryMessage: Sendable, Codable, JSONEncodable {
    /// When it was said. The model is shown it beside a person's message, so it can tell an hour ago from just now.
    let createdAt: Date?
    /// Who said it, when several people share the thread. The model is shown it as a label, never as who is asking now.
    let name: String?
    /// user is what a person said, assistant what the agent answered. These are the only turns a resumed conversation hands the model; instructions say anything a system message would.
    let role: HistoryRole
    let text: String

    init(createdAt: Date? = nil, name: String? = nil, role: HistoryRole, text: String) {
        self.createdAt = createdAt
        self.name = name
        self.role = role
        self.text = text
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)
        self.role = try container.decode(HistoryRole.self, forKey: .role)
        self.text = try container.decode(String.self, forKey: .text)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(createdAt, forKey: .createdAt)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encode(role, forKey: .role)
        try container.encode(text, forKey: .text)
    }

}

extension HistoryMessage: Hashable {
    static func == (lhs: HistoryMessage, rhs: HistoryMessage) -> Bool {
        lhs.createdAt == rhs.createdAt &&
        lhs.name == rhs.name &&
        lhs.role == rhs.role &&
        lhs.text == rhs.text
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(createdAt)
        hasher.combine(name)
        hasher.combine(role)
        hasher.combine(text)
    }
}
