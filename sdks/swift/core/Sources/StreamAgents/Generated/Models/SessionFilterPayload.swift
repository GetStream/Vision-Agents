import Foundation
import StreamCore

/// Which sessions to list. The keys of one object are ANDed. $and and $or each take 1 to 10 filters and nest up to 3 deep; a whole filter holds at most 30 field conditions. An empty filter lists every session the caller may see. Any other key, and any other operator, is refused with a 400 that names it and lists the fields that can be filtered.
final class SessionFilterPayload: Sendable, Codable, JSONEncodable {
    /// Every one of these filters holds.
    let and: [SessionFilterPayload]?
    /// A window of time: $gte is its start, which it holds, and $lt its end, which it does not. Either may be left out, and $gte is before $lt.
    let createdAt: TimeWindow?
    /// Labels the session's custom object holds, every one of them: {"origin": "test"} matches a session whose custom has origin set to the string "test".
    let custom: [String: String]?
    /// At least one of these filters holds. It chooses among the sessions the caller may see and never adds to them. text and user_id are refused inside it.
    let or: [SessionFilterPayload]?
    /// A full-text match, Stream's $q.
    let text: TextMatch?

    init(and: [SessionFilterPayload]? = nil, createdAt: TimeWindow? = nil, custom: [String: String]? = nil, or: [SessionFilterPayload]? = nil, text: TextMatch? = nil) {
        self.and = and
        self.createdAt = createdAt
        self.custom = custom
        self.or = or
        self.text = text
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.and = try container.decodeIfPresent([SessionFilterPayload].self, forKey: .and)
        self.createdAt = try container.decodeIfPresent(TimeWindow.self, forKey: .createdAt)
        self.custom = try container.decodeIfPresent([String: String].self, forKey: .custom)
        self.or = try container.decodeIfPresent([SessionFilterPayload].self, forKey: .or)
        self.text = try container.decodeIfPresent(TextMatch.self, forKey: .text)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(and, forKey: .and)
        try container.encodeIfPresent(createdAt, forKey: .createdAt)
        try container.encodeIfPresent(custom, forKey: .custom)
        try container.encodeIfPresent(or, forKey: .or)
        try container.encodeIfPresent(text, forKey: .text)
    }

}

extension SessionFilterPayload: Hashable {
    static func == (lhs: SessionFilterPayload, rhs: SessionFilterPayload) -> Bool {
        lhs.and == rhs.and &&
        lhs.createdAt == rhs.createdAt &&
        lhs.custom == rhs.custom &&
        lhs.or == rhs.or &&
        lhs.text == rhs.text
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(and)
        hasher.combine(createdAt)
        hasher.combine(custom)
        hasher.combine(or)
        hasher.combine(text)
    }
}
