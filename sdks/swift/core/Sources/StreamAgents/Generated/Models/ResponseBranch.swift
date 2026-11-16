import Foundation
import StreamCore

final class ResponseBranchOf: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let question = ResponseBranchOf(rawValue: "question")
    static let reply = ResponseBranchOf(rawValue: "reply")
}

final class ResponseBranch: Sendable, Codable, JSONEncodable {
    /// How many there are.
    let count: Int
    /// Its place, from 1, oldest first.
    let index: Int
    /// reply counts the answers to one question, a regenerate's; question the questions asked at one point, an edit's.
    let of: ResponseBranchOf

    init(count: Int, index: Int, of: ResponseBranchOf) {
        self.count = count
        self.index = index
        self.of = of
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.count = try container.decode(Int.self, forKey: .count)
        self.index = try container.decode(Int.self, forKey: .index)
        self.of = try container.decode(ResponseBranchOf.self, forKey: .of)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(count, forKey: .count)
        try container.encode(index, forKey: .index)
        try container.encode(of, forKey: .of)
    }

}

extension ResponseBranch: Hashable {
    static func == (lhs: ResponseBranch, rhs: ResponseBranch) -> Bool {
        lhs.count == rhs.count &&
        lhs.index == rhs.index &&
        lhs.of == rhs.of
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(count)
        hasher.combine(index)
        hasher.combine(of)
    }
}
