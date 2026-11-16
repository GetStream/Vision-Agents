import Foundation
import StreamCore

final class RewindSessionRequest: Sendable, Codable, JSONEncodable {
    /// The response to carry on from. It is kept; everything after it is not.
    let responseId: String

    init(responseId: String) {
        self.responseId = responseId
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.responseId = try container.decode(String.self, forKey: .responseId)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(responseId, forKey: .responseId)
    }

}

extension RewindSessionRequest: Hashable {
    static func == (lhs: RewindSessionRequest, rhs: RewindSessionRequest) -> Bool {
        lhs.responseId == rhs.responseId
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(responseId)
    }
}
