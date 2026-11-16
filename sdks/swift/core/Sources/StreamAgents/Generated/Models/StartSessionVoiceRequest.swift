import Foundation
import StreamCore

final class StartSessionVoiceRequest: Sendable, Codable, JSONEncodable {
    /// An existing Stream call for the agent to join, by its type and id.
    let call: SessionCall?

    init(call: SessionCall? = nil) {
        self.call = call
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.call = try container.decodeIfPresent(SessionCall.self, forKey: .call)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(call, forKey: .call)
    }

}

extension StartSessionVoiceRequest: Hashable {
    static func == (lhs: StartSessionVoiceRequest, rhs: StartSessionVoiceRequest) -> Bool {
        lhs.call == rhs.call
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(call)
    }
}
