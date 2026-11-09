import Foundation
import StreamCore

/// What the agent does when an answering machine picks up.
final class SessionVoicemail: Sendable, Codable, JSONEncodable {
    /// hang_up: end the call at once, saying nothing.
    /// leave_message: say message once the greeting has ended or once the call counts as a machine, then hang up.
    let action: VoicemailAction
    /// Said word for word once the machine's greeting has ended, or once the call counts as a machine, then the agent hangs up. Required for leave_message, and not only spaces. Ignored for hang_up.
    let message: String?

    init(action: VoicemailAction, message: String? = nil) {
        self.action = action
        self.message = message
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.action = try container.decode(VoicemailAction.self, forKey: .action)
        self.message = try container.decodeIfPresent(String.self, forKey: .message)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(action, forKey: .action)
        try container.encodeIfPresent(message, forKey: .message)
    }

}

extension SessionVoicemail: Hashable {
    static func == (lhs: SessionVoicemail, rhs: SessionVoicemail) -> Bool {
        lhs.action == rhs.action &&
        lhs.message == rhs.message
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(action)
        hasher.combine(message)
    }
}
