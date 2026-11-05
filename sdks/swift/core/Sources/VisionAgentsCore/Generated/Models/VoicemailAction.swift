import Foundation
import StreamCore

/// hang_up: end the call at once, saying nothing.
/// leave_message: say message once the greeting has ended or once the call counts as a machine, then hang up.
final class VoicemailAction: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let hangUp = VoicemailAction(rawValue: "hang_up")
    static let leaveMessage = VoicemailAction(rawValue: "leave_message")
}