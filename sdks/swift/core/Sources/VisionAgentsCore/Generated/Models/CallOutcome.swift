import Foundation
import StreamCore

/// How a placed call ended.
/// completed: the person answered and the call ended.
/// busy: the line was busy or the person declined.
/// no_answer: it rang until the ring timeout.
/// failed: the vendor could not place it, for a bad number or a carrier fault.
/// canceled: the vendor says the call was hung up before an answer.
/// unknown: the vendor reports nothing, as Sinch and Bird do, and the phone never joined.
final class CallOutcome: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let busy = CallOutcome(rawValue: "busy")
    static let canceled = CallOutcome(rawValue: "canceled")
    static let completed = CallOutcome(rawValue: "completed")
    static let failed = CallOutcome(rawValue: "failed")
    static let noAnswer = CallOutcome(rawValue: "no_answer")
    static let unknown = CallOutcome(rawValue: "unknown")
}