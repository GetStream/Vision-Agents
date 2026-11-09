import Foundation
import StreamCore

/// Whose doing a session's end was: none, the agent or the app ended it; caller, the person on the call; router, the router itself; vendor, a model vendor; carrier, the phone carrier; customer_config, the app's configuration or its own key; customer_credit, the organization's credit or spend limits.
final class SessionEndFault: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let caller = SessionEndFault(rawValue: "caller")
    static let carrier = SessionEndFault(rawValue: "carrier")
    static let customerConfig = SessionEndFault(rawValue: "customer_config")
    static let customerCredit = SessionEndFault(rawValue: "customer_credit")
    static let none = SessionEndFault(rawValue: "none")
    static let router = SessionEndFault(rawValue: "router")
    static let vendor = SessionEndFault(rawValue: "vendor")
}