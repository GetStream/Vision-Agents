import Foundation
import StreamCore

/// user is what a person said, assistant what the agent answered. These are the only turns a resumed conversation hands the model; instructions say anything a system message would.
final class HistoryRole: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let assistant = HistoryRole(rawValue: "assistant")
    static let user = HistoryRole(rawValue: "user")
}