import Foundation
import StreamCore

/// A model that hears the caller's audio and rules an agent's floor. chorus-asr is the one there is.
final class FloorControlModel: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let chorusAsr = FloorControlModel(rawValue: "chorus-asr")
}