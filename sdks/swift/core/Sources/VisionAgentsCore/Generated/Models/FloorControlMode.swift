import Foundation
import StreamCore

/// What the agent does with the model's rulings. off asks it nothing. shadow asks it about each turn beside the flow controller and acts on nothing it says, to compare the two on real calls. act asks it first and acts on its ruling, and the flow controller rules a turn it does not answer in time.
final class FloorControlMode: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let act = FloorControlMode(rawValue: "act")
    static let off = FloorControlMode(rawValue: "off")
    static let shadow = FloorControlMode(rawValue: "shadow")
}