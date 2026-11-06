import Foundation
import StreamCore

/// Whether a model that hears the caller's audio rules the agent's floor: when a caller has finished, and what to do about speech over the agent. An agent opts in for itself, and one that never chose is off. shadow and act need the router's chorus-asr server, and a write asking for either on a router without one is refused. Off, the flow controller rules the floor as it does with no such model, and end_of_turn does nothing; chorus-asr still transcribes for an agent whose stt names stream/chorus-asr. Each session runs by its own, so agents on one router may choose differently.
final class FloorControl: Sendable, Codable, JSONEncodable {
    /// What the agent does with the model's rulings. off asks it nothing. shadow asks it about each turn beside the flow controller and acts on nothing it says, to compare the two on real calls. act asks it first and acts on its ruling, and the flow controller rules a turn it does not answer in time.
    let mode: FloorControlMode
    /// A model that hears the caller's audio and rules an agent's floor. chorus-asr is the one there is.
    let model: FloorControlModel

    init(mode: FloorControlMode, model: FloorControlModel) {
        self.mode = mode
        self.model = model
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.mode = try container.decode(FloorControlMode.self, forKey: .mode)
        self.model = try container.decode(FloorControlModel.self, forKey: .model)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(mode, forKey: .mode)
        try container.encode(model, forKey: .model)
    }

}

extension FloorControl: Hashable {
    static func == (lhs: FloorControl, rhs: FloorControl) -> Bool {
        lhs.mode == rhs.mode &&
        lhs.model == rhs.model
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(mode)
        hasher.combine(model)
    }
}
