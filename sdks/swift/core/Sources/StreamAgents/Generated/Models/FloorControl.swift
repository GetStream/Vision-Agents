import Foundation
import StreamCore

/// The model that controls an agent's floor, and how long a caller may wait once they stop. Left out or null, the stored choice stays; an empty object is no choice again, which the router's default rules: the flow controller, or stream/chorus-asr at the server's default point where the router defaults every agent to it; {model: flow-controller} keeps the flow controller whatever the router's default; anything else replaces the stored choice whole.
final class FloorControl: Sendable, Codable, JSONEncodable {
    /// The most a caller who has finished may wait for the answer, in milliseconds. It chooses the point of the model's measured trade-off with the fewest cut-ins whose measured p90 wait is within it, and is refused (400) when it is shorter than every point's. Left out, the server's default point. It is stream/chorus-asr's: with flow-controller it is refused (400).
    let maxWaitMs: Int?
    /// The model that controls the floor, hearing the caller's audio: when they have finished, and what to do about speech over the agent. stream/chorus-asr is the one model there is; flow-controller keeps the flow controller, which rules the floor where no model does, whatever the router defaults an agent that chose none to. Required, unless floor_control is empty. stream/chorus-asr on a router without a chorus-asr server is refused (400 not_configured).
    let model: String?

    init(maxWaitMs: Int? = nil, model: String? = nil) {
        self.maxWaitMs = maxWaitMs
        self.model = model
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.maxWaitMs = try container.decodeIfPresent(Int.self, forKey: .maxWaitMs)
        self.model = try container.decodeIfPresent(String.self, forKey: .model)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(maxWaitMs, forKey: .maxWaitMs)
        try container.encodeIfPresent(model, forKey: .model)
    }

}

extension FloorControl: Hashable {
    static func == (lhs: FloorControl, rhs: FloorControl) -> Bool {
        lhs.maxWaitMs == rhs.maxWaitMs &&
        lhs.model == rhs.model
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(maxWaitMs)
        hasher.combine(model)
    }
}
