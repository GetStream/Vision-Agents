import Foundation
import StreamCore

/// What chorus-asr's p_eot must reach for a turn to have ended, at a check from from_ms into the pause on.
final class EndOfTurnBar: Sendable, Codable, JSONEncodable {
    /// What p_eot, chorus-asr's probability that the caller has finished, must reach for the turn to have ended: strictly between 0 and 1.
    let bar: Double
    /// How far into the pause, in milliseconds, the bar begins.
    let fromMs: Int

    init(bar: Double, fromMs: Int) {
        self.bar = bar
        self.fromMs = fromMs
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.bar = try container.decode(Double.self, forKey: .bar)
        self.fromMs = try container.decode(Int.self, forKey: .fromMs)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(bar, forKey: .bar)
        try container.encode(fromMs, forKey: .fromMs)
    }

}

extension EndOfTurnBar: Hashable {
    static func == (lhs: EndOfTurnBar, rhs: EndOfTurnBar) -> Bool {
        lhs.bar == rhs.bar &&
        lhs.fromMs == rhs.fromMs
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(bar)
        hasher.combine(fromMs)
    }
}
