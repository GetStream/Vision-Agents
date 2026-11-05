import Foundation
import StreamCore

/// When chorus-asr has a caller's turn ended, where the router asks it about turns: one of max_wait_ms, preset and bars. An empty object is the server's default preset. A preset or a wait chooses one of the server's measured points: its bars, and how many checks in a row of one pause must reach them. When a pause is checked, how much more a turn's first pause asks, and the pause that ends a turn whatever the model says stay the server's. What a preset or a wait means is the policy of the model the server runs, read when a pause is checked; one that model no longer publishes ends turns by its default.
final class EndOfTurn: Sendable, Codable, JSONEncodable {
    /// Bars of the agent's own: the first from 0 ms, each later one from further into the pause. The turn has ended at the first check of a pause where p_eot reaches the last bar to have begun. Unlike a measured point, nobody has measured what they do to callers.
    let bars: [EndOfTurnBar]?
    /// The most a caller who has finished may wait for the answer, in milliseconds. It chooses the point of chorus-asr's measured trade-off with the fewest cut-ins whose measured p90 wait is within it. Refused when it is shorter than every point's.
    let maxWaitMs: Int?
    /// One of the presets the chorus-asr server publishes, by name, such as safe or fast, each naming one of its measured points. Refused when the server publishes none by that name.
    let preset: String?

    init(bars: [EndOfTurnBar]? = nil, maxWaitMs: Int? = nil, preset: String? = nil) {
        self.bars = bars
        self.maxWaitMs = maxWaitMs
        self.preset = preset
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.bars = try container.decodeIfPresent([EndOfTurnBar].self, forKey: .bars)
        self.maxWaitMs = try container.decodeIfPresent(Int.self, forKey: .maxWaitMs)
        self.preset = try container.decodeIfPresent(String.self, forKey: .preset)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(bars, forKey: .bars)
        try container.encodeIfPresent(maxWaitMs, forKey: .maxWaitMs)
        try container.encodeIfPresent(preset, forKey: .preset)
    }

}

extension EndOfTurn: Hashable {
    static func == (lhs: EndOfTurn, rhs: EndOfTurn) -> Bool {
        lhs.bars == rhs.bars &&
        lhs.maxWaitMs == rhs.maxWaitMs &&
        lhs.preset == rhs.preset
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(bars)
        hasher.combine(maxWaitMs)
        hasher.combine(preset)
    }
}
