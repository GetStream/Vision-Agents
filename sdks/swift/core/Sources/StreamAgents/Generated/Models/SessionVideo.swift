import Foundation
import StreamCore

final class SessionVideo: Sendable, Codable, JSONEncodable {
    /// Number of recent frames captured for a visual task. Default one.
    let maxFrames: Int?
    /// Track or processor source. Omitted requires one unambiguous available source.
    let source: String?

    init(maxFrames: Int? = nil, source: String? = nil) {
        self.maxFrames = maxFrames
        self.source = source
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.maxFrames = try container.decodeIfPresent(Int.self, forKey: .maxFrames)
        self.source = try container.decodeIfPresent(String.self, forKey: .source)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(maxFrames, forKey: .maxFrames)
        try container.encodeIfPresent(source, forKey: .source)
    }

}

extension SessionVideo: Hashable {
    static func == (lhs: SessionVideo, rhs: SessionVideo) -> Bool {
        lhs.maxFrames == rhs.maxFrames &&
        lhs.source == rhs.source
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(maxFrames)
        hasher.combine(source)
    }
}
