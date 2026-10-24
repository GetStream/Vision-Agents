import Foundation
import StreamCore

final class VideoSource: Sendable, Codable, JSONEncodable {
    /// How many frames to sample, evenly spaced across the clip. Default 8.
    let maxFrames: Int?
    /// Public HTTPS URL or base64 video data URI, such as data:video/mp4;base64,.... At most 50 MB either way. The router fetches a URL itself, and refuses one that resolves to a private or loopback address.
    let url: String

    init(maxFrames: Int? = nil, url: String) {
        self.maxFrames = maxFrames
        self.url = url
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.maxFrames = try container.decodeIfPresent(Int.self, forKey: .maxFrames)
        self.url = try container.decode(String.self, forKey: .url)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(maxFrames, forKey: .maxFrames)
        try container.encode(url, forKey: .url)
    }

}

extension VideoSource: Hashable {
    static func == (lhs: VideoSource, rhs: VideoSource) -> Bool {
        lhs.maxFrames == rhs.maxFrames &&
        lhs.url == rhs.url
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(maxFrames)
        hasher.combine(url)
    }
}
