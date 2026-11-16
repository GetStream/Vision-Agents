import Foundation
import StreamCore

final class AudioSource: Sendable, Codable, JSONEncodable {
    /// Public HTTP(S) URL or base64 audio data URI, such as data:audio/mp4;base64,.... At most 8 MiB, whether fetched or sent as a data URI. The router fetches a URL itself, and refuses one that resolves to a private or loopback address.
    let url: String

    init(url: String) {
        self.url = url
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.url = try container.decode(String.self, forKey: .url)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encode(url, forKey: .url)
    }

}

extension AudioSource: Hashable {
    static func == (lhs: AudioSource, rhs: AudioSource) -> Bool {
        lhs.url == rhs.url
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(url)
    }
}
