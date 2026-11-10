import Foundation
import StreamCore

final class ImageSourcePayloadDetail: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let auto = ImageSourcePayloadDetail(rawValue: "auto")
    static let high = ImageSourcePayloadDetail(rawValue: "high")
    static let low = ImageSourcePayloadDetail(rawValue: "low")
}

final class ImageSourcePayload: Sendable, Codable, JSONEncodable {
    let detail: ImageSourcePayloadDetail?
    /// Absolute HTTP(S) URL or base64 image data URI.
    let url: String

    init(detail: ImageSourcePayloadDetail? = nil, url: String) {
        self.detail = detail
        self.url = url
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.detail = try container.decodeIfPresent(ImageSourcePayloadDetail.self, forKey: .detail)
        self.url = try container.decode(String.self, forKey: .url)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(detail, forKey: .detail)
        try container.encode(url, forKey: .url)
    }

}

extension ImageSourcePayload: Hashable {
    static func == (lhs: ImageSourcePayload, rhs: ImageSourcePayload) -> Bool {
        lhs.detail == rhs.detail &&
        lhs.url == rhs.url
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(detail)
        hasher.combine(url)
    }
}
