import Foundation
import StreamCore

/// A file sent with a question. The router reads it as text once, and the agent answers from it on this turn and every turn of the conversation after it.
final class FileSource: Sendable, Codable, JSONEncodable {
    /// What the file is called, shown on the person's message and to the model. Left out, the last part of the URL's path, or document for a data URI.
    let name: String?
    /// Public HTTP(S) URL or base64 data URI, such as data:application/pdf;base64,.... A PDF or a Word document (DOCX or DOC), told apart by its bytes. At most 20 MiB fetched, or 8 MiB as a data URI. The router fetches a URL itself, and refuses one that resolves to a private or loopback address.
    let url: String

    init(name: String? = nil, url: String) {
        self.name = name
        self.url = url
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)
        self.url = try container.decode(String.self, forKey: .url)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encode(url, forKey: .url)
    }

}

extension FileSource: Hashable {
    static func == (lhs: FileSource, rhs: FileSource) -> Bool {
        lhs.name == rhs.name &&
        lhs.url == rhs.url
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(name)
        hasher.combine(url)
    }
}
