import Foundation
import StreamCore

final class CreateResponseRequest: Sendable, Codable, JSONEncodable {
    let images: [ImageSourcePayload]?
    /// Caps this answer, reasoning included, and the replies that finish it after its tools. Omitted keeps the session's.
    let maxOutputTokens: Int?
    /// How long the model may think before this answer, and the replies that finish it after its tools. One of the efforts the session's model accepts, such as none, minimal, low, medium, high or max; any other is a 400. Omitted keeps the session's.
    let reasoningEffort: String?
    /// Generated and sent by the SDKs, one per question, so a retry of the same question is answered once. Required for personal persistent text conversations, and ignored by a session not kept in Stream Chat. With one, images are data URIs, at most 8, uploaded to the conversation's channel and shown on the person's message, and videos are refused. A retry with the same id, text and images starts no second turn and returns no id.
    let requestId: String?
    /// What to answer, as though it had been said.
    let text: String
    /// Recorded clips to show the agent. The router samples evenly spaced frames from each and hands them to the vision skill with their timestamps, which is how every vision model is shown a video, since none of the ones routed here take one whole.
    let videos: [VideoSource]?

    init(images: [ImageSourcePayload]? = nil, maxOutputTokens: Int? = nil, reasoningEffort: String? = nil, requestId: String? = nil, text: String, videos: [VideoSource]? = nil) {
        self.images = images
        self.maxOutputTokens = maxOutputTokens
        self.reasoningEffort = reasoningEffort
        self.requestId = requestId
        self.text = text
        self.videos = videos
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.images = try container.decodeIfPresent([ImageSourcePayload].self, forKey: .images)
        self.maxOutputTokens = try container.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        self.reasoningEffort = try container.decodeIfPresent(String.self, forKey: .reasoningEffort)
        self.requestId = try container.decodeIfPresent(String.self, forKey: .requestId)
        self.text = try container.decode(String.self, forKey: .text)
        self.videos = try container.decodeIfPresent([VideoSource].self, forKey: .videos)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(images, forKey: .images)
        try container.encodeIfPresent(maxOutputTokens, forKey: .maxOutputTokens)
        try container.encodeIfPresent(reasoningEffort, forKey: .reasoningEffort)
        try container.encodeIfPresent(requestId, forKey: .requestId)
        try container.encode(text, forKey: .text)
        try container.encodeIfPresent(videos, forKey: .videos)
    }

}

extension CreateResponseRequest: Hashable {
    static func == (lhs: CreateResponseRequest, rhs: CreateResponseRequest) -> Bool {
        lhs.images == rhs.images &&
        lhs.maxOutputTokens == rhs.maxOutputTokens &&
        lhs.reasoningEffort == rhs.reasoningEffort &&
        lhs.requestId == rhs.requestId &&
        lhs.text == rhs.text &&
        lhs.videos == rhs.videos
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(images)
        hasher.combine(maxOutputTokens)
        hasher.combine(reasoningEffort)
        hasher.combine(requestId)
        hasher.combine(text)
        hasher.combine(videos)
    }
}
