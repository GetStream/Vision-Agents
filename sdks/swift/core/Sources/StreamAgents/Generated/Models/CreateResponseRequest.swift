import Foundation
import StreamCore

final class CreateResponseRequest: Sendable, Codable, JSONEncodable {
    let audio: AudioSource?
    /// A file to answer questions about: a PDF or a Word document. The router reads it as text before the turn starts, and the agent answers from it on this turn and on every turn of the conversation after it, including a conversation carried on later or forked by the same person. In a conversation kept in Stream Chat the person's message shows the file. At most 100 pages, and at most 120,000 characters of text together with the files the conversation already holds, or a quarter of what the model reads at once. Only in a person's own conversation: refused in a thread, in a channel others write in, in a session a worker opened for a channel's message, and by a speech-to-speech session. Not with a voice note. Refused before any turn starts when it cannot be read, is too large, or reads as a prompt injection under the organization's policy.
    let files: [FileSource]?
    let images: [ImageSourcePayload]?
    /// Caps this answer, reasoning included, and the replies that finish it after its tools. Omitted keeps the session's.
    let maxOutputTokens: Int?
    /// How long the model may think before this answer, and the replies that finish it after its tools. One of the efforts the session's model accepts, such as none, minimal, low, medium, high or max; any other is a 400. Omitted keeps the session's.
    let reasoningEffort: String?
    /// An earlier response of a conversation kept in Stream Chat to regenerate or edit. With no text its question is answered again; with text, the text is asked in its place. Requires request_id, and takes no images, audio, videos, files or ui_action.
    let replaces: String?
    /// Generated and sent by the SDKs, one per question, so a retry of the same question is answered once. Required for personal persistent text conversations, and ignored by a session not kept in Stream Chat. With one, images are data URIs, at most 8, uploaded to the conversation's channel and shown on the person's message, a voice note is uploaded and shown there too, and the frames of videos are shown to the model alone. A retry with the same id, text, images and voice note starts no second turn, returns no id and does not fetch its videos again: videos are not part of what a retry is matched on. On a session a dispatch worker opened for a message handed out from a channel, that message is already there: nothing is written or uploaded for it, and a reused id is answered as a duplicate even with other text or images.
    let requestId: String?
    /// What to answer, as though it had been said. May be empty when images or audio carry the question. Images with no text are shown on the person's message in a session kept in Stream Chat, so they are then data URIs, at most 8, with or without a request_id.
    let text: String
    /// What the person did in the generated UI of an earlier reply, such as tapping a place or submitting a form, which the text says in their words: an object naming the reply (message_id), the UI's format and the action (name), with what it carries (payload), of at most 8 KB as JSON. Their message carries it as custom.ai_ui_action, and the model is told it after the text.
    let uiAction: [String: RawJSON]?
    /// Recorded clips to show the agent. The router samples evenly spaced frames from each and hands them to the vision skill with their timestamps, which is how every vision model is shown a video, since none of the ones routed here take one whole.
    let videos: [VideoSource]?

    init(audio: AudioSource? = nil, files: [FileSource]? = nil, images: [ImageSourcePayload]? = nil, maxOutputTokens: Int? = nil, reasoningEffort: String? = nil, replaces: String? = nil, requestId: String? = nil, text: String, uiAction: [String: RawJSON]? = nil, videos: [VideoSource]? = nil) {
        self.audio = audio
        self.files = files
        self.images = images
        self.maxOutputTokens = maxOutputTokens
        self.reasoningEffort = reasoningEffort
        self.replaces = replaces
        self.requestId = requestId
        self.text = text
        self.uiAction = uiAction
        self.videos = videos
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: StringCodingKey.self)
        self.audio = try container.decodeIfPresent(AudioSource.self, forKey: .audio)
        self.files = try container.decodeIfPresent([FileSource].self, forKey: .files)
        self.images = try container.decodeIfPresent([ImageSourcePayload].self, forKey: .images)
        self.maxOutputTokens = try container.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        self.reasoningEffort = try container.decodeIfPresent(String.self, forKey: .reasoningEffort)
        self.replaces = try container.decodeIfPresent(String.self, forKey: .replaces)
        self.requestId = try container.decodeIfPresent(String.self, forKey: .requestId)
        self.text = try container.decode(String.self, forKey: .text)
        self.uiAction = try container.decodeIfPresent([String: RawJSON].self, forKey: .uiAction)
        self.videos = try container.decodeIfPresent([VideoSource].self, forKey: .videos)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: StringCodingKey.self)
        try container.encodeIfPresent(audio, forKey: .audio)
        try container.encodeIfPresent(files, forKey: .files)
        try container.encodeIfPresent(images, forKey: .images)
        try container.encodeIfPresent(maxOutputTokens, forKey: .maxOutputTokens)
        try container.encodeIfPresent(reasoningEffort, forKey: .reasoningEffort)
        try container.encodeIfPresent(replaces, forKey: .replaces)
        try container.encodeIfPresent(requestId, forKey: .requestId)
        try container.encode(text, forKey: .text)
        try container.encodeIfPresent(uiAction, forKey: .uiAction)
        try container.encodeIfPresent(videos, forKey: .videos)
    }

}

extension CreateResponseRequest: Hashable {
    static func == (lhs: CreateResponseRequest, rhs: CreateResponseRequest) -> Bool {
        lhs.audio == rhs.audio &&
        lhs.files == rhs.files &&
        lhs.images == rhs.images &&
        lhs.maxOutputTokens == rhs.maxOutputTokens &&
        lhs.reasoningEffort == rhs.reasoningEffort &&
        lhs.replaces == rhs.replaces &&
        lhs.requestId == rhs.requestId &&
        lhs.text == rhs.text &&
        lhs.uiAction == rhs.uiAction &&
        lhs.videos == rhs.videos
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(audio)
        hasher.combine(files)
        hasher.combine(images)
        hasher.combine(maxOutputTokens)
        hasher.combine(reasoningEffort)
        hasher.combine(replaces)
        hasher.combine(requestId)
        hasher.combine(text)
        hasher.combine(uiAction)
        hasher.combine(videos)
    }
}
