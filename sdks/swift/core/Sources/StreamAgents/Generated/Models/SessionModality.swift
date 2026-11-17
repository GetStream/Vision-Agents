import Foundation
import StreamCore

/// How the user took part: text for a conversation held in writing, voice for a call, and video once the agent has seen the user's video. It only moves up, from text or voice to video.
final class SessionModality: RawRepresentable, Codable, Hashable, Sendable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = rawValue
    }

    static let text = SessionModality(rawValue: "text")
    static let video = SessionModality(rawValue: "video")
    static let voice = SessionModality(rawValue: "voice")
}