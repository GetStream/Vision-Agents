import SwiftUI
import VisionAgentsCore

/// Closed captions for a spoken conversation: what is being said, a sentence at a time, gone a
/// moment after the talking stops.
///
/// The agent's reply is written before it is spoken, so its words are revealed at the pace of
/// its voice, counting the time its audio is heard, rather than as they arrive. What the
/// person said appears once it has been transcribed.
struct Captions {
    struct Caption: Equatable {
        var text: String
        var isPerson: Bool
    }

    private(set) var caption: Caption?
    private(set) var isVisible = false

    /// About how fast the agent speaks, in characters a second.
    private static let pace: Double = 15
    /// How long a caption stays up once nothing new is being said.
    private static let linger: TimeInterval = 2.5
    /// How long the agent has to be quiet, with its reply written, for the reply to be over.
    private static let finished: TimeInterval = 1.5

    private var turnID: String?
    private var spoken: TimeInterval = 0
    private var lastAudible: TimeInterval = -.infinity
    private var lastChange: TimeInterval = -.infinity

    mutating func update(
        turn: Turn?,
        agentIsAudible: Bool,
        agentIsAnswering: Bool,
        now: TimeInterval,
        elapsed: TimeInterval
    ) {
        var next: Caption?
        if let turn, turn.speaker.isAgent {
            if turn.id != turnID {
                turnID = turn.id
                spoken = 0
            }
            if agentIsAudible {
                spoken += elapsed
                lastAudible = now
            }
            // Once the agent has stopped for good, whatever the pacing has not reached yet is
            // shown, so a reply never ends half captioned.
            let isOver = spoken > 0 && !agentIsAnswering && now - lastAudible > Self.finished
            let count = isOver ? turn.text.count : min(turn.text.count, Int(spoken * Self.pace))
            let sentence = Self.sentence(in: turn.text, upTo: count)
            next = sentence.isEmpty ? nil : Caption(text: sentence, isPerson: false)
        } else if let turn {
            let sentence = Self.sentence(in: turn.text, upTo: turn.text.count)
            next = sentence.isEmpty ? nil : Caption(text: sentence, isPerson: true)
        }
        if let next, next != caption {
            caption = next
            lastChange = now
        }
        isVisible = caption != nil && (agentIsAudible || now - lastChange < Self.linger)
    }

    /// The sentence being said once the first `count` characters of `text` have been: whole
    /// words only, starting after the last sentence that has ended.
    static func sentence(in text: String, upTo count: Int) -> String {
        var said = String(text.prefix(count))
        if count < text.count {
            // Hold back a word until all of it has been said.
            if let space = said.lastIndex(where: \.isWhitespace) {
                said = String(said[..<space])
            } else {
                said = ""
            }
        }
        var start = said.startIndex
        var index = said.startIndex
        while index < said.endIndex {
            let next = said.index(after: index)
            if ".!?".contains(said[index]) || said[index].isNewline,
               next < said.endIndex, said[next].isWhitespace {
                start = next
            }
            index = next
        }
        return said[start...].trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// A caption, white on a dark band at the bottom of the screen like a film's. What the person
/// said is in italics.
struct CaptionView: View {
    let caption: Captions.Caption?
    let isVisible: Bool

    var body: some View {
        Text(caption?.text ?? "")
            .font(.body.weight(.medium))
            .italic(caption?.isPerson == true)
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .lineLimit(2)
            .truncationMode(.head)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.black.opacity(0.7), in: .rect(cornerRadius: 10))
            .opacity(isVisible ? 1 : 0)
            .animation(.easeOut(duration: 0.35), value: isVisible)
            .padding(.horizontal)
            // Two lines' room whether or not a caption is up, so the glow never moves.
            .frame(maxWidth: .infinity, minHeight: 72, maxHeight: 72, alignment: .bottom)
    }
}
