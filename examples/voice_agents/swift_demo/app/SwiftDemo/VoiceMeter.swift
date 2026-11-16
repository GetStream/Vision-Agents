import AVFoundation
import StreamVideo
import StreamWebRTC
import os

/// How loud one side of the call is, measured from its audio as it flows.
///
/// The call's own audio levels arrive a few times a second, too coarse to move with the words.
/// This hears every 10 ms of audio instead, so the glow swells on each syllable and settles in
/// each pause. The agent is heard through `renderer`, added to its track; this device's
/// microphone through `filter`, set on the call as its audio filter, which reads the audio
/// after WebRTC's processing and passes it on unchanged. Both are called on WebRTC's audio
/// threads; the view reads on the main thread.
final class VoiceMeter: Sendable {
    private struct Window {
        var sumOfSquares: Float = 0
        var samples = 0
        var lastAudible: TimeInterval = -.infinity
        /// When the current stretch of audible sound began.
        var stretchStart: TimeInterval = -.infinity
    }

    /// How long a voice counts as talking after it was last heard, so the gaps between words
    /// and sentences read as pauses rather than the end of what is being said.
    private static let hold: TimeInterval = 0.6

    /// Hears a track as it plays, for the agent.
    let renderer: TrackRenderer
    /// Hears this device's microphone, for the person.
    let filter: MicrophoneFilter

    private let window: OSAllocatedUnfairLock<Window>
    private let onset: TimeInterval

    /// - Parameters:
    ///   - audible: The level, from 0 to 1, below which the voice counts as silent.
    ///   - onset: How long the voice has to be heard before it counts as talking.
    init(audible: Float, onset: TimeInterval = 0) {
        let window = OSAllocatedUnfairLock(initialState: Window())
        let measure: @Sendable (Float, Int) -> Void = { sumOfSquares, samples in
            Self.add(sumOfSquares, samples, to: window, audible: audible)
        }
        self.window = window
        self.onset = onset
        renderer = TrackRenderer(measure: measure)
        filter = MicrophoneFilter(measure: measure)
    }

    /// The voice's level since the last reading, from 0 to 1, or nil when it is not talking.
    func read() -> Double? {
        let now = ProcessInfo.processInfo.systemUptime
        let onset = onset
        return window.withLock { window in
            defer {
                window.sumOfSquares = 0
                window.samples = 0
            }
            guard now - window.lastAudible < Self.hold,
                  window.lastAudible - window.stretchStart >= onset
            else { return nil }
            guard window.samples > 0 else { return 0 }
            return Double(Self.level(sumOfSquares: window.sumOfSquares, samples: window.samples))
        }
    }

    private static func add(
        _ sumOfSquares: Float,
        _ samples: Int,
        to window: OSAllocatedUnfairLock<Window>,
        audible: Float
    ) {
        guard samples > 0 else { return }
        let level = level(sumOfSquares: sumOfSquares, samples: samples)
        let now = ProcessInfo.processInfo.systemUptime
        window.withLock { window in
            window.sumOfSquares += sumOfSquares
            window.samples += samples
            guard level > audible else { return }
            if now - window.lastAudible >= hold { window.stretchStart = now }
            window.lastAudible = now
        }
    }

    /// RMS scaled the way the web demo scales its microphone, so normal speech peaks near 1.
    private static func level(sumOfSquares: Float, samples: Int) -> Float {
        min(1, (sumOfSquares / Float(samples)).squareRoot() * 4.2)
    }

    /// The agent's audio, as its track plays.
    final class TrackRenderer: NSObject, RTCAudioRenderer, Sendable {
        private let measure: @Sendable (Float, Int) -> Void

        fileprivate init(measure: @escaping @Sendable (Float, Int) -> Void) {
            self.measure = measure
        }

        func render(pcmBuffer: AVAudioPCMBuffer) {
            let frames = Int(pcmBuffer.frameLength)
            if let channel = pcmBuffer.floatChannelData?[0] {
                measure(sumOfSquares(frames) { channel[$0] }, frames)
            } else if let channel = pcmBuffer.int16ChannelData?[0] {
                measure(sumOfSquares(frames) { Float(channel[$0]) / 32768 }, frames)
            }
        }
    }

    /// This device's microphone, after WebRTC's processing. Its samples are WebRTC's, scaled to
    /// 16-bit range, and pass through unchanged.
    struct MicrophoneFilter: AudioFilter {
        let id = "io.getstream.visionagents.SwiftDemo.VoiceMeter"
        private let measure: @Sendable (Float, Int) -> Void

        fileprivate init(measure: @escaping @Sendable (Float, Int) -> Void) {
            self.measure = measure
        }

        func applyEffect(to audioBuffer: inout RTCAudioBuffer) {
            let frames = Int(audioBuffer.frames)
            let channel = audioBuffer.rawBuffer(forChannel: 0)
            measure(sumOfSquares(frames) { channel[$0] / 32768 }, frames)
        }
    }
}

private func sumOfSquares(_ frames: Int, sample: (Int) -> Float) -> Float {
    var sum: Float = 0
    for index in 0..<frames {
        let value = sample(index)
        sum += value * value
    }
    return sum
}
