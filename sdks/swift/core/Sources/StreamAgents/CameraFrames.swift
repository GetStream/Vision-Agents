import CoreImage
import Foundation
import ImageIO
import StreamVideo

/// What this device's camera showed lately, kept so the agent can look when it asks.
///
/// The router asks with `get_video_frames` when its vision skill needs to see, naming the
/// moment it took the question and how many frames it wants. The frames are sampled off the
/// call's own camera twice a second, through a video filter that leaves every frame as it was,
/// and the newest eight are kept: about four seconds.
final class CameraFrames: @unchecked Sendable {
    /// The tool the router asks for frames with. It is answered here rather than offered to the
    /// model as one of the app's tools.
    static let toolName = "get_video_frames"

    struct Frame: Equatable {
        let id: Int
        let capturedAt: Date
        let jpeg: Data
    }

    /// How often a frame is kept, in seconds.
    static let every: TimeInterval = 0.5
    /// How many are kept.
    static let keep = 8
    /// How old the newest may be at the moment asked about, as the router's own workers allow.
    static let freshFor: TimeInterval = 5
    /// How far this device's clock may run ahead of the router's: a frame taken just after the
    /// question by this clock may be just before it by the router's.
    static let skew: TimeInterval = 2
    /// The longest side a frame is kept at, in pixels.
    private static let longestSide: CGFloat = 1024

    private let lock = NSLock()
    private var kept: [Frame] = []
    private var lastKept = Date.distantPast
    private var counter = 0
    private let context = CIContext()

    /// The filter the call's camera runs through, which keeps a frame now and then and leaves
    /// each one as it was.
    var filter: VideoFilter {
        VideoFilter(id: "vision-agents-frames", name: "Frames the agent may look at") { [weak self] input in
            self?.offer(input.originalImage.oriented(input.originalImageOrientation))
            return input.originalImage
        }
    }

    /// Keeps the frame if one is due.
    func offer(_ image: CIImage, at now: Date = Date()) {
        lock.lock()
        let due = now.timeIntervalSince(lastKept) >= Self.every
        if due {
            lastKept = now
        }
        lock.unlock()
        guard due, let jpeg = jpeg(image) else { return }
        record(jpeg, at: now)
    }

    func record(_ jpeg: Data, at capturedAt: Date) {
        lock.lock()
        defer { lock.unlock() }
        counter += 1
        kept.append(Frame(id: counter, capturedAt: capturedAt, jpeg: jpeg))
        if kept.count > Self.keep {
            kept.removeFirst(kept.count - Self.keep)
        }
    }

    /// Forgets every frame, for a camera turned off: what it showed before is not what it shows.
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        kept = []
        lastKept = .distantPast
    }

    /// The newest `limit` frames taken at or before `at`, oldest first, as the router's own
    /// workers pick them: never a newer frame in place of one the question was asked about.
    func select(at: Date, limit: Int) throws -> [Frame] {
        lock.lock()
        defer { lock.unlock() }
        let before = kept.filter { $0.capturedAt <= at.addingTimeInterval(Self.skew) }
        guard let newest = before.last, at.timeIntervalSince(newest.capturedAt) <= Self.freshFor else {
            throw NoFrames()
        }
        return Array(before.suffix(max(1, min(limit, Self.keep))))
    }

    private func jpeg(_ image: CIImage) -> Data? {
        let longest = max(image.extent.width, image.extent.height)
        guard longest > 0 else { return nil }
        let scale = min(1, Self.longestSide / longest)
        return context.jpegRepresentation(
            of: image.transformed(by: CGAffineTransform(scaleX: scale, y: scale)),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
            options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.7])
    }

    /// The camera has shown nothing recent enough.
    struct NoFrames: LocalizedError {
        var errorDescription: String? {
            "The camera has shown nothing in the last few seconds; ask them to point it at what they mean again."
        }
    }
}
