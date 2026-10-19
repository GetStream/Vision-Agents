import CoreImage
import Foundation
import Testing

@testable import VisionAgentsCore

@Suite struct CameraFramesTests {
    private let start = Date(timeIntervalSince1970: 1_800_000_000)

    @Test func aFrameIsKeptTwiceASecondAtMost() throws {
        let frames = CameraFrames()
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))

        frames.offer(image, at: start)
        frames.offer(image, at: start.addingTimeInterval(0.2))
        frames.offer(image, at: start.addingTimeInterval(0.6))

        #expect(try frames.select(at: start.addingTimeInterval(1), limit: 8).count == 2)
    }

    @Test func onlyTheNewestEightAreKept() throws {
        let frames = CameraFrames()
        for second in 0..<12 {
            frames.record(Data([UInt8(second)]), at: start.addingTimeInterval(Double(second) / 4))
        }

        let kept = try frames.select(at: start.addingTimeInterval(3), limit: 8)
        #expect(kept.map(\.jpeg) == (4..<12).map { Data([UInt8($0)]) })
    }

    @Test func theNewestFramesAtTheMomentAreGivenOldestFirst() throws {
        let frames = CameraFrames()
        for second in 0..<6 {
            frames.record(Data([UInt8(second)]), at: start.addingTimeInterval(Double(second)))
        }

        let picked = try frames.select(at: start.addingTimeInterval(2.5), limit: 2)

        // Frames up to the moment, and the clock skew past it a phone may run ahead by.
        #expect(picked.map(\.jpeg) == [Data([3]), Data([4])])
    }

    @Test func nothingRecentEnoughIsSaidRatherThanAnOldFrameGiven() {
        let frames = CameraFrames()
        frames.record(Data([1]), at: start)

        #expect(throws: CameraFrames.NoFrames.self) {
            try frames.select(at: start.addingTimeInterval(CameraFrames.freshFor + 1), limit: 1)
        }
    }

    @Test func aCameraTurnedOffForgetsWhatItShowed() {
        let frames = CameraFrames()
        frames.record(Data([1]), at: start)

        frames.clear()

        #expect(throws: CameraFrames.NoFrames.self) { try frames.select(at: start, limit: 1) }
    }
}
