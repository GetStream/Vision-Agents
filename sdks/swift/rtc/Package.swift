// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "vision-agents-rtc",
    // iOS only, because Stream's Video SDK is: its manifest declares `[.iOS(.v13)]` and
    // WebRTC has no macOS slice here.
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "VisionAgentsRTC", targets: ["VisionAgentsRTC"])
    ],
    // A package of its own rather than a third product of the core package. SPM resolves and
    // fetches every dependency a manifest declares, whether or not the product you depend on
    // uses it, so folding this in would put StreamWebRTC's binary in the checkout of anybody
    // who only wanted to hold a text conversation.
    dependencies: [
        .package(name: "vision-agents-core", path: "../core"),
        .package(url: "https://github.com/GetStream/stream-video-swift", from: "1.51.0"),
    ],
    targets: [
        .target(
            name: "VisionAgentsRTC",
            dependencies: [
                .product(name: "VisionAgentsCore", package: "vision-agents-core"),
                .product(name: "StreamVideo", package: "stream-video-swift"),
                .product(name: "StreamVideoSwiftUI", package: "stream-video-swift"),
            ]
        )
    ]
)
