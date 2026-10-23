// swift-tools-version:6.0
import PackageDescription

// The Swift SDK, whose sources are in sdks/swift/core. The manifest is at the root of the
// repository because that is the only place SwiftPM looks for it: an app adds this
// repository's URL, and each `v` tag, the Python packages' too, is a release of it.
let package = Package(
    name: "vision-agents-core",
    // iOS 15 is the floor. The session state is an ObservableObject rather than @Observable,
    // which would have made it iOS 17, and nothing else here or in the Stream SDKs needs more.
    // iOS only, because Stream Video is, and Stream Chat's macOS build does not compile under
    // Swift 6.
    platforms: [.iOS(.v15)],
    products: [
        .library(name: "VisionAgentsCore", targets: ["VisionAgentsCore"])
    ],
    // The one package an app depends on. The conversation comes off the session socket; a
    // text session's channel opens on Stream Chat and a call is joined on Stream Video, as
    // the user `setUser` named. The views are Stream Chat's AI components (StreamChatAI, in
    // the same stream-chat-swift package), which an app adds itself. 5.13 is the first
    // stream-chat-swift with them, so an app adding them resolves the version this does.
    // StreamCore is what both of them stand on: the generated client runs on its runtime, as
    // theirs does, and `User` and `RawJSON` are its own.
    dependencies: [
        .package(url: "https://github.com/GetStream/stream-core-swift.git", from: "0.13.0"),
        .package(url: "https://github.com/GetStream/stream-chat-swift", from: "5.13.0"),
        .package(url: "https://github.com/GetStream/stream-video-swift", from: "1.54.0"),
    ],
    targets: [
        .target(
            name: "VisionAgentsCore",
            dependencies: [
                .product(name: "StreamCore", package: "stream-core-swift"),
                .product(name: "StreamChat", package: "stream-chat-swift"),
                .product(name: "StreamVideo", package: "stream-video-swift"),
            ],
            path: "sdks/swift/core/Sources/VisionAgentsCore"
        ),
        .testTarget(
            name: "VisionAgentsCoreTests",
            dependencies: ["VisionAgentsCore"],
            path: "sdks/swift/core/Tests/VisionAgentsCoreTests"
        ),
    ]
)
