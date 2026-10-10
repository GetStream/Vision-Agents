// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "vision-agents-chat",
    // iOS only: Stream Chat's macOS build does not compile under Swift 6.
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "VisionAgentsChat", targets: ["VisionAgentsChat"])
    ],
    // A package of its own, so an app that never reads the channel resolves no chat SDK. 5.3 is
    // the oldest Stream Chat that builds under Swift 6: 4.x's ChatClient is not Sendable, and
    // 5.0 to 5.2 resolve a stream-core-swift that does not compile.
    dependencies: [
        .package(name: "vision-agents-core", path: "../core"),
        .package(url: "https://github.com/GetStream/stream-chat-swift", from: "5.3.0"),
    ],
    targets: [
        .target(
            name: "VisionAgentsChat",
            dependencies: [
                .product(name: "VisionAgentsCore", package: "vision-agents-core"),
                .product(name: "StreamChat", package: "stream-chat-swift"),
            ]
        )
    ]
)
