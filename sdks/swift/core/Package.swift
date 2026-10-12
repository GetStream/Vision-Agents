// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "vision-agents-core",
    // iOS 17 is the floor because the session state is @Observable. The alternative was
    // shipping an ObservableObject path beside it for older phones, which doubles the state
    // layer to serve devices that will not be running a new SDK anyway. iOS only, because
    // Stream Video is, and Stream Chat's macOS build does not compile under Swift 6.
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "VisionAgentsCore", targets: ["VisionAgentsCore"])
    ],
    // The one package an app depends on. The conversation comes off the session socket; a
    // text session's channel opens on Stream Chat and a call is joined on Stream Video, as
    // the user `setUser` named. The views are Stream Chat's AI components (StreamChatAI, in
    // the same stream-chat-swift package), which an app adds itself. 5.13 is the first
    // stream-chat-swift with them, so an app adding them resolves the version this does.
    dependencies: [
        .package(url: "https://github.com/apple/swift-openapi-runtime", from: "1.12.1"),
        .package(url: "https://github.com/apple/swift-openapi-urlsession", from: "1.3.1"),
        .package(url: "https://github.com/apple/swift-http-types", from: "1.4.0"),
        .package(url: "https://github.com/GetStream/stream-chat-swift", from: "5.13.0"),
        .package(url: "https://github.com/GetStream/stream-video-swift", from: "1.51.0"),
    ],
    targets: [
        .target(
            name: "VisionAgentsCore",
            dependencies: [
                .product(name: "OpenAPIRuntime", package: "swift-openapi-runtime"),
                .product(name: "OpenAPIURLSession", package: "swift-openapi-urlsession"),
                .product(name: "HTTPTypes", package: "swift-http-types"),
                .product(name: "StreamChat", package: "stream-chat-swift"),
                .product(name: "StreamVideo", package: "stream-video-swift"),
            ]
        ),
        .testTarget(name: "VisionAgentsCoreTests", dependencies: ["VisionAgentsCore"]),
    ]
)
