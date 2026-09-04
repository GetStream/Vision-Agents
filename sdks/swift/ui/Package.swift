// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "vision-agents-ui",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "VisionAgentsUI", targets: ["VisionAgentsUI"])
    ],
    // Core is by path because both live in this repo. Published, this becomes a version
    // requirement on the same package; a path dependency cannot be resolved by anybody who
    // does not have the whole repo checked out.
    //
    // StreamChatAI is where the transcript's markdown rendering and the composer come from.
    // It is Stream's own AI chat surface, so an app using both gets one composer rather than
    // two that almost agree. It brings Splash, swift-markdown-ui and the MCP SDK with it,
    // which is why it is a dependency of `ui` alone and not of `core`.
    dependencies: [
        .package(name: "vision-agents-core", path: "../core"),
        .package(url: "https://github.com/GetStream/stream-chat-swift-ai.git", from: "0.7.0"),
    ],
    targets: [
        .target(
            name: "VisionAgentsUI",
            dependencies: [
                .product(name: "VisionAgentsCore", package: "vision-agents-core"),
                .product(name: "StreamChatAI", package: "stream-chat-swift-ai"),
            ]
        )
    ]
)
