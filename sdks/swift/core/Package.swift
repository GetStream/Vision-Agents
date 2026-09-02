// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "vision-agents-core",
    // iOS 17 is the floor because the session state is @Observable. The alternative was
    // shipping an ObservableObject path beside it for older phones, which doubles the state
    // layer to serve devices that will not be running a new SDK anyway.
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "VisionAgentsCore", targets: ["VisionAgentsCore"])
    ],
    // Deliberately no Stream SDK here. The live conversation comes off the session socket and
    // the stored one comes from the router, which reads the chat channel on the caller's
    // behalf, so a chat SDK would be a second way to do what this already does. Callers who
    // want Stream Chat itself get the credentials from `chatToken` and bring their own
    // dependency. Stream's Video SDK is a real requirement and lives in the RTC package.
    dependencies: [
        .package(url: "https://github.com/apple/swift-openapi-runtime", from: "1.12.1"),
        .package(url: "https://github.com/apple/swift-openapi-urlsession", from: "1.3.1"),
        .package(url: "https://github.com/apple/swift-http-types", from: "1.4.0"),
    ],
    targets: [
        .target(
            name: "VisionAgentsCore",
            dependencies: [
                .product(name: "OpenAPIRuntime", package: "swift-openapi-runtime"),
                .product(name: "OpenAPIURLSession", package: "swift-openapi-urlsession"),
                .product(name: "HTTPTypes", package: "swift-http-types"),
            ]
        ),
        .testTarget(name: "VisionAgentsCoreTests", dependencies: ["VisionAgentsCore"]),
    ]
)
