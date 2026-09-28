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
    //
    // AG-UI is here rather than in a package of its own because it is what the session
    // speaks: `aguiEvents()` and the approvals are the protocol's own types, so a caller
    // holding a session already holds them. It is pure Swift with no dependencies of its
    // own, which is why it can be in core when Stream's Video SDK cannot.
    //
    // Pinned to a revision because the repository has no tags yet. It becomes
    // `from: "0.1.0"` the moment it publishes one; a branch would make resolution
    // irreproducible.
    dependencies: [
        .package(url: "https://github.com/apple/swift-openapi-runtime", from: "1.12.1"),
        .package(url: "https://github.com/apple/swift-openapi-urlsession", from: "1.3.1"),
        .package(url: "https://github.com/apple/swift-http-types", from: "1.4.0"),
        .package(
            url: "https://github.com/martinmitrevski/ag-ui-swift",
            revision: "48e7b711f738bd029dc6ca0db2b6a126a0f1bb1d"),
    ],
    targets: [
        .target(
            name: "VisionAgentsCore",
            dependencies: [
                .product(name: "OpenAPIRuntime", package: "swift-openapi-runtime"),
                .product(name: "OpenAPIURLSession", package: "swift-openapi-urlsession"),
                .product(name: "HTTPTypes", package: "swift-http-types"),
                .product(name: "AGUI", package: "ag-ui-swift"),
            ]
        ),
        .testTarget(
            name: "VisionAgentsCoreTests",
            dependencies: [
                "VisionAgentsCore",
                // Declared here as well as on the library, because a test that checks the
                // events against AG-UI's own verifier and reducer imports them itself.
                .product(name: "AGUI", package: "ag-ui-swift"),
            ]
        ),
    ]
)
