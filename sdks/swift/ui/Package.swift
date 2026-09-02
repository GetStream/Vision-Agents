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
    dependencies: [
        .package(name: "vision-agents-core", path: "../core")
    ],
    targets: [
        .target(
            name: "VisionAgentsUI",
            dependencies: [
                .product(name: "VisionAgentsCore", package: "vision-agents-core")
            ]
        )
    ]
)
