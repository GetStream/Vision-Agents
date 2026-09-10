import Foundation
import VisionAgentsCore

/// Where the router is, and how this app talks to it.
///
/// There is no sign-in. The router is running in the mode where it trusts the customer
/// id it is given, which is what `docker compose up` does.
enum Demo {
    /// The simulator reaches the Mac's localhost, so this works as it stands. On a device,
    /// which is where the camera is, this has to be the Mac's address on the same Wi-Fi:
    /// `ipconfig getifaddr en0`, for example `http://192.168.1.20:8080`.
    static let routerURL = URL(string: "http://localhost:8080")!

    static let customerID = "examples"

    static let agentName = "flower_spotter"

    static let agents = VisionAgents(url: routerURL, customerID: customerID)
}
