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

    /// The backend that signs this device's Stream token: the swift demo's, `go run ./backend`
    /// in examples/voice_agents/swift_demo. On a device, the Mac's address, as above.
    static let backendURL = URL(string: "http://localhost:8099")!

    static let customerID = "examples"

    /// The `STREAM_API_KEY` the router runs with: the call the agent is on is in that Stream
    /// app, and this device joins it there.
    static let streamAPIKey = ""

    static let agentName = "flower_spotter"

    private static let user = User(id: "flower-spotter", name: "Flower spotter")

    static let agents: VisionAgents = {
        let agents = VisionAgents(url: routerURL, customerID: customerID, apiKey: streamAPIKey)
        agents.setUser(user) { try await streamToken() }
        return agents
    }()

    /// Asks the backend for a Stream user token, which joins the agent's call.
    private static func streamToken() async throws -> String {
        var request = URLRequest(url: backendURL.appending(path: "stream-token"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["user_id": user.id])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw AgentsError.unreadable("the backend would not mint a Stream token")
        }
        return try JSONDecoder().decode(MintedToken.self, from: data).token
    }

    private struct MintedToken: Decodable {
        let token: String
    }
}
