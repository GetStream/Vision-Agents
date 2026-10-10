import Foundation
import VisionAgentsCore

/// Where the router is, where this app's own backend is, and the one tool it answers itself.
///
/// There is no sign-in. The router is running in the mode where it trusts the customer id it
/// is given, `ROUTER_AUTH_MODE=noauth`, so the constants are the whole of the configuration.
/// In front of a real deployment the router is reached by the key and the same token, and
/// nothing else here would change.
enum Demo {
    /// The simulator reaches the Mac's localhost, so this works as it stands. On a device,
    /// put your Mac's address on the network here, for example http://192.168.1.20:8080.
    static let routerURL = URL(string: "http://localhost:8080")!

    /// This app's own backend, which is `go run ./backend`. A device holds no secret to sign
    /// a Stream token with, so it asks this for one.
    static let backendURL = URL(string: "http://localhost:8099")!

    /// Whichever customer id you started the router with. `compose.yaml` builds the
    /// dashboard with `examples`, and a config belongs to one customer, so anything else
    /// here works but will not appear on the dashboard.
    static let customerID = "examples"

    /// The `STREAM_API_KEY` the router runs with: the call the agent is on is in that Stream
    /// app, and this device joins it there.
    static let streamAPIKey = ""

    /// The agent config `go run ./configure` stored, which prints the id to put here.
    ///
    /// An id rather than a name because reading the configs is server-side only: the app is
    /// told which agent it talks to rather than finding out.
    static let agentID = ""

    private static let user = User(id: "demo-caller", name: "Demo caller")

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

    /// Orders the agent can look up.
    ///
    /// The point of this being here rather than in the backend is that it does not have to
    /// leave the phone. A real one would read whatever the signed-in person's session gives
    /// it; the agent asks, and only ever sees the answer.
    private static let orders: [String: String] = [
        "A-1042": "2 Larkspur wool throws, 78.00, paid by card ending 4242, "
            + "delivered on 14 August, unopened",
        "A-1043": "1 linen apron, 24.00, paid by card ending 4242, "
            + "delivered on 2 September, worn",
    ]

    static let lookupOrder = AgentTool(
        name: "lookup_order",
        description: "Look up one of the caller's orders by its order number, such as A-1042.",
        parameters: .strings(
            ["order_id": "the order number, such as A-1042"], required: ["order_id"])
    ) { arguments in
        let id = arguments["order_id"]?.stringValue.uppercased() ?? ""
        guard let order = orders[id] else {
            return "There is no order \(id) on this account."
        }
        return "Order \(id): \(order)."
    }
}
