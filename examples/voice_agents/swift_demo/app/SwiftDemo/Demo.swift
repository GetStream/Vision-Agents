import Foundation
import VisionAgentsCore

/// Where the router is, and the one tool this app answers itself.
///
/// There is no sign-in. The router is running in the mode where it trusts the customer id it
/// is given, which is what `docker compose up` and `go run ./cmd/router` do, so two constants
/// are the whole of the configuration. In front of a real deployment the customer id would
/// come from your own backend along with a token, and nothing else here would change.
enum Demo {
    /// The simulator reaches the Mac's localhost, so this works as it stands. On a device,
    /// put your Mac's address on the network here, for example http://192.168.1.20:8080.
    static let routerURL = URL(string: "http://localhost:8080")!

    /// Whichever customer id you started the router with. `compose.yaml` builds the
    /// dashboard with `examples`, and a config belongs to one customer, so anything else
    /// here works but will not appear on the dashboard.
    static let customerID = "examples"

    /// The agent `configure` stored. Leave it empty to be asked to pick one.
    static let agentName = "swift_demo"

    static let agents = VisionAgents(url: routerURL, customerID: customerID)

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
