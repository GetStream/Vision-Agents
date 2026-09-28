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

    /// The tools this app answers. A written conversation and a spoken one get the same two:
    /// looking an order up runs without asking, and refunding one never does.
    static var tools: [AgentTool] { [lookupOrder, refundOrder] }

    /// Orders the agent can look up.
    ///
    /// The point of this being here rather than in the backend is that it does not have to
    /// leave the phone. A real one would read whatever the signed-in person's session gives
    /// it; the agent asks, and only ever sees the answer.
    /// Delivery is given as how long ago rather than as a date, because nothing tells the
    /// model what today is. A policy with a thirty-day window and an order delivered "on 14
    /// August" is a sum the model cannot do, so it asked the caller what the date was and the
    /// refund never got decided. Both of these are mocked anyway: a real one would read the
    /// signed-in person's orders and could say either.
    private static let orders: [String: String] = [
        "A-1042": "2 Larkspur wool throws, 78.00, paid by card ending 4242, "
            + "delivered 6 days ago, unopened",
        "A-1043": "1 linen apron, 24.00, paid by card ending 4242, "
            + "delivered 40 days ago, worn",
    ]

    /// Refunds one, which nobody does on the model's word alone.
    ///
    /// The `approval` closure is what puts a person in the loop. The SDK does not run this
    /// tool when the model asks for it: it publishes the question on
    /// `session.pendingApprovals`, `ConversationView` puts a card on screen for it, and the
    /// answer decides whether the body below ever runs. The agent is holding its turn open on
    /// the call the whole time, so what it says next is what actually happened.
    static let refundOrder = AgentTool(
        name: "refund_order",
        description: """
            Refund an order the caller is owed money for, to the payment method they used. \
            The caller approves it on their phone before it goes through; that happens on its \
            own, so call this and then say what came back.
            """,
        parameters: .strings(
            [
                "order_id": "the order number, such as A-1042",
                "amount": "how much to refund, such as 78.00",
            ],
            required: ["order_id", "amount"]),
        approval: { arguments in
            let id = arguments["order_id"]?.stringValue.uppercased() ?? ""
            let amount = arguments["amount"]?.stringValue ?? ""
            return "Refund \(amount) for order \(id)?"
        }
    ) { arguments in
        let id = arguments["order_id"]?.stringValue.uppercased() ?? ""
        let amount = arguments["amount"]?.stringValue ?? ""
        guard orders[id] != nil else {
            return "There is no order \(id) on this account."
        }
        return "Refunded \(amount) for order \(id) to the card ending 4242."
    }

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
