import Foundation
import VisionAgentsCore

/// Which Stream app this is, where this app's own backend is, and the tools it answers itself.
///
/// The app talks to Stream's hosted router, which knows the app by its key and this device's
/// user by the token the backend signs. There is no sign-in: the backend signs for whoever
/// the app says it is, so the constants are the whole of the configuration.
enum Demo {
    /// This app's own backend, which is `go run ./backend`. A device holds no secret to sign
    /// a Stream token with, so it asks this for one. The simulator reaches the Mac's
    /// localhost, so this works as it stands. On a device, put your Mac's address on the
    /// network here, for example http://192.168.1.20:8099.
    static let backendURL = URL(string: "http://localhost:8099")!

    /// Your Stream app's `STREAM_API_KEY`: the hosted router keeps the agent under that app,
    /// and the chat channel a written conversation is kept in and the call the agent is on
    /// are both in it, and this device connects to them there.
    static let streamAPIKey = ""

    /// The agent `go run ./configure` stored, by the name in `agent.yaml`.
    ///
    /// The app is told which agent it talks to rather than finding out, because reading the
    /// configs is server-side only.
    static let agentName = "swift_demo"

    /// Who this device is. A call waiting for approval is addressed to them.
    static let user = User(id: "demo-caller", name: "Demo caller")

    static let agents: VisionAgents = {
        let agents = VisionAgents(apiKey: streamAPIKey)
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

    /// The tools the agent may call, which run on this phone.
    static let tools = [lookupOrder, issueRefund]

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

    private static let lookupOrder = AgentTool(
        name: "lookup_order",
        description: "Look up one of the caller's orders by its order number, such as A-1042.",
        parameters: .strings(
            ["order_id": "the order number, such as A-1042"], required: ["order_id"]),
        displayTitle: "Looking up your order"
    ) { arguments in
        let id = arguments["order_id"]?.stringValue?.uppercased() ?? ""
        guard let order = orders[id] else {
            return "There is no order \(id) on this account."
        }
        return "Order \(id): \(order)."
    }

    /// Refunds an order, once the person says it may.
    ///
    /// The agent decides a refund is owed; the person decides whether it goes ahead. Each call
    /// waits in the session's `approvals` until they answer: on the reply's step in Chat,
    /// where Stream's AI components ask, or on the card over the call in Voice.
    private static let issueRefund = AgentTool(
        name: "issue_refund",
        description: "Refund one of the caller's orders to the card it was paid with, once the "
            + "refund skill has said they are owed it.",
        parameters: .strings(
            [
                "order_id": "the order number, such as A-1042",
                "amount": "how much to refund, such as 78.00",
                "reason": "why the caller is owed it, in a few words they would recognise",
            ],
            required: ["order_id", "amount", "reason"]),
        displayTitle: "Refunding your order",
        approval: .init(
            title: "Refund this order?",
            message: "The money goes back to the card the order was paid with.",
            reasonArgument: "reason",
            allowTitle: "Refund",
            declineTitle: "Not now")
    ) { arguments in
        let id = arguments["order_id"]?.stringValue?.uppercased() ?? ""
        let amount = arguments["amount"]?.stringValue ?? ""
        guard orders[id] != nil else {
            return "There is no order \(id) on this account."
        }
        return "Refunded \(amount) for order \(id) to the card ending 4242. It shows within five "
            + "working days."
    }
}
