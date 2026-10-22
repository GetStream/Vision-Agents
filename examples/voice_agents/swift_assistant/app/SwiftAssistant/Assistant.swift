import Foundation
import VisionAgentsCore

/// Where the router is, where the token backend is, and who this phone is.
///
/// There is no sign-in. The router runs in the mode where it trusts the customer and user ids
/// it is given, `ROUTER_AUTH_MODE=proxy`, so these constants are the whole configuration. In
/// front of a real deployment the router is reached by the key and the same token, and
/// nothing else here would change.
enum Assistant {
    /// The simulator reaches the Mac's localhost. On a device, put your Mac's address on the
    /// network here, for example http://192.168.1.20:8080.
    static let routerURL = URL(string: "http://localhost:8080")!

    /// This app's backend, `go run ./backend`. A phone holds no secret to sign a Stream token
    /// with and may not reach its person's Gmail connection on the router, so it asks this.
    static let backendURL = URL(string: "http://localhost:8098")!

    /// The scheme the router sends a sign-in sheet back to, once the person has answered the
    /// provider an agent's tool needs, such as Linear or Gmail. The router takes an app's own
    /// scheme only in reverse domain form.
    static let callbackScheme = "io.getstream.visionagents.swiftassistant"

    /// Where the sign-ins a conversation asks for go back to, kept with its session.
    static let returnURL = URL(string: callbackScheme + ":/connected")!

    /// The customer `go run ./configure` stored the agent under.
    static let customerID = "examples"

    /// The `STREAM_API_KEY` the router runs with, which the conversations' channels and calls
    /// are in.
    static let streamAPIKey = ""

    /// The agent `go run ./configure` stored, by the name in `agent.yaml`.
    static let agentName = "swift_assistant"

    /// The model a written conversation runs on. Grok reads photos and streams its reasoning,
    /// which is what fills a reply's thinking panel. A call keeps the agent's faster model.
    static let chatModel = "xai/grok-4.7"

    /// Who this phone is. Every conversation listed is theirs.
    static let user = User(id: "juno-owner", name: "Alex")

    static let agents: VisionAgents = {
        let agents = VisionAgents(url: routerURL, customerID: customerID, apiKey: streamAPIKey)
        agents.setUser(user) { try await streamToken() }
        return agents
    }()

    /// The agent's conversations: opening one, and listing and searching the ones there were.
    static var sessions: Sessions { agents.agent(agentName).sessions }

    /// The tools the agent may call, which run on this phone.
    static let tools = DeviceTools.all

    /// Asks the backend for a Stream user token, which opens the chat and joins the call.
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
