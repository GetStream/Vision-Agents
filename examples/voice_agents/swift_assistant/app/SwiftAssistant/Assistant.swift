import Foundation
import VisionAgentsCore

/// Which Stream app this is, where the token backend is, and who this phone is.
///
/// The app talks to Stream's hosted router, which knows the app by its key and this phone's
/// user by the token the backend signs. There is no sign-in: the backend signs for whoever
/// the app says it is, so these constants are the whole configuration.
enum Assistant {
    /// This app's backend, `go run ./backend`. A phone holds no secret to sign a Stream token
    /// with and may not reach its person's Gmail connection on the router, so it asks this.
    /// The simulator reaches the Mac's localhost. On a device, put your Mac's address on the
    /// network here, for example http://192.168.1.20:8098.
    static let backendURL = URL(string: "http://localhost:8098")!

    /// The scheme the router sends a sign-in sheet back to, once the person has answered the
    /// provider an agent's tool needs, such as Linear or Gmail. The router takes an app's own
    /// scheme only in reverse domain form.
    static let callbackScheme = "io.getstream.visionagents.swiftassistant"

    /// Where the sign-ins a conversation asks for go back to, kept with its session.
    static let returnURL = URL(string: callbackScheme + ":/connected")!

    /// Your Stream app's `STREAM_API_KEY`: the hosted router keeps the agent under that app,
    /// and the conversations' channels and calls are in it.
    static let streamAPIKey = ""

    /// The agent `go run ./configure` stored, by the name in `agent.yaml`.
    static let agentName = "swift_assistant"

    /// The model a written conversation runs on. Grok reads photos and streams its reasoning,
    /// which is what fills a reply's thinking panel. A call keeps the agent's faster model.
    static let chatModel = "xai/grok-4.7"

    /// Who this phone is. Every conversation listed is theirs.
    static let user = User(id: "juno-owner", name: "Alex")

    static let agents: VisionAgents = {
        let agents = VisionAgents(apiKey: streamAPIKey)
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
