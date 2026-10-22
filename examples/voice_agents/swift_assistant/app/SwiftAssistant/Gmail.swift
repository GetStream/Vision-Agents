import Foundation

/// Sending from the person's Gmail.
///
/// The person connects Gmail in the chat, the first time the agent reaches for it: its call
/// waits on a sign-in shown on its step, and the router keeps the Google credential, which the
/// app never sees. Google's Gmail MCP server writes drafts but cannot send, so a send goes
/// through the app's backend, which posts to Gmail's REST API through the router's proxy for
/// the same connection.
enum Gmail {
    /// Sends a plain-text email from the person's Gmail, through the backend.
    static func send(to: String, subject: String, body: String) async throws -> String {
        var request = URLRequest(url: Assistant.backendURL.appending(path: "gmail/send"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(
            ["user_id": Assistant.user.id, "to": to, "subject": subject, "body": body])
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let said = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            throw ToolFailure(said.isEmpty ? "The backend could not reach Gmail." : said)
        }
        return try JSONDecoder().decode(Sent.self, from: data).id
    }

    private struct Sent: Decodable {
        let id: String
    }
}
