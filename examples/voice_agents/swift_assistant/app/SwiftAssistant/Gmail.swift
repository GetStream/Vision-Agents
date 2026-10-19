import AuthenticationServices
import Foundation
import Observation
import SwiftUI

/// The person's Gmail, as the agent reaches it: connected once, with Google's consent, and then
/// used in every conversation.
///
/// The router holds the Google credential and the app never sees it. The app's backend starts
/// the consent, since only a server may, and hands back the router's launch page; the app opens
/// it in a web authentication sheet with the handoff token in its fragment, and the router sends
/// the sheet back to the app's own scheme once Google has answered. A conversation already open
/// picks the account up on its next Gmail call.
@MainActor
@Observable
final class GmailAccount {
    /// Whether Gmail is connected, or nil before the backend has said.
    private(set) var isConnected: Bool?

    /// Whether a connect or a disconnect is under way.
    private(set) var isBusy = false

    /// Why the last attempt failed, or nil.
    var failure: String?

    /// The scheme the router sends the sign-in sheet back to, which the backend names as the
    /// consent's return URL. The router takes an app's own scheme only in reverse domain form.
    private static let callbackScheme = "io.getstream.visionagents.swiftassistant"

    /// Asks the backend whether Gmail is connected.
    func refresh() async {
        do {
            let answer = try await Self.call("GET", "gmail", query: ["user_id": Assistant.user.id])
            isConnected = try JSONDecoder().decode(Status.self, from: answer).connected
        } catch is CancellationError {
        } catch {
            failure = error.localizedDescription
        }
    }

    /// Connects Gmail: Google's consent, in a sheet over the app.
    func connect(with session: WebAuthenticationSession) async {
        isBusy = true
        failure = nil
        defer { isBusy = false }
        do {
            let consent = try JSONDecoder().decode(
                Consent.self,
                from: try await Self.call("POST", "gmail/connect", body: ["user_id": Assistant.user.id]))
            // The fragment never reaches a server: the launch page reads it and hands it off.
            guard var launch = URLComponents(string: consent.launchURL) else {
                throw ToolFailure("The router's launch page is not a URL.")
            }
            launch.fragment = "handoff_token=" + consent.handoffToken
            guard let url = launch.url else { throw ToolFailure("The router's launch page is not a URL.") }
            let returned = try await session.authenticate(
                using: url, callbackURLScheme: Self.callbackScheme)
            let status = URLComponents(url: returned, resolvingAgainstBaseURL: false)?
                .queryItems?.first { $0.name == "status" }?.value
            if status != "connected" {
                failure = "Gmail was not connected (\(status ?? "no answer"))."
            }
        } catch ASWebAuthenticationSessionError.canceledLogin {
        } catch is CancellationError {
        } catch {
            failure = error.localizedDescription
        }
        await refresh()
    }

    /// Disconnects Gmail, which revokes what Google granted.
    func disconnect() async {
        isBusy = true
        failure = nil
        defer { isBusy = false }
        do {
            _ = try await Self.call("DELETE", "gmail", query: ["user_id": Assistant.user.id])
        } catch is CancellationError {
        } catch {
            failure = error.localizedDescription
        }
        await refresh()
    }

    /// Sends a plain-text email from the person's Gmail, through the backend.
    nonisolated static func send(to: String, subject: String, body: String) async throws -> String {
        let answer = try await call(
            "POST", "gmail/send",
            body: ["user_id": Assistant.user.id, "to": to, "subject": subject, "body": body])
        return try JSONDecoder().decode(Sent.self, from: answer).id
    }

    /// One call to the app's backend, and what it answered.
    private nonisolated static func call(
        _ method: String, _ path: String, query: [String: String] = [:], body: [String: String]? = nil
    ) async throws -> Data {
        var address = URLComponents(
            url: Assistant.backendURL.appending(path: path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty {
            address.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) }
        }
        var request = URLRequest(url: address.url!)
        request.httpMethod = method
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONEncoder().encode(body)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let said = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            throw ToolFailure(said.isEmpty ? "The backend could not reach Gmail." : said)
        }
        return data
    }

    private struct Status: Decodable {
        let connected: Bool
    }

    private struct Consent: Decodable {
        let launchURL: String
        let handoffToken: String

        enum CodingKeys: String, CodingKey {
            case launchURL = "launch_url"
            case handoffToken = "handoff_token"
        }
    }

    private struct Sent: Decodable {
        let id: String
    }
}

/// The Gmail button at the top right of a conversation: connects Gmail, and once it is, says
/// so and offers to disconnect it.
struct GmailButton: View {
    let gmail: GmailAccount

    @Environment(\.webAuthenticationSession) private var webAuthenticationSession

    var body: some View {
        Group {
            if gmail.isBusy {
                ProgressView()
            } else if gmail.isConnected == true {
                Menu {
                    Text("Gmail is connected")
                    Button("Disconnect Gmail", systemImage: "xmark.circle", role: .destructive) {
                        Task { await gmail.disconnect() }
                    }
                } label: {
                    Image(systemName: "envelope.badge.shield.half.filled")
                        .foregroundStyle(.tint)
                }
                .accessibilityLabel("Gmail, connected")
            } else {
                Button {
                    Task { await gmail.connect(with: webAuthenticationSession) }
                } label: {
                    Image(systemName: "envelope")
                }
                .accessibilityLabel("Connect Gmail")
            }
        }
        .task { await gmail.refresh() }
        .alert(
            "Gmail",
            isPresented: Binding(get: { gmail.failure != nil }, set: { if !$0 { gmail.failure = nil } }),
            actions: { Button("OK") { gmail.failure = nil } },
            message: { Text(gmail.failure ?? "") })
    }
}
