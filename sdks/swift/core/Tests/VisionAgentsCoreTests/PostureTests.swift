import CryptoKit
import Foundation
import Testing

@testable import VisionAgentsCore

/// What a phone holding this SDK can reach: a conversation, and nothing that is the app's
/// backend's to do.
struct PostureTests {
    static let repo = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("../../../../..").standardized
    static let generated = repo.appendingPathComponent(
        "sdks/swift/core/Sources/VisionAgentsCore/Generated/Types+Operations.swift")
    static let spec = repo.appendingPathComponent("acceleration/api/openapi.yaml")

    /// The generated client is committed, so a stale or hand-edited one would ship a method
    /// for an operation the router refuses a device, or one it no longer marks for devices.
    @Test func theGeneratedClientReachesOnlyWhatADeviceMay() throws {
        let generated = try String(contentsOf: Self.generated, encoding: .utf8)
        let operations = generated.matches(of: /static let id: Swift.String = "(\w+)"/).map {
            String($0.output.1)
        }
        let accessible = try Self.clientAccessible()

        #expect(!operations.isEmpty)
        #expect(accessible.contains("createSession") && !accessible.contains("createCallToken"))
        #expect(Set(operations).subtracting(accessible).isEmpty, "server-side only: \(Set(operations).subtracting(accessible).sorted())")
    }

    /// The operation ids the spec marks `x-client-accessible: true`, read a method block at a
    /// time so a mark on one operation is never credited to its neighbour.
    static func clientAccessible() throws -> Set<String> {
        let spec = try String(contentsOf: Self.spec, encoding: .utf8)
        var accessible: Set<String> = []
        var operation = ""
        var marked = false
        let methods = ["get", "put", "post", "patch", "delete", "head", "options"].map { "    \($0):" }
        for line in spec.split(separator: "\n", omittingEmptySubsequences: false) {
            let indent = line.prefix(while: { $0 == " " }).count
            if methods.contains(String(line)) || (indent <= 2 && !line.isEmpty) {
                if marked && !operation.isEmpty { accessible.insert(operation) }
                operation = ""
                marked = false
            } else if line.hasPrefix("      operationId: ") {
                operation = String(line.dropFirst("      operationId: ".count))
            } else if line == "      x-client-accessible: true" {
                marked = true
            }
        }
        if marked && !operation.isEmpty { accessible.insert(operation) }
        return accessible
    }
}

/// The same promise against a router that verifies credentials, which needs one running in
/// `api_key` mode and a key minted for it with `router keys create`:
///
///     INTEGRATION_ROUTER_URL=http://localhost:8099 INTEGRATION_ROUTER_KEY=vak_live_… INTEGRATION_ROUTER_SECRET=vas_live_… swift test
struct KeyedRouter {
    static let environment = ProcessInfo.processInfo.environment
    static let url = URL(string: environment["INTEGRATION_ROUTER_URL"] ?? "http://localhost:8099")!
    static let key = environment["INTEGRATION_ROUTER_KEY"] ?? ""
    static let secret = environment["INTEGRATION_ROUTER_SECRET"] ?? ""

    static var available: Bool { !key.isEmpty && !secret.isEmpty }

    /// A device's backend: the key, and a token naming one person.
    static func device() -> Backend {
        let backend = Backend(apiKey: key, url: url)
        let id = "swift-device-\(UUID().uuidString)"
        backend.setUser(User(id: id), token: sign(["user_id": id]))
        return backend
    }

    /// An HS256 token, which is what the app's own backend would have minted.
    static func sign(_ claims: [String: Any]) -> String {
        let header = ["alg": "HS256", "typ": "JWT"]
        var claims = claims
        claims["exp"] = Int(Date().timeIntervalSince1970) + 3600
        let encoded = [header as [String: Any], claims].map {
            base64URL(try! JSONSerialization.data(withJSONObject: $0))
        }.joined(separator: ".")
        let signature = HMAC<SHA256>.authenticationCode(
            for: Data(encoded.utf8), using: SymmetricKey(data: Data(secret.utf8)))
        return encoded + "." + base64URL(Data(signature))
    }

    static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    /// Sends what the SDK sends for this device, to a path the generated client has no method for.
    static func request(_ backend: Backend, _ method: String, _ path: String) async throws -> (Int, String) {
        var components = URLComponents(url: url.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        components.queryItems = backend.credentialQuery
        var request = URLRequest(url: components.url!)
        request.httpMethod = method
        for (name, value) in try await backend.headers() {
            request.setValue(value, forHTTPHeaderField: name)
        }
        if method == "POST" {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data("{}".utf8)
        }
        let (body, response) = try await URLSession.shared.data(for: request)
        let code = (try? JSONSerialization.jsonObject(with: body) as? [String: Any])
            .flatMap { $0["error"] as? [String: Any] }?["code"] as? String
        return ((response as! HTTPURLResponse).statusCode, code ?? "")
    }
}

@Suite(.enabled(if: KeyedRouter.available), .serialized)
struct KeyedRouterPostureTests {
    @Test(arguments: [
        ("POST", "/v1/agents/calls/default:anything/token"),
        ("POST", "/v1/agents/chat-token"),
        ("POST", "/v1/agents/guests/claim"),
        ("GET", "/v1/agents/configs"),
    ])
    func aDeviceMayNotMintTokensOrConfigureAgents(method: String, path: String) async throws {
        let (status, code) = try await KeyedRouter.request(KeyedRouter.device(), method, path)

        #expect(status == 403, "\(method) \(path) answered \(status)")
        #expect(code == "server_side_only")
    }

    @Test func aDeviceMayStillListItsOwnConversations() async throws {
        let (status, code) = try await KeyedRouter.request(KeyedRouter.device(), "POST", "/v1/agents/sessions/query")

        #expect(status == 200, "it answered \(status) \(code)")
    }
}
