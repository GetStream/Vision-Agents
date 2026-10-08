import Foundation
import HTTPTypes
import OpenAPIRuntime
import OpenAPIURLSession
import os

/// Somebody a device is acting for, as Stream knows them.
///
/// Only the id reaches the router; the token is what proves it. The name and image are carried
/// so chat and video have something to show without a second lookup.
public struct User: Sendable, Hashable {
    public let id: String
    public var name: String
    public var image: String

    public init(id: String, name: String = "", image: String = "") {
        self.id = id
        self.name = name
        self.image = image
    }
}

/// Hands over a token for the user, and a fresh one when asked again.
///
/// A provider rather than a string, because a token expires and an hour-long conversation
/// should not end when it does. It is asked once, then again after a 401.
public typealias TokenProvider = @Sendable () async throws -> String

/// Where the router is and who is asking.
///
/// A phone holds no secret worth having, so this carries no API secret. A deployment is
/// reached by the app's API key and a token the app's own backend minted for the user, given
/// to `setUser`. A router running locally with nothing in front of it is reached by customer
/// id instead.
public struct Backend: Sendable {
    /// Stream's hosted router.
    public static let defaultURL = URL(string: "https://accelerate.gcp.stream-io-api.com")!

    /// The router's base URL, with no path.
    public let url: URL

    /// The app's public API key. Empty for a local router reached by customer id.
    public let apiKey: String

    /// Which tenant's agents, calls and configs these are, on a local router.
    public let customerID: String

    /// The session used for both requests and sockets. Sharing one means one connection pool
    /// and one set of timeouts.
    public let urlSession: URLSession

    let identity = Identity()

    public init(apiKey: String, url: URL = Backend.defaultURL, urlSession: URLSession = .shared) {
        self.url = url
        self.apiKey = apiKey
        customerID = ""
        self.urlSession = urlSession
    }

    /// A router running locally with nothing in front of it.
    public init(url: URL, customerID: String, urlSession: URLSession = .shared) {
        self.url = url
        apiKey = ""
        self.customerID = customerID
        self.urlSession = urlSession
    }

    /// Who this is acting for, or nil until `setUser`.
    public var user: User? { identity.user }

    /// Says who this device is acting for, and how to prove it.
    ///
    /// On a local router the token is not read, and the user id is what names the end user.
    public func setUser(_ user: User, token: @escaping TokenProvider) {
        identity.set(user, token: token)
    }

    /// Says who this device is acting for, with a token that will not be refreshed.
    public func setUser(_ user: User, token: String) {
        setUser(user, token: { token })
    }

    /// Forgets the user, which is what signing out is.
    public func clearUser() {
        identity.clear()
    }

    /// The headers every request and every socket handshake carries.
    ///
    /// `Stream-Auth-Type: jwt` says this caller is somebody's device rather than their
    /// backend. Saying so is what makes the router refuse the paths that configure an agent,
    /// and it is said even to a local router, which would otherwise assume a caller with no
    /// proxy in front of it is a backend. The key travels in the query instead, where both
    /// Stream's proxy and the router read it.
    func headers() async throws -> [String: String] {
        var headers = ["Stream-Auth-Type": "jwt"]
        if !apiKey.isEmpty {
            headers["Authorization"] = "Bearer \(try await identity.token())"
            return headers
        }
        guard !customerID.isEmpty else {
            throw AgentsError.configuration("pass an apiKey, or a customerID for a local router")
        }
        headers["X-Customer-Id"] = customerID
        if let user = identity.user {
            headers["X-Stream-User-Id"] = user.id
        }
        return headers
    }

    /// Drops the token held, so the next request asks the provider again. Reports whether
    /// there is a provider to ask, which is whether a retry could go any differently.
    func expireToken() -> Bool {
        !apiKey.isEmpty && identity.expire()
    }

    /// The query every request and socket carries: the API key, which is the public half of
    /// the credential. The token is never here, because a URL ends up in every log it passes.
    var credentialQuery: [URLQueryItem] {
        if !apiKey.isEmpty { return [URLQueryItem(name: "api_key", value: apiKey)] }
        return [URLQueryItem(name: "customer_id", value: customerID)]
    }

    /// The socket URL for a path under the router.
    public func socketURL(path: String, query: [String: String] = [:]) -> URL {
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        components.scheme = components.scheme == "https" ? "wss" : "ws"
        components.path = path
        components.queryItems =
            credentialQuery
            + query.sorted { $0.key < $1.key }.map { URLQueryItem(name: $0.key, value: $0.value) }
        return components.url!
    }
}

/// The user a backend acts for, and their token, shared by every copy of that backend.
///
/// A token is fetched once however many requests are waiting for it, and kept until a 401
/// says it expired.
final class Identity: Sendable {
    private struct State {
        var user: User?
        var provider: TokenProvider?
        var token: Task<String, any Error>?
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var user: User? { state.withLock { $0.user } }

    func set(_ user: User, token: @escaping TokenProvider) {
        state.withLock { $0 = State(user: user, provider: token) }
    }

    func clear() {
        state.withLock { $0 = State() }
    }

    func expire() -> Bool {
        state.withLock {
            $0.token = nil
            return $0.provider != nil
        }
    }

    func token() async throws -> String {
        let pending: Task<String, any Error>? = state.withLock {
            if let token = $0.token { return token }
            guard let provider = $0.provider else { return nil }
            let token = Task { try await provider() }
            $0.token = token
            return token
        }
        guard let pending else {
            throw AgentsError.configuration(
                "an api key needs a user token to go with it; call setUser first")
        }
        do {
            return try await pending.value
        } catch {
            // A token that could not be had is asked for again next time, not remembered.
            state.withLock { if $0.token == pending { $0.token = nil } }
            throw error
        }
    }
}

/// Puts the backend's credentials on every request the generated client makes, and asks for
/// a fresh token once when the router says the one it had expired.
struct CredentialMiddleware: ClientMiddleware {
    let backend: Backend

    func intercept(
        _ request: HTTPRequest,
        body: HTTPBody?,
        baseURL: URL,
        operationID: String,
        next: (HTTPRequest, HTTPBody?, URL) async throws -> (HTTPResponse, HTTPBody?)
    ) async throws -> (HTTPResponse, HTTPBody?) {
        let answer = try await next(try await credentialed(request), body, baseURL)
        guard answer.0.status == .unauthorized, backend.expireToken() else { return answer }
        return try await next(try await credentialed(request), body, baseURL)
    }

    private func credentialed(_ request: HTTPRequest) async throws -> HTTPRequest {
        var request = request
        for (name, value) in try await backend.headers() {
            request.headerFields[HTTPField.Name(name)!] = value
        }
        var components = URLComponents(string: request.path ?? "/") ?? URLComponents()
        components.queryItems = (components.queryItems ?? []) + backend.credentialQuery
        request.path = components.string
        return request
    }
}

/// Throws every answer that is not a success as the `HTTPFailure` it reports, before the
/// generated client reads it.
///
/// Every status goes through here, documented or not, because the generated outputs carry no
/// headers to read `X-Request-Id` from, and because a body that is not the envelope under a
/// documented status would otherwise fail to decode and read as a transport failure.
struct FailureMiddleware: ClientMiddleware {
    func intercept(
        _ request: HTTPRequest,
        body: HTTPBody?,
        baseURL: URL,
        operationID: String,
        next: (HTTPRequest, HTTPBody?, URL) async throws -> (HTTPResponse, HTTPBody?)
    ) async throws -> (HTTPResponse, HTTPBody?) {
        let (response, answer) = try await next(request, body, baseURL)
        guard response.status.kind != .successful else { return (response, answer) }
        throw AgentsError.http(
            HTTPFailure(
                status: response.status.code,
                requestID: response.headerFields[HTTPField.Name("X-Request-Id")!] ?? "",
                body: try await prefix(of: answer)))
    }

    private func prefix(of body: HTTPBody?) async throws -> Data {
        var data = Data()
        guard let body else { return data }
        do {
            for try await chunk in body {
                data.append(contentsOf: chunk.prefix(maximumFailureBody - data.count))
                if data.count >= maximumFailureBody { break }
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // A body that breaks off leaves the status as the failure, with what did arrive.
        }
        return data
    }
}

/// Reads the timestamps the router actually sends.
///
/// Go's `time.Time` marshals to RFC 3339 with however many fractional digits the value needs
/// and none when it needs none, so one timestamp is `...:50.89279Z` and the next is `...:50Z`.
/// The generator's default reads only the second, which is how a client ends up refusing a
/// perfectly good agent config.
struct RouterDates: DateTranscoder {
    private let iso8601 = Date.ISO8601FormatStyle(includingFractionalSeconds: true)

    func encode(_ date: Date) throws -> String {
        iso8601.format(date)
    }

    func decode(_ string: String) throws -> Date {
        guard let date = try? iso8601.parse(string) else {
            throw AgentsError.unreadable("\(string) is not a timestamp")
        }
        return date
    }
}

extension Backend {
    /// The generated client, already carrying the credentials.
    ///
    /// The transport is a parameter so a test can answer requests itself without a server and
    /// without pretending to be `URLSession`.
    func client(transport: (any ClientTransport)? = nil) -> Client {
        Client(
            serverURL: url,
            configuration: .init(dateTranscoder: RouterDates()),
            transport: transport ?? URLSessionTransport(
                configuration: .init(session: urlSession)),
            // Outermost, so a 401 is a failure only once the credentials have had their retry.
            middlewares: [FailureMiddleware(), CredentialMiddleware(backend: self)])
    }
}
