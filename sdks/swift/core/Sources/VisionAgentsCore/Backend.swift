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

/// The one identity Stream's own SDKs connect with: the app's key, the user `setUser` named
/// and their token. It is the identity the router is reached with, so chat and video need
/// nothing of their own.
public struct StreamCredentials: Sendable, Hashable {
    public let apiKey: String
    public let user: User
    public let token: String
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

    /// The app's public API key. Beside a customer id it is Stream's alone: the router is still
    /// reached by customer id, and the key is what chat and video connect with.
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
    public init(url: URL, customerID: String, apiKey: String = "", urlSession: URLSession = .shared) {
        self.url = url
        self.apiKey = apiKey
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

    /// The key, user and token for Stream Chat, Stream Video or any other Stream product.
    ///
    /// `refresh` is for a token the caller was told has expired: the one held is dropped and
    /// the provider asked again, unless a fresh one is already on its way, so two clients
    /// refreshing at once fetch one token.
    public func streamCredentials(refresh: Bool = false) async throws -> StreamCredentials {
        guard !apiKey.isEmpty, let user = identity.user else {
            throw AgentsError.configuration(
                "Stream Chat and Video need the app's Stream key and a user: build VisionAgents "
                    + "with apiKey and call setUser")
        }
        if refresh {
            identity.refresh()
        }
        return StreamCredentials(apiKey: apiKey, user: user, token: try await identity.token())
    }

    /// Records a Stream client the app owns, so it is used rather than another being built.
    /// It is never disconnected here, and it stays across `setUser`.
    func give<Client: Sendable>(_ kind: String, _ client: Client) {
        identity.give(kind, client)
    }

    /// The client the app gave for `kind`, or the one built for the current key and user.
    ///
    /// Built once however many sessions ask, keyed before any token is asked for, so a second
    /// session mints nothing. One that failed to open is not kept.
    func shared<Client: Sendable>(
        _ kind: String,
        open: @escaping @Sendable (StreamCredentials) async throws -> Client,
        disconnect: @escaping @Sendable (Client) async -> Void
    ) async throws -> Client {
        if let given: Client = identity.given(kind) {
            return given
        }
        guard !apiKey.isEmpty, let user = identity.user else {
            throw AgentsError.configuration(
                "\(kind) connects to Stream rather than to the router, so it needs the app's "
                    + "Stream key and a user: build VisionAgents with apiKey and call setUser")
        }
        return try await identity.built(
            "\(kind):\(apiKey):\(user.id)",
            open: { try await open(try await streamCredentials()) },
            disconnect: disconnect)
    }

    /// Disconnects every Stream client built here. The ones the app gave are left alone.
    func disconnectStream() async {
        await identity.disconnectBuilt()
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
        if customerID.isEmpty {
            guard !apiKey.isEmpty else {
                throw AgentsError.configuration("pass an apiKey, or a customerID for a local router")
            }
            headers["Authorization"] = "Bearer \(try await identity.token())"
            return headers
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
        customerID.isEmpty && identity.expire()
    }

    /// The query every request and socket carries: the API key, which is the public half of
    /// the credential. The token is never here, because a URL ends up in every log it passes.
    var credentialQuery: [URLQueryItem] {
        if customerID.isEmpty { return [URLQueryItem(name: "api_key", value: apiKey)] }
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

/// The user a backend acts for, their token and the Stream clients connected as them, shared
/// by every copy of that backend.
///
/// A token is fetched once however many requests are waiting for it, and kept until a 401
/// says it expired.
final class Identity: Sendable {
    private struct State {
        var user: User?
        var provider: TokenProvider?
        var token: Task<String, any Error>?
        /// Whether `token` has finished, which is what makes it one a refresh may drop.
        var fetched = false
        /// The app's own clients by kind, kept whoever signs in.
        var given: [String: any Sendable] = [:]
        /// The clients built here, by kind, key and user.
        var built: [String: Built] = [:]
    }

    private struct Built {
        let opening: Task<any Sendable, any Error>
        let disconnect: @Sendable (any Sendable) async -> Void
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var user: User? { state.withLock { $0.user } }

    func set(_ user: User, token: @escaping TokenProvider) {
        state.withLock {
            $0 = State(user: user, provider: token, given: $0.given, built: $0.built)
        }
    }

    func clear() {
        state.withLock { $0 = State(given: $0.given, built: $0.built) }
    }

    func expire() -> Bool {
        state.withLock {
            $0.token = nil
            return $0.provider != nil
        }
    }

    /// Drops the token held, unless the one held is still being fetched.
    func refresh() {
        state.withLock {
            if $0.fetched {
                $0.token = nil
                $0.fetched = false
            }
        }
    }

    func token() async throws -> String {
        let pending: Task<String, any Error>? = state.withLock {
            if let token = $0.token { return token }
            guard let provider = $0.provider else { return nil }
            let token = Task { try await provider() }
            $0.token = token
            $0.fetched = false
            return token
        }
        guard let pending else {
            throw AgentsError.configuration(
                "an api key needs a user token to go with it; call setUser first")
        }
        do {
            let token = try await pending.value
            state.withLock { if $0.token == pending { $0.fetched = true } }
            return token
        } catch {
            // A token that could not be had is asked for again next time, not remembered.
            state.withLock { if $0.token == pending { $0.token = nil } }
            throw error
        }
    }

    func give<Client: Sendable>(_ kind: String, _ client: Client) {
        state.withLock { $0.given[kind] = client }
    }

    func given<Client: Sendable>(_ kind: String) -> Client? {
        state.withLock { $0.given[kind] as? Client }
    }

    func built<Client: Sendable>(
        _ key: String,
        open: @escaping @Sendable () async throws -> Client,
        disconnect: @escaping @Sendable (Client) async -> Void
    ) async throws -> Client {
        let opening = state.withLock {
            if let built = $0.built[key] { return built.opening }
            let opening = Task<any Sendable, any Error> { try await open() }
            $0.built[key] = Built(
                opening: opening,
                disconnect: { client in
                    if let client = client as? Client { await disconnect(client) }
                })
            return opening
        }
        do {
            guard let client = try await opening.value as? Client else {
                throw AgentsError.configuration("\(key) was opened as another kind of client")
            }
            return client
        } catch {
            state.withLock { if $0.built[key]?.opening == opening { $0.built[key] = nil } }
            throw error
        }
    }

    func disconnectBuilt() async {
        let built = state.withLock {
            let built = Array($0.built.values)
            $0.built = [:]
            return built
        }
        for entry in built {
            // One that never opened has nothing to disconnect, and its opener was told why.
            guard let client = try? await entry.opening.value else { continue }
            await entry.disconnect(client)
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
