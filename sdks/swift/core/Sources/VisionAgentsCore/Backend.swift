import Foundation
import HTTPTypes
import OpenAPIRuntime
import OpenAPIURLSession

/// Where the router is and who is asking.
///
/// A phone holds no secret worth having, so this carries no API secret. In the deployment
/// mode the demo uses the customer id is the whole of it; in front of a real deployment the
/// proxy verifies a token and this is what it verifies on behalf of.
public struct Backend: Sendable {
    /// The router's base URL, with no path.
    public let url: URL

    /// Which tenant's agents, calls and configs these are.
    public let customerID: String

    /// The session used for both requests and sockets. Sharing one means one connection pool
    /// and one set of timeouts.
    public let urlSession: URLSession

    public init(url: URL, customerID: String, urlSession: URLSession = .shared) {
        self.url = url
        self.customerID = customerID
        self.urlSession = urlSession
    }

    /// The headers every request and every socket handshake carries.
    ///
    /// `Stream-Auth-Type: jwt` says this caller is somebody's device rather than their
    /// backend. Saying so is what makes the router refuse the paths that configure an agent,
    /// and it is said even in the demo's open mode, where the router would otherwise assume a
    /// caller with no proxy in front of it is a backend.
    public var headers: [String: String] {
        ["X-Customer-Id": customerID, "Stream-Auth-Type": "jwt"]
    }

    /// The socket URL for a path under the router.
    ///
    /// The customer id is repeated in the query because a browser cannot set headers on a
    /// socket handshake. `URLSessionWebSocketTask` can, and does, but the query costs nothing
    /// and keeps one URL shape across the SDKs.
    public func socketURL(path: String, query: [String: String] = [:]) -> URL {
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        components.scheme = components.scheme == "https" ? "wss" : "ws"
        components.path = path
        components.queryItems =
            [URLQueryItem(name: "customer_id", value: customerID)]
            + query.sorted { $0.key < $1.key }.map { URLQueryItem(name: $0.key, value: $0.value) }
        return components.url!
    }
}

/// Puts the backend's headers on every request the generated client makes.
struct HeaderMiddleware: ClientMiddleware {
    let headers: [String: String]

    func intercept(
        _ request: HTTPRequest,
        body: HTTPBody?,
        baseURL: URL,
        operationID: String,
        next: (HTTPRequest, HTTPBody?, URL) async throws -> (HTTPResponse, HTTPBody?)
    ) async throws -> (HTTPResponse, HTTPBody?) {
        var request = request
        for (name, value) in headers {
            request.headerFields[HTTPField.Name(name)!] = value
        }
        return try await next(request, body, baseURL)
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
    /// The generated client, already carrying the headers.
    ///
    /// The transport is a parameter so a test can answer requests itself without a server and
    /// without pretending to be `URLSession`.
    func client(transport: (any ClientTransport)? = nil) -> Client {
        Client(
            serverURL: url,
            configuration: .init(dateTranscoder: RouterDates()),
            transport: transport ?? URLSessionTransport(
                configuration: .init(session: urlSession)),
            middlewares: [HeaderMiddleware(headers: headers)])
    }
}
