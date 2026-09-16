import Foundation
import OpenAPIRuntime

/// Looking something up, configured once.
///
/// Search is the one routed modality a device may reach: a question and its answer are one
/// round trip, and the answer is for whoever asked rather than for the app.
///
///     let router = Router(url: url, customerID: "acme", config: "healthcare")
///     let answer = try await router.search("what changed in the pricing page")
///
/// Transcription, a voice and a model are not here. Those run over the per-modality socket,
/// which the router refuses to a device, so a pipeline of your own belongs to a backend and
/// has the Go or the Python SDK. Everything the named config holds is a default that a
/// per-call option overrides.
public struct Router: Sendable {
    public let backend: Backend

    /// A stored router config, by the name it was stored under or by its id. Without one,
    /// every call says what it wants for itself.
    public let config: String

    /// Cost labels carried onto everything routed here, on top of the config's own.
    public var tags: [String: String]

    public init(
        url: URL,
        customerID: String,
        config: String = "",
        tags: [String: String] = [:],
        urlSession: URLSession = .shared
    ) {
        backend = Backend(url: url, customerID: customerID, urlSession: urlSession)
        self.config = config
        self.tags = tags
    }

    public init(backend: Backend, config: String = "", tags: [String: String] = [:]) {
        self.backend = backend
        self.config = config
        self.tags = tags
    }

    /// Answers one question out of what is true now.
    public func search(
        _ query: String,
        options: SearchOptions = SearchOptions()
    ) async throws -> SearchAnswer {
        let body = Components.Schemas.SearchRequest(
            configId: named(),
            query: query,
            options: options.schema,
            tags: labels().map { .init(additionalProperties: $0) })

        let output = try await call { try await $0.search(body: .json(body)) }
        switch output {
        case .ok(let response):
            return SearchAnswer(try response.body.json)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// The config a request is made under, when there is one.
    func named() -> String? { config.isEmpty ? nil : config }

    /// The cost labels a request carries, when there are any.
    func labels() -> [String: String]? { tags.isEmpty ? nil : tags }

    /// Runs one request, reporting a transport failure as one and leaving cancellation alone.
    func call<T>(_ body: (Client) async throws -> T) async throws -> T {
        do {
            return try await body(backend.client())
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AgentsError {
            throw error
        } catch let error as ClientError {
            if error.underlyingError is CancellationError { throw CancellationError() }
            throw AgentsError.transport(error.underlyingError)
        } catch {
            throw AgentsError.transport(error)
        }
    }
}
