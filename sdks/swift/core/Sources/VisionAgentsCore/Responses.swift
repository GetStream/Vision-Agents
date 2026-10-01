import Foundation

/// A session's turns, as the router wrote them down.
///
/// Read rather than watched: this is the same whether the conversation is still going or ended
/// last week. Deltas are not here, so a caller who wants to watch words arrive starts the
/// `AgentSession` and reads its `turns`.
public struct Responses: Sendable {
    public let sessionID: String

    private let backend: Backend

    init(backend: Backend, sessionID: String) {
        self.backend = backend
        self.sessionID = sessionID
    }

    /// Asks the agent something and names the turn it answers as.
    ///
    /// It returns as soon as the agent has started answering rather than when it has finished,
    /// so the result is a handle on an answer in progress: `items(responseID:)` reads what has
    /// been written down so far.
    public func create(_ text: String, images: [ImageSource] = []) async throws -> Response {
        let body = Components.Schemas.CreateResponseRequest(
            text: text, images: images.isEmpty ? nil : images.map(\.schema))
        let output = try await backend.call {
            try await $0.createResponse(path: .init(id: sessionID), body: .json(body))
        }
        switch output {
        case .accepted(let response):
            return Response(try response.body.json)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .forbidden(let response):
            throw AgentsError.http(status: 403, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .conflict(let response):
            throw AgentsError.http(status: 409, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// The turns so far, oldest first.
    ///
    /// A session that records nothing has none, and one rewound has none after the response
    /// it went back to.
    public func list(limit: Int? = nil, offset: Int? = nil) async throws -> [Response] {
        let output = try await backend.call {
            try await $0.listResponses(
                path: .init(id: sessionID), query: .init(limit: limit, offset: offset))
        }
        switch output {
        case .ok(let response):
            return try response.body.json.map(Response.init)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .forbidden(let response):
            throw AgentsError.http(status: 403, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// One page of what the agent did, turn by turn, in the order it happened.
    ///
    /// Every turn in the session, or only `responseID`'s. Nothing comes back for an incognito
    /// session, which has none to return.
    public func items(
        responseID: String? = nil, limit: Int? = nil, offset: Int? = nil
    ) async throws -> [ResponseItem] {
        let output = try await backend.call {
            try await $0.listResponseItems(
                path: .init(id: sessionID),
                query: .init(responseId: responseID, limit: limit, offset: offset))
        }
        switch output {
        case .ok(let response):
            return try response.body.json.map(ResponseItem.init)
        case .unauthorized(let response):
            throw AgentsError.http(status: 401, message: try response.body.json.error)
        case .forbidden(let response):
            throw AgentsError.http(status: 403, message: try response.body.json.error)
        case .notFound(let response):
            throw AgentsError.http(status: 404, message: try response.body.json.error)
        case .undocumented(let status, _):
            throw AgentsError.http(status: status, message: "unexpected")
        }
    }

    /// Goes back to a response and carries on from there, as though nothing after it was said.
    ///
    /// The model forgets the later turns and they drop out of `list` and `items`. A transcript
    /// an `AgentSession` is showing still has them, so read it back after this.
    public func rewind(to responseID: String) async throws {
        try await VisionAgents(backend: backend).rewind(sessionID: sessionID, to: responseID)
    }
}
