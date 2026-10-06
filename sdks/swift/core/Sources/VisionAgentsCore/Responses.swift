import Foundation

/// A session's turns, as the router wrote them down.
///
/// Read rather than watched: this is the same whether the conversation is still going or ended
/// last week. Deltas are not here, so a caller who wants to watch words arrive starts the
/// `AgentSession` and reads its `turns`.
public struct Responses: Sendable {
    public let sessionID: String

    private let backend: Backend
    /// A conversation kept in Stream Chat, whose every question is a command the router can
    /// tell apart from a retry.
    private let kept: Bool
    /// Shows what was asked in the transcript of the `AgentSession` asking it, since a
    /// conversation in writing is never heard back.
    private let asked: (@MainActor @Sendable (String) -> Void)?

    init(
        backend: Backend, sessionID: String, kept: Bool = false,
        asked: (@MainActor @Sendable (String) -> Void)? = nil
    ) {
        self.backend = backend
        self.sessionID = sessionID
        self.kept = kept
        self.asked = asked
    }

    /// Asks the agent something and names the turn it answers as.
    ///
    /// It returns as soon as the agent has started answering rather than when it has finished,
    /// so the result is a handle on an answer in progress: `items(responseID:)` reads what has
    /// been written down so far. `commandID` names the question so a retry with the same id and
    /// text starts no second turn; a conversation kept in Stream Chat gets a fresh one when
    /// none is given.
    public func create(
        _ text: String, images: [ImageSource] = [], commandID: String? = nil
    ) async throws -> Response {
        // A command carries text only, so a question with images goes without one.
        let commandID = commandID ?? (kept && images.isEmpty ? UUID().uuidString : nil)
        let body = Components.Schemas.CreateResponseRequest(
            commandId: commandID, images: images.isEmpty ? nil : images.map(\.schema), text: text)
        await asked?(text)
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

    /// One page of the turns so far, oldest first.
    ///
    /// A session that records nothing has none, and one rewound has none after the response
    /// it went back to. Pass the page's `nextCursor` as `cursor` for the next one.
    public func list(limit: Int? = nil, cursor: String? = nil) async throws -> Page<Response> {
        let output = try await backend.call {
            try await $0.listResponses(
                path: .init(id: sessionID), query: .init(limit: limit, cursor: cursor))
        }
        switch output {
        case .ok(let response):
            let page = try response.body.json
            return Page(
                items: page.items.map(Response.init), hasMore: page.hasMore,
                nextCursor: page.nextCursor)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
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
        responseID: String? = nil, limit: Int? = nil, cursor: String? = nil
    ) async throws -> Page<ResponseItem> {
        let output = try await backend.call {
            try await $0.listResponseItems(
                path: .init(id: sessionID),
                query: .init(responseId: responseID, limit: limit, cursor: cursor))
        }
        switch output {
        case .ok(let response):
            let page = try response.body.json
            return Page(
                items: page.items.map(ResponseItem.init), hasMore: page.hasMore,
                nextCursor: page.nextCursor)
        case .badRequest(let response):
            throw AgentsError.http(status: 400, message: try response.body.json.error)
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
