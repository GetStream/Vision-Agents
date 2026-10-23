import Foundation

/// A session's turns, as the router wrote them down.
///
/// Read rather than watched: this is the same whether the conversation is still going or ended
/// last week. Deltas are not here, so a caller who wants to watch words arrive starts the
/// `AgentSession` and reads its `turns`.
public struct Responses: Sendable {
    public let sessionID: String

    private let backend: Backend
    /// Shows what was asked in the transcript of the `AgentSession` asking it, since a
    /// conversation in writing is never heard back.
    private let asked: (@MainActor @Sendable (String) -> Void)?

    init(
        backend: Backend, sessionID: String,
        asked: (@MainActor @Sendable (String) -> Void)? = nil
    ) {
        self.backend = backend
        self.sessionID = sessionID
        self.asked = asked
    }

    /// Asks the agent something and names the turn it answers as.
    ///
    /// It returns as soon as the agent has started answering rather than when it has finished,
    /// so the result is a handle on an answer in progress: `items(responseID:)` reads what has
    /// been written down so far. Every question carries a fresh request id, so a retry of the
    /// same request starts no second turn. In a conversation kept in Stream Chat, `images` are
    /// data URIs, which the router shows on the person's message.
    ///
    /// `maxOutputTokens` caps this answer, reasoning included, and `reasoningEffort` is how long
    /// the model may think first, one of the efforts the session's model accepts. Both cover the
    /// replies that finish it after its tools; left nil, the session's own apply.
    public func create(
        _ text: String, images: [ImageSource] = [], maxOutputTokens: Int? = nil,
        reasoningEffort: String? = nil
    ) async throws -> Response {
        let body = Self.body(
            text, images: images, maxOutputTokens: maxOutputTokens, reasoningEffort: reasoningEffort)
        await asked?(text)
        return Response(
            try await backend.call { try await $0.createResponse(id: sessionID, createResponseRequest: body) })
    }

    static func body(
        _ text: String, images: [ImageSource], maxOutputTokens: Int? = nil,
        reasoningEffort: String? = nil
    ) -> CreateResponseRequest {
        .init(
            images: images.isEmpty ? nil : images.map(\.schema), maxOutputTokens: maxOutputTokens,
            reasoningEffort: reasoningEffort, requestId: UUID().uuidString, text: text)
    }

    /// One page of the turns so far, oldest first.
    ///
    /// A session that records nothing has none, and one rewound has none after the response
    /// it went back to. Pass the page's `nextCursor` as `cursor` for the next one.
    public func list(limit: Int? = nil, cursor: String? = nil) async throws -> Page<Response> {
        let page = try await backend.call {
            try await $0.listResponses(id: sessionID, cursor: cursor, limit: limit)
        }
        return Page(
            items: page.items.map(Response.init), hasMore: page.hasMore, nextCursor: page.nextCursor)
    }

    /// One page of what the agent did, turn by turn, in the order it happened.
    ///
    /// Every turn in the session, or only `responseID`'s. Nothing comes back for an incognito
    /// session, which has none to return.
    public func items(
        responseID: String? = nil, limit: Int? = nil, cursor: String? = nil
    ) async throws -> Page<ResponseItem> {
        let page = try await backend.call {
            try await $0.listResponseItems(
                id: sessionID, cursor: cursor, limit: limit, responseId: responseID)
        }
        return Page(
            items: page.items.map(ResponseItem.init), hasMore: page.hasMore, nextCursor: page.nextCursor)
    }

    /// Goes back to a response and carries on from there, as though nothing after it was said.
    ///
    /// The model forgets the later turns and they drop out of `list` and `items`. A transcript
    /// an `AgentSession` is showing still has them, so read it back after this.
    public func rewind(to responseID: String) async throws {
        try await VisionAgents(backend: backend).rewind(sessionID: sessionID, to: responseID)
    }
}
