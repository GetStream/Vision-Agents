import Foundation
import Testing

@testable import StreamAgents

/// The exact body `POST /v1/agents/sessions/query` is sent, and the pin a session update carries.
@Suite struct SessionQueryTests {
    private static let empty = SessionServer.Answer(
        status: 200, headers: ["Content-Type": "application/json"],
        body: #"{"items":[],"has_more":false}"#)

    /// Runs `ask` against a router that lists nothing and hands back the request it was sent.
    private func sent(
        _ ask: (StreamAgents) async throws -> Void
    ) async throws -> SessionServer.Request {
        let server = try SessionServer(answer: Self.empty)
        defer { server.listener.cancel() }
        try await ask(StreamAgents(url: try await server.url(), customerID: "acme"))
        let request = try await server.request()
        #expect(request.method == "POST")
        #expect(request.path == "/v1/agents/sessions/query")
        return request
    }

    private func project(_ id: String) -> RawJSON {
        .dictionary(["project_id": .dictionary(["$eq": .string(id)])])
    }

    @Test func aQueryWithNoFilterSendsOnlyTheAgent() async throws {
        let request = try await sent { _ = try await $0.agent("myagent").sessions.query() }

        #expect(request.body == ["filter": .dictionary(["agent": .string("myagent")])])
    }

    @Test func aQueryOfEveryAgentWithNoFilterSendsNothing() async throws {
        let request = try await sent { _ = try await $0.sessions.query() }

        #expect(request.body == [:])
    }

    @Test func aProjectFilterSitsBesideTheAgent() async throws {
        let request = try await sent {
            _ = try await $0.agent("myagent").sessions.query(filter: .projectID("a"))
        }

        #expect(
            request.body == [
                "filter": .dictionary([
                    "agent": .string("myagent"), "project_id": .dictionary(["$eq": .string("a")]),
                ])
            ])
    }

    @Test func anOrOfTwoProjectsIsNotWrappedToNameTheAgent() async throws {
        let request = try await sent {
            _ = try await $0.agent("myagent").sessions.query(
                filter: .or([.projectID("a"), .projectID("b")]))
        }

        #expect(
            request.body == [
                "filter": .dictionary([
                    "agent": .string("myagent"), "$or": .array([project("a"), project("b")]),
                ])
            ])
    }

    @Test func aNestedAndOfOrsIsSentAsWritten() async throws {
        let request = try await sent {
            _ = try await $0.agent("myagent").sessions.query(
                filter: .and([
                    .or([.projectID("a"), .projectID("b")]),
                    .or([.projectIDIn(["c", "d"]), .and([.projectID("e"), .agent("other")])]),
                ]))
        }

        #expect(
            request.body == [
                "filter": .dictionary([
                    "agent": .string("myagent"),
                    "$and": .array([
                        .dictionary(["$or": .array([project("a"), project("b")])]),
                        .dictionary([
                            "$or": .array([
                                .dictionary([
                                    "project_id": .dictionary(["$in": .array([.string("c"), .string("d")])])
                                ]),
                                .dictionary([
                                    "$and": .array([project("e"), .dictionary(["agent": .string("other")])])
                                ]),
                            ])
                        ]),
                    ]),
                ])
            ])
    }

    @Test func anyOfSeveralProjectsIsAnIn() async throws {
        let request = try await sent {
            _ = try await $0.agent("myagent").sessions.query(filter: .projectIDIn(["a", "b"]))
        }

        #expect(
            request.body == [
                "filter": .dictionary([
                    "agent": .string("myagent"),
                    "project_id": .dictionary(["$in": .array([.string("a"), .string("b")])]),
                ])
            ])
    }

    @Test func theAgentOfTheHandleWinsOverOneAtTheTopOfTheFilter() async throws {
        let request = try await sent {
            _ = try await $0.agent("myagent").sessions.query(filter: .agent("other"))
        }

        #expect(request.body == ["filter": .dictionary(["agent": .string("myagent")])])
    }

    @Test func withNoHandleTheFilterNamesTheAgent() async throws {
        let request = try await sent { _ = try await $0.sessions.query(filter: .agent("other")) }

        #expect(request.body == ["filter": .dictionary(["agent": .string("other")])])
    }

    @Test func aSearchPutsItsTextAtTheTopBesideTheProjectAndSendsNoSort() async throws {
        let request = try await sent {
            _ = try await $0.agent("myagent").sessions.search("refund", filter: .projectID("a"), limit: 10)
        }

        #expect(
            request.body == [
                "filter": .dictionary([
                    "agent": .string("myagent"), "project_id": .dictionary(["$eq": .string("a")]),
                    "text": .dictionary(["$q": .string("refund")]),
                ]),
                "limit": .number(10),
            ])
    }

    @Test func aSearchAmongAnOrKeepsItsTextOutOfTheOr() async throws {
        let request = try await sent {
            _ = try await $0.sessions.search("refund", filter: .or([.projectID("a"), .projectID("b")]))
        }

        #expect(
            request.body == [
                "filter": .dictionary([
                    "$or": .array([project("a"), project("b")]),
                    "text": .dictionary(["$q": .string("refund")]),
                ])
            ])
    }

    @Test(arguments: [
        ([SessionSort.lastMessageAt()], [("last_message_at", -1)]),
        ([.lastMessageAt(ascending: true)], [("last_message_at", 1)]),
        ([.pinned], [("pinned", -1)]),
        ([.pinned, .lastMessageAt()], [("pinned", -1), ("last_message_at", -1)]),
        ([.pinned, .lastMessageAt(ascending: true)], [("pinned", -1), ("last_message_at", 1)]),
        ([.relevance], [("relevance", -1)]),
    ] as [([SessionSort], [(String, Double)])])
    func eachSortIsSentAsFieldAndDirection(sort: [SessionSort], wire: [(String, Double)]) async throws {
        let request = try await sent { _ = try await $0.sessions.query(sort: sort) }

        let expected = wire.map { RawJSON.dictionary(["field": .string($0.0), "direction": .number($0.1)]) }
        #expect(request.body == ["sort": .array(expected)])
    }

    @Test func theNextPageSendsTheCursorWithTheSameFilterAndSort() async throws {
        let request = try await sent {
            _ = try await $0.agent("myagent").sessions.query(
                filter: .projectID("a"), sort: [.pinned, .lastMessageAt()], limit: 20, cursor: "c2")
        }

        #expect(
            request.body == [
                "filter": .dictionary([
                    "agent": .string("myagent"), "project_id": .dictionary(["$eq": .string("a")]),
                ]),
                "sort": .array([
                    .dictionary(["field": .string("pinned"), "direction": .number(-1)]),
                    .dictionary(["field": .string("last_message_at"), "direction": .number(-1)]),
                ]),
                "limit": .number(20), "cursor": .string("c2"),
            ])
    }

    @Test func aPageReadsWhenEachSessionWasPinnedAndWhereTheNextStarts() async throws {
        let server = try SessionServer(
            answer: .init(
                status: 200, headers: ["Content-Type": "application/json"],
                body: #"""
                    {"items":[
                      {"id":"s1","agent_id":"a1","call_id":"","call_type":"default","created_at":"2026-10-02T17:00:00Z",
                       "modality":"text","state":"ended","user_id":"jlahey","pinned_at":"2026-11-11T09:30:00Z"},
                      {"id":"s2","agent_id":"a1","call_id":"","call_type":"default","created_at":"2026-10-02T17:00:00Z",
                       "modality":"text","state":"live","user_id":"jlahey"}
                    ],"has_more":true,"next_cursor":"c2"}
                    """#))
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

        let page = try await agents.sessions.query(sort: [.pinned])

        #expect(page.items.map(\.id) == ["s1", "s2"])
        #expect(page.items[0].pinnedAt == Date(timeIntervalSince1970: 1_794_389_400))
        #expect(page.items[1].pinnedAt == nil)
        #expect(page.hasMore)
        #expect(page.nextCursor == "c2")
    }

    @Test func aPinByIDSendsPinnedAndReadsWhenItWasPinned() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

        let session = try await agents.agent("myagent").sessions.pin("s1")

        #expect(session.pinnedAt == Date(timeIntervalSince1970: 1_794_389_400))
        let request = try await server.request()
        #expect(request.method == "PATCH")
        #expect(request.path == "/v1/agents/sessions/s1")
        #expect(request.body == ["pinned": .bool(true)])
    }

    @Test func anUnpinByIDSendsPinnedFalse() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

        let session = try await agents.sessions.unpin("s1")

        #expect(session.pinnedAt == nil)
        let request = try await server.request()
        #expect(request.method == "PATCH")
        #expect(request.path == "/v1/agents/sessions/s1")
        #expect(request.body == ["pinned": .bool(false)])
    }

    @Test func anUpdateCanPinWhileItRenames() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let agents = StreamAgents(url: try await server.url(), customerID: "acme")

        _ = try await agents.sessions.update("s1", title: "Keep", pinned: true)

        #expect(try await server.request().body == ["title": .string("Keep"), "pinned": .bool(true)])
    }

    @Test @MainActor func aHeldSessionPinsItselfAndKeepsWhatCameBack() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let held = try await held(at: server.url())

        try await held.pin()

        #expect(held.session.pinnedAt == Date(timeIntervalSince1970: 1_794_389_400))
        let request = try await server.request()
        #expect(request.method == "PATCH")
        #expect(request.path == "/v1/agents/sessions/s1")
        #expect(request.body == ["pinned": .bool(true)])
    }

    @Test @MainActor func aHeldSessionUnpinsItself() async throws {
        let server = try SessionServer()
        defer { server.listener.cancel() }
        let held = try await held(at: server.url())

        try await held.unpin()

        #expect(held.session.pinnedAt == nil)
        #expect(try await server.request().body == ["pinned": .bool(false)])
    }

    @MainActor private func held(at url: URL) throws -> AgentSession {
        let payload = try JSONDecoder.default.decode(
            SessionPayload.self,
            from: Data(
                #"{"id":"s1","agent_id":"a1","call_id":"","call_type":"default","created_at":"2026-10-02T17:00:00Z","modality":"text","state":"live","user_id":"jlahey"}"#
                    .utf8))
        return AgentSession(backend: Backend(url: url, customerID: "acme"), session: Session(payload), tools: [])
    }
}
