import Foundation
import StreamCore

/// A running or finished conversation.
public struct Session: Sendable, Hashable, Identifiable {
    /// What the router holds this session by. This addresses the session and its socket, and
    /// it is not the call id.
    public let id: String
    /// The Stream call the agent joined, which is what a video SDK joins. Empty for a text
    /// session.
    public let callID: String
    public let callType: String
    /// Keys the transcript, and names the chat channel it is written to.
    public let agentID: String
    public let isText: Bool
    public let state: State
    /// How the user took part. It only moves up, from text or voice to video.
    public let modality: Modality
    /// The Stream Chat channel a text session is kept in. Empty for one kept nowhere.
    public let conversationID: String
    public let projectID: String
    /// The team the session's usage is counted under. Empty for none.
    public let team: String
    public let title: String
    public let description: String
    /// The caller's own labels.
    public let custom: [String: RawJSON]
    public let instructions: String
    public let llm: String
    public let createdAt: Date
    /// When the conversation was pinned. Nil for one that is not pinned.
    public let pinnedAt: Date?

    public enum State: String, Sendable {
        case live
        case ended
    }

    public enum Modality: String, Sendable {
        case text
        case voice
        case video
        /// A modality this SDK has never heard of.
        case unknown
    }

    init(_ schema: SessionPayload) {
        id = schema.id
        callID = schema.callId
        callType = schema.callType
        agentID = schema.agentId
        isText = schema.text ?? false
        state = State(rawValue: schema.state.rawValue) ?? .ended
        modality = Modality(rawValue: schema.modality.rawValue) ?? .unknown
        conversationID = schema.conversationId ?? ""
        projectID = schema.projectId ?? ""
        team = schema.team ?? ""
        title = schema.title ?? ""
        description = schema.description ?? ""
        custom = schema.custom ?? [:]
        instructions = schema.instructions ?? ""
        llm = schema.llm ?? ""
        createdAt = schema.createdAt
        pinnedAt = schema.pinnedAt
    }
}

/// Somebody talking to an agent before they have signed up.
public struct Guest: Sendable, Hashable {
    public let id: String
    /// A Stream user token with role guest, which chat and video connect with as well.
    public let token: String
    public let name: String
    public let custom: [String: RawJSON]
    /// When the token stops working, or nil when the router did not say.
    public let expiresAt: Date?

    public init(
        id: String, token: String, name: String = "", custom: [String: RawJSON] = [:],
        expiresAt: Date? = nil
    ) {
        self.id = id
        self.token = token
        self.name = name
        self.custom = custom
        self.expiresAt = expiresAt
    }

    init(_ schema: GuestUser) {
        id = schema.id
        token = schema.token
        name = schema.name ?? ""
        custom = schema.custom ?? [:]
        expiresAt = schema.expiresAt
    }
}

/// One page of a list, and where the next one starts.
public struct Page<Item: Sendable>: Sendable {
    public let items: [Item]
    /// Whether there is another page after this one.
    public let hasMore: Bool
    /// Pass as `cursor` for the next page. Nil on the last one.
    public let nextCursor: String?
}

extension Page: Equatable where Item: Equatable {}
extension Page: Hashable where Item: Hashable {}

/// Which of this caller's conversations to list, written as the router reads it.
///
/// Each case is one condition, and `and` and `or` put conditions together: each takes 1 to 10
/// filters, and they nest at most 3 deep.
///
/// ```swift
/// .or([.projectID("a"), .projectID("b")])
/// .and([.projectIDIn(["a", "b"]), .agent("support")])
/// ```
///
/// A device is always narrowed to its own user's sessions, so there is no filter by user, and
/// none by review score, which only a server-side caller may read. Words are searched for with
/// `Sessions.search`, which is why there is no text case either.
public indirect enum SessionFilter: Sendable, Hashable, Encodable {
    /// Only this project's: `{"project_id": {"$eq": id}}`.
    case projectID(String)
    /// Any of these projects', up to 50: `{"project_id": {"$in": ids}}`.
    case projectIDIn([String])
    /// Only the ones opened against the agent of this name: `{"agent": name}`. From
    /// `Agent.sessions` the agent is already named, and naming another at the top of the
    /// filter is overridden.
    case agent(String)
    /// Every one of these holds: `{"$and": [...]}`.
    case and([SessionFilter])
    /// At least one of these holds: `{"$or": [...]}`.
    case or([SessionFilter])

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: WireKey.self)
        try encode(into: &container)
    }

    /// Writes this condition's one key, so the top of a query can set its own keys beside it.
    func encode(into container: inout KeyedEncodingContainer<WireKey>) throws {
        switch self {
        case .projectID(let id):
            var match = container.nestedContainer(keyedBy: WireKey.self, forKey: "project_id")
            try match.encode(id, forKey: "$eq")
        case .projectIDIn(let ids):
            var match = container.nestedContainer(keyedBy: WireKey.self, forKey: "project_id")
            try match.encode(ids, forKey: "$in")
        case .agent(let name):
            try container.encode(name, forKey: "agent")
        case .and(let filters):
            try container.encode(filters, forKey: "$and")
        case .or(let filters):
            try container.encode(filters, forKey: "$or")
        }
    }
}

/// One step of the order a query lists conversations in.
///
/// Leaving the sort out is `[.lastMessageAt()]`. `[.pinned]` or `[.pinned, .lastMessageAt()]`
/// lists the pinned ones first. A device cannot sort by review score.
public struct SessionSort: Sendable, Hashable, Encodable {
    let field: String
    let direction: Int

    /// By when the agent last answered, or when the session began if it has not: most recent
    /// first, or oldest first when `ascending`.
    public static func lastMessageAt(ascending: Bool = false) -> SessionSort {
        SessionSort(field: "last_message_at", direction: ascending ? 1 : -1)
    }

    /// The pinned conversations first. It comes first in a sort, and `lastMessageAt` may follow.
    public static let pinned = SessionSort(field: "pinned", direction: -1)

    /// Best match first. It only sorts a text search, which `Sessions.search` already sorts so.
    public static let relevance = SessionSort(field: "relevance", direction: -1)
}

/// The body of `POST /v1/agents/sessions/query`.
///
/// Hand-written because the generated `SessionFilterPayload` cannot hold it: the generator
/// drops a string-or-object field, which is every field condition there is. The caller's
/// filter is one key at the top; the agent and the search text are set beside it rather than
/// wrapped around it in an `$and`, since the router only reads `text` at the top.
struct SessionQueryBody: Encodable, Sendable {
    var filter: SessionFilter?
    /// The agent an `Agent.sessions` query is narrowed to, which wins over one in `filter`.
    var agent: String?
    var text: String?
    var sort: [SessionSort]?
    var limit: Int?
    var cursor: String?

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: WireKey.self)
        if filter != nil || agent != nil || text != nil {
            var top = container.nestedContainer(keyedBy: WireKey.self, forKey: "filter")
            if let filter, !(agent != nil && filter.isAgent) {
                try filter.encode(into: &top)
            }
            try top.encodeIfPresent(agent, forKey: "agent")
            if let text {
                var match = top.nestedContainer(keyedBy: WireKey.self, forKey: "text")
                try match.encode(text, forKey: "$q")
            }
        }
        if let sort, !sort.isEmpty { try container.encode(sort, forKey: "sort") }
        try container.encodeIfPresent(limit, forKey: "limit")
        try container.encodeIfPresent(cursor, forKey: "cursor")
    }
}

extension SessionFilter {
    fileprivate var isAgent: Bool {
        if case .agent = self { return true }
        return false
    }
}

/// A key of the query's JSON, written as it goes on the wire.
struct WireKey: CodingKey, ExpressibleByStringLiteral {
    let stringValue: String
    var intValue: Int? { nil }

    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
    init(stringLiteral value: String) { stringValue = value }
}

/// One turn of a session as the router wrote it down: what was asked, and how answering it
/// ended. This is what a rewind or a fork names.
public struct Response: Sendable, Hashable, Identifiable {
    public let id: String
    public let sessionID: String
    /// What the person asked. Empty for a turn the agent started on its own, like a greeting.
    public let said: String
    public let status: Status
    /// What went wrong, for a failed turn.
    public let error: String
    public let createdAt: Date
    public let finishedAt: Date?

    public enum Status: String, Sendable {
        case running
        case completed
        case failed
        /// Interrupted by the caller, which is not a failure: what was said still counts.
        case cancelled
        /// A status this SDK has never heard of.
        case unknown
    }

    init(_ schema: AgentResponse) {
        id = schema.id
        sessionID = schema.sessionId
        said = schema.said ?? ""
        status = Status(rawValue: schema.status.rawValue) ?? .unknown
        error = schema.error ?? ""
        createdAt = schema.createdAt
        finishedAt = schema.finishedAt
    }
}

/// One thing the agent did while answering a response: what was said, thought, answered,
/// blocked or failed, in the order it happened.
public struct ResponseItem: Sendable, Hashable {
    public let responseID: String
    /// Where this falls within its response.
    public let ordinal: Int
    public let kind: Kind
    public let text: String
    /// The tool called, for an item that records a tool call.
    public let toolName: String

    public enum Kind: String, Sendable {
        case said
        case thought
        case answer
        case blocked
        case error
        /// A kind this SDK has never heard of.
        case unknown
    }

    init(_ schema: AgentResponseItem) {
        responseID = schema.responseId
        ordinal = schema.ordinal
        kind = Kind(rawValue: schema.kind.rawValue) ?? .unknown
        text = schema.text ?? ""
        toolName = schema.toolName ?? ""
    }
}

/// An image to show the agent with what is asked, by URL or as a data URI.
public struct ImageSource: Sendable, Hashable {
    public var url: String
    /// How closely the model looks. Nil lets the model decide.
    public var detail: Detail?

    public enum Detail: String, Sendable {
        case auto
        case low
        case high
    }

    public init(url: String, detail: Detail? = nil) {
        self.url = url
        self.detail = detail
    }

    var schema: ImageSourcePayload {
        .init(detail: detail.map { .init(rawValue: $0.rawValue) }, url: url)
    }
}

/// What to change about a conversation while continuing it as a new one.
///
/// `nil` means the fork keeps what the parent had.
public struct ForkOptions: Sendable {
    /// Carry the history only up to the end of this response, and branch from there.
    public var responseID: String?
    /// Another agent to continue as, by the name its config was synced under.
    public var agent: String?
    /// Another agent config to continue as, by id.
    public var configID: String?
    public var title: String?
    public var description: String?
    public var projectID: String?
    /// The fork's own labels, in place of the parent's.
    public var custom: [String: RawJSON]?
    /// What to change about the models for the fork alone, which is the usual reason to fork:
    /// asking the same question of another model.
    public var modelOverwrites: ModelOverwrites?
    /// Keeps the fork off the record. The parent is kept as it was.
    public var incognito: Bool?
    /// Start the fork with none of the parent's history. Cannot be combined with
    /// `responseID`, which is a point in that history.
    public var withoutHistory = false

    public init(responseID: String? = nil) {
        self.responseID = responseID
    }
}

/// What to change about the models for one conversation, over whatever its config decided.
///
/// `nil` leaves a field to the config. Instructions and tools are not here: a device that could
/// rewrite those could make the agent another one.
public struct ModelOverwrites: Sendable, Hashable {
    /// A provider/model or a capability shortcut.
    public var llm: String?
    public var stt: String?
    public var tts: String?
    /// A speech-to-speech target, which opens no transcriber, model or voice.
    public var sts: String?
    public var search: String?
    /// How hard the model reasons before answering. A model that does not reason ignores it.
    public var thinking: Thinking?
    /// Zero is a real request for a deterministic model, which nil is not.
    public var temperature: Double?
    /// Caps every reply, reasoning included.
    public var maxOutputTokens: Int?
    /// How much detail to give. A model that does not take it ignores it.
    public var verbosity: Verbosity?

    public enum Thinking: String, Sendable {
        case none
        case minimal
        case low
        case medium
        case high
    }

    public enum Verbosity: String, Sendable {
        case low
        case medium
        case high
    }

    public init(
        llm: String? = nil, stt: String? = nil, tts: String? = nil, sts: String? = nil,
        search: String? = nil, thinking: Thinking? = nil, temperature: Double? = nil,
        maxOutputTokens: Int? = nil, verbosity: Verbosity? = nil
    ) {
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.sts = sts
        self.search = search
        self.thinking = thinking
        self.temperature = temperature
        self.maxOutputTokens = maxOutputTokens
        self.verbosity = verbosity
    }

    var schema: ModelOverwritesPayload {
        .init(
            llm: llm, maxOutputTokens: maxOutputTokens, search: search, sts: sts, stt: stt,
            temperature: temperature, thinking: thinking.map { .init(rawValue: $0.rawValue) }, tts: tts,
            verbosity: verbosity.map { .init(rawValue: $0.rawValue) })
    }
}

/// What the agent says as it joins a call, before anyone speaks.
public struct Greeting: Sendable, Hashable {
    /// Empty means the agent waits to be spoken to.
    public var text: String
    /// Nil says it word for word, as `exact` does.
    public var mode: Mode?

    public enum Mode: String, Sendable {
        /// Word for word.
        case exact
        /// The model rewords it on every call, so callers do not hear the same opening.
        case variation
    }

    public init(_ text: String, mode: Mode? = nil) {
        self.text = text
        self.mode = mode
    }

    var schema: GreetingPayload {
        .init(mode: mode.map { .init(rawValue: $0.rawValue) }, text: text)
    }
}
