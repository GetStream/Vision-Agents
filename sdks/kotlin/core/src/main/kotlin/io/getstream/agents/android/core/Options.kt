package io.getstream.agents.android.core

import io.getstream.agents.android.core.generated.Equals
import io.getstream.agents.android.core.generated.EqualsOrIn
import io.getstream.agents.android.core.generated.Greeting as GreetingSchema
import io.getstream.agents.android.core.generated.GreetingMode
import io.getstream.agents.android.core.generated.SessionFilter as FilterSchema
import io.getstream.agents.android.core.generated.SessionMemory
import io.getstream.agents.android.core.generated.SessionSort as SortSchema
import io.getstream.agents.android.core.generated.SessionSortField as SortFieldSchema
import io.getstream.agents.android.core.generated.TextMatch
import java.util.TimeZone
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject

/**
 * What a session should be, for the cases the shorthands do not cover.
 *
 * Everything is optional because everything has an answer already: a named config decides
 * what this does not say, and the router decides what the config does not. Null means "leave
 * it off the request", never "send a copy of the server's default".
 */
public data class SessionOptions(
    /**
     * An id to hold the session by, for a caller that wants to know it before the session
     * exists: up to 64 letters, digits, `-` and `_`. One already taken is refused with a 409.
     * Null lets the router generate one.
     */
    val id: String? = null,
    /** An agent config to start from, by the name it was synced under. */
    val agent: String? = null,
    /** An agent config to start from, by id. Naming both this and [agent] is refused. */
    val configId: String? = null,
    /** What to call the conversation, for a list a person reads. The model never sees it. */
    val title: String? = null,
    /** A longer note about the conversation, searched alongside the title. */
    val description: String? = null,
    /** What the conversation belongs to. Also recorded as its "project" cost tag. */
    val projectId: String? = null,
    /** Anything of yours to remember about the session. Sessions can be queried by it. */
    val custom: JsonObject? = null,
    /** Record nothing about this conversation: it cannot be found, rewound or forked afterwards. */
    val incognito: Boolean? = null,
    /**
     * Has the agent join the session's own call, `agent:<session id>`, as soon as it opens.
     * False holds the conversation in writing until [AgentSession.startVoice] starts it.
     */
    val startVoice: Boolean = false,
    /** What to change about the models for this conversation alone. */
    val modelOverwrites: ModelOverwrites? = null,
    /** What the agent opens the call with. */
    val greeting: Greeting? = null,
    val llm: String? = null,
    val stt: String? = null,
    val tts: String? = null,
    /**
     * The voice the agent speaks in: a provider's own voice id, or one of your voices by id or by
     * the name you gave it. Prefix it with `custom:` to mean only yours.
     */
    val voice: String? = null,
    /** Functions of yours the agent may call, answered on this device. */
    val tools: List<AgentTool> = emptyList(),
    /** Cost labels, carried onto every request the session makes. */
    val tags: Map<String, String> = emptyMap(),
    /**
     * Where a sign-in the conversation asks for goes back to. Name the app's own scheme in
     * reverse domain form, such as `com.example.app:/connected`, and the router hands the
     * browser back to the app. It is kept with the session, so a conversation carried on later
     * goes back there too. Null sends the sign-in back to the dashboard, which a phone cannot
     * open.
     */
    val returnUrl: String? = null,
    /**
     * Has the agent remember the signed-in user: what it learned about them before is recalled
     * here, and what it learns here is kept for their next session. Null remembers nothing, and
     * an [incognito] session never does.
     */
    val memory: Memory? = null,
    /**
     * The user's IANA zone, which the device knows and the router does not. The agent reads
     * dates in it, and it is remembered as the user's for the scheduled tasks that follow.
     * Left out, it is the device's own zone, `TimeZone.getDefault().id`, when the device names
     * it the IANA way. A device set to a bare offset, such as `GMT+05:30`, sends none rather
     * than a zone the router would refuse. Null sends none. A zone named here is sent as it
     * is, and one the router does not know is refused.
     */
    val timezone: String? = deviceTimezone(),
) {
    /**
     * What a session remembers about the user [Backend.setUser] named. It is always theirs: a
     * device recalls nobody else's.
     */
    public data class Memory(
        /** Your own labels, which narrow what is recalled. They cannot widen it past the user. */
        val filter: Map<String, String> = emptyMap(),
        /** Separates two of your apps sharing one memory account. */
        val appId: String? = null,
    ) {
        /** The router remembers only for a user id, and a device only ever means its own user. */
        internal fun schema(userId: String?): SessionMemory =
            SessionMemory(appId = appId?.ifEmpty { null }, filter = filter.ifEmpty { null }, userId = userId)
    }
}

/**
 * Where the device's zone is read from: the JVM's default, which Android keeps as the zone the
 * phone is set to. Tests stand another in.
 */
internal var deviceZoneId: () -> String? = { TimeZone.getDefault().id }

/**
 * The device's zone, when the device names it the IANA way, such as `Europe/Amsterdam` or
 * `UTC`. A bare offset, such as `GMT+05:30`, is not a name the router knows, so it is none
 * rather than a session the router refuses.
 */
internal fun deviceTimezone(): String? =
    deviceZoneId()?.takeIf { it.contains('/') || it == "UTC" || it == "GMT" }

/** What the agent says as it joins a call, before anyone speaks. */
public data class Greeting(
    /** Empty means the agent waits to be spoken to. */
    val text: String,
    /** Null says it word for word, as [Mode.Exact] does. */
    val mode: Mode? = null,
) {
    public enum class Mode {
        /** Word for word. */
        Exact,

        /** The model rewords it on every call, so callers do not hear the same opening. */
        Variation,
    }

    internal val schema: GreetingSchema
        get() = GreetingSchema(
            text,
            when (mode) {
                Mode.Exact -> GreetingMode.exact
                Mode.Variation -> GreetingMode.variation
                null -> null
            },
        )
}

/**
 * What to change about a conversation while continuing it as a new one.
 *
 * Null means the fork keeps what the parent had.
 */
public data class ForkOptions(
    /** Carry the history only up to the end of this response, and branch from there. */
    val responseId: String? = null,
    /** Another agent config to continue as, by name. */
    val agent: String? = null,
    /** Another agent config to continue as, by id. */
    val configId: String? = null,
    val title: String? = null,
    val description: String? = null,
    val projectId: String? = null,
    val custom: JsonObject? = null,
    val modelOverwrites: ModelOverwrites? = null,
    /** Keep the fork off the record. */
    val incognito: Boolean? = null,
    /**
     * Start the fork with none of the parent's history. Cannot be combined with [responseId],
     * which is a point in that history.
     */
    val withoutHistory: Boolean = false,
)

/**
 * Which conversations to list, in the shape the router takes.
 *
 * Only ever this caller's own, whatever is asked for: the router narrows a device to its own
 * sessions, so this cannot be widened to anybody else's. A device filters on the project, the
 * agent and a text search, combined with [And] and [Or]:
 *
 *     SessionFilter.Or(SessionFilter.ProjectId("a"), SessionFilter.ProjectId("b"))
 *     SessionFilter.And(SessionFilter.ProjectIdIn(listOf("a", "b")), SessionFilter.Text("billing"))
 *
 * [And] and [Or] each take 1 to 10 filters and nest up to 3 deep. [toJson] is what is sent.
 */
public sealed class SessionFilter {
    internal abstract fun schema(): FilterSchema

    /** The filter as it goes on the wire, with the router's own key names. */
    public fun toJson(): JsonObject = wire.encodeToJsonElement(FilterSchema.serializer(), schema()).jsonObject

    /** Only this project's: `{"project_id": {"$eq": eq}}`. */
    public data class ProjectId(val eq: String) : SessionFilter() {
        override fun schema(): FilterSchema = FilterSchema(projectId = EqualsOrIn(dollarEq = eq))
    }

    /** Only those in one of these projects, 1 to 50 of them: `{"project_id": {"$in": [...]}}`. */
    public data class ProjectIdIn(val values: List<String>) : SessionFilter() {
        override fun schema(): FilterSchema = FilterSchema(projectId = EqualsOrIn(dollarIn = values))
    }

    /**
     * Only those opened against this agent name: `{"agent": {"$eq": eq}}`. From the sessions
     * of [StreamAgents.agent] the agent is that one, whatever is said here.
     */
    public data class Agent(val eq: String) : SessionFilter() {
        override fun schema(): FilterSchema = FilterSchema(agent = Equals(eq))
    }

    /**
     * Only those whose title, description, project or agent name match these words:
     * `{"text": {"$q": q}}`. Once per filter, on its own or as a direct item of the outermost
     * [And], and never inside an [Or]. [Sessions.search] sets it for you.
     */
    public data class Text(val q: String) : SessionFilter() {
        override fun schema(): FilterSchema = FilterSchema(text = TextMatch(q))
    }

    /** Every one of these holds: `{"$and": [...]}`. */
    public data class And(val filters: List<SessionFilter>) : SessionFilter() {
        public constructor(vararg filters: SessionFilter) : this(filters.toList())

        override fun schema(): FilterSchema = FilterSchema(dollarAnd = filters.map { it.schema() })
    }

    /** At least one of these holds: `{"$or": [...]}`. */
    public data class Or(val filters: List<SessionFilter>) : SessionFilter() {
        public constructor(vararg filters: SessionFilter) : this(filters.toList())

        override fun schema(): FilterSchema = FilterSchema(dollarOr = filters.map { it.schema() })
    }
}

/** What a list of conversations can be ordered by. */
public enum class SessionSortField {
    /** When the last message was sent. This is the order when no sort is given. */
    LastMessageAt,

    /** Pinned conversations first. Only descending, alone or followed by [LastMessageAt]. */
    Pinned,

    /** Best match first, which only sorts a filter holding [SessionFilter.Text]. Only descending. */
    Relevance,
}

/** One key to order by: [direction] is -1 for descending, the default, or 1 for ascending. */
public data class SessionSort(val field: SessionSortField, val direction: Int = -1) {
    internal fun schema(): SortSchema = SortSchema(
        field = when (field) {
            SessionSortField.LastMessageAt -> SortFieldSchema.last_message_at
            SessionSortField.Pinned -> SortFieldSchema.pinned
            SessionSortField.Relevance -> SortFieldSchema.relevance
        },
        direction = when (direction) {
            -1 -> SortSchema.Direction.Minus1
            1 -> SortSchema.Direction._1
            else -> throw AgentsException.Configuration("a sort direction is -1 or 1, not $direction")
        },
    )
}
