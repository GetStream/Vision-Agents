package io.getstream.visionagents.android.core

import io.getstream.visionagents.android.core.generated.Equals
import io.getstream.visionagents.android.core.generated.Greeting as GreetingSchema
import io.getstream.visionagents.android.core.generated.GreetingMode
import io.getstream.visionagents.android.core.generated.SessionFilter
import io.getstream.visionagents.android.core.generated.SessionMemory
import io.getstream.visionagents.android.core.generated.SessionQuery as QuerySchema
import io.getstream.visionagents.android.core.generated.SessionSort
import io.getstream.visionagents.android.core.generated.SessionSortField
import io.getstream.visionagents.android.core.generated.TextMatch
import io.getstream.visionagents.android.core.generated.TimeRange
import kotlin.time.Instant
import kotlinx.serialization.json.JsonObject

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
 * Which conversations to list.
 *
 * Only ever this caller's own, whatever is asked for: the router narrows a device to its own
 * sessions, so this cannot be widened to anybody else's.
 */
public data class SessionQuery(
    /** Only those opened against this agent name. */
    val agent: String? = null,
    /** Only those created with this agent id, which names their transcript channel. */
    val agentId: String? = null,
    /** Only this project's. A search covers every project, so the router refuses it there. */
    val projectId: String? = null,
    /** Only those the user took part in this way. Omitted is every way. */
    val modality: Session.Modality? = null,
    /** Only the live ones, or only the ended ones. Omitted is both. */
    val state: Session.State? = null,
    /** Up to 200. Omitted is 25. */
    val limit: Int? = null,
    /** The [Page.nextCursor] of the page before, asked with the same filters. Omitted is the first page. */
    val cursor: String? = null,
    /** Only those started at or after this. */
    val createdSince: Instant? = null,
    /** Only those started before this. */
    val createdBefore: Instant? = null,
    /** Only those whose [SessionOptions.custom] labels hold every one of these pairs. */
    val custom: Map<String, String> = emptyMap(),
    /** The order. Omitted is most recently active first, or best match first for a search. */
    val sort: Sort? = null,
) {
    /** An order to list in. Every one is descending: the router offers no ascending one. */
    public enum class Sort {
        /** Most recently active first. */
        UpdatedAt,

        /** Best match first, which only sorts a search. */
        Relevance,
    }

    internal fun schema(text: String): QuerySchema = QuerySchema(
        filter = SessionFilter(
            agent = equals(agent),
            agentId = equals(agentId),
            createdAt = if (createdSince == null && createdBefore == null) {
                null
            } else {
                TimeRange(dollarGte = createdSince?.toString(), dollarLt = createdBefore?.toString())
            },
            custom = custom.ifEmpty { null },
            projectId = equals(projectId),
            modality = equals(
                when (modality) {
                    Session.Modality.Unknown -> throw AgentsException.Configuration("an unknown modality cannot be asked for")
                    else -> modality?.name?.lowercase()
                },
            ),
            state = equals(state?.name?.lowercase()),
            text = text.ifEmpty { null }?.let(::TextMatch),
        ),
        limit = limit?.toLong(),
        cursor = cursor?.ifEmpty { null },
        sort = sort?.let {
            listOf(
                SessionSort(
                    when (it) {
                        Sort.UpdatedAt -> SessionSortField.updated_at
                        Sort.Relevance -> SessionSortField.relevance
                    },
                ),
            )
        },
    )

    private fun equals(value: String?): Equals? = value?.ifEmpty { null }?.let(::Equals)
}
