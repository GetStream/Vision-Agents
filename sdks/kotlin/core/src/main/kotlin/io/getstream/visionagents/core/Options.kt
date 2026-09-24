package io.getstream.visionagents.core

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
    /** An agent config to start from, by the name it was synced under. */
    val agent: String? = null,
    /** An agent config to start from, by id. Naming both this and [agent] is refused. */
    val configId: String? = null,
    /** What to call the conversation, for a list a person reads. The model never sees it. */
    val title: String? = null,
    /** A longer note about the conversation, searched alongside the title. */
    val description: String? = null,
    /** What the conversation belongs to. Also recorded as its "project" cost tag. */
    val project: String? = null,
    /** Anything of yours to remember about the session. Sessions can be queried by it. */
    val custom: JsonObject? = null,
    /** Record nothing about this conversation: it cannot be found, rewound or forked afterwards. */
    val incognito: Boolean? = null,
    /** Keep a text conversation in Stream Chat. */
    val persistConversation: Boolean? = null,
    /** The Stream Chat channel of a persistent conversation to carry on. */
    val conversationId: String? = null,
    /** What to change about the models for this conversation alone. */
    val modelOverwrites: ModelOverwrites? = null,
    /** The system prompt. */
    val instructions: String? = null,
    /** Said on joining without going through the model. */
    val greeting: String? = null,
    val llm: String? = null,
    val stt: String? = null,
    val tts: String? = null,
    /** A provider-specific voice id. */
    val voice: String? = null,
    /** Functions of yours the agent may call, answered on this device. */
    val tools: List<AgentTool> = emptyList(),
    /** Cost labels, carried onto every request the session makes. */
    val tags: Map<String, String> = emptyMap(),
)

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
    val project: String? = null,
    val custom: JsonObject? = null,
    val modelOverwrites: ModelOverwrites? = null,
    val instructions: String? = null,
    /** Keep the fork off the record. */
    val incognito: Boolean? = null,
    /**
     * Start the fork with none of the parent's history. Cannot be combined with [responseId],
     * which is a point in that history.
     */
    val withoutHistory: Boolean = false,
    /** The call the fork joins, which a voice session needs and a text session refuses. */
    val callId: String? = null,
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
    val configId: String? = null,
    val project: String? = null,
    /** Omitted is both. */
    val state: State? = null,
    /** Labels a session's custom object must contain, all of them. */
    val custom: JsonObject? = null,
    val createdAfter: Instant? = null,
    val createdBefore: Instant? = null,
    /** Up to 200. Omitted is 25. */
    val limit: Int? = null,
    val offset: Int? = null,
) {
    public enum class State { Running, Closed }

    internal fun parameters(): List<Pair<String, String>> = queryOf(
        "agent" to agent,
        "config_id" to configId,
        "project" to project,
        "state" to state?.name?.lowercase(),
        "custom" to custom?.toString(),
        "created_after" to createdAfter?.toString(),
        "created_before" to createdBefore?.toString(),
        "limit" to limit,
        "offset" to offset,
    )
}
