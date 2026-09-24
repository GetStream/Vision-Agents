package io.getstream.visionagents.core

import io.getstream.visionagents.core.generated.AgentResponse as ResponseSchema
import io.getstream.visionagents.core.generated.AgentResponseItem as ItemSchema
import io.getstream.visionagents.core.generated.GuestUser as GuestSchema
import io.getstream.visionagents.core.generated.ImageSource as ImageSchema
import io.getstream.visionagents.core.generated.ModelOverwrites as OverwritesSchema
import io.getstream.visionagents.core.generated.Session as SessionSchema
import io.getstream.visionagents.core.generated.SessionState
import kotlin.time.Instant
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

/** A running or finished conversation. */
public data class Session(
    /**
     * What the router holds this session by. This addresses the session and its socket, and
     * it is not the call id.
     */
    val id: String,
    /** The Stream call the agent joined, which is what a video SDK joins. Empty for a text session. */
    val callId: String,
    val callType: String,
    /** Keys the transcript, and names the chat channel it is written to. */
    val agentId: String,
    /** The name the agent was addressed as, when it was addressed by one. */
    val agent: String,
    val configId: String,
    val isText: Boolean,
    val state: State,
    val title: String,
    val description: String,
    val project: String,
    val custom: JsonObject,
    val instructions: String,
    val llm: String,
    /** The Stream Chat channel a persistent text conversation is kept in, or empty. */
    val conversationId: String,
    /** The session this one was forked from, or empty. */
    val forkedFrom: String,
    val isIncognito: Boolean,
    val createdAt: Instant,
    val closedAt: Instant?,
    val lastResponseAt: Instant?,
) {
    public enum class State {
        Live,
        Ended,
    }

    internal companion object {
        fun of(schema: SessionSchema): Session = Session(
            id = schema.id,
            callId = schema.callId,
            callType = schema.callType,
            agentId = schema.agentId,
            agent = schema.agent.orEmpty(),
            configId = schema.configId.orEmpty(),
            isText = schema.text ?: false,
            state = if (schema.state == SessionState.live) State.Live else State.Ended,
            title = schema.title.orEmpty(),
            description = schema.description.orEmpty(),
            project = schema.project.orEmpty(),
            custom = JsonObject(schema.custom.orEmpty()),
            instructions = schema.instructions.orEmpty(),
            llm = schema.llm.orEmpty(),
            conversationId = schema.conversationId.orEmpty(),
            forkedFrom = schema.forkedFrom.orEmpty(),
            isIncognito = schema.incognito ?: false,
            createdAt = instant(schema.createdAt),
            closedAt = instantOrNull(schema.closedAt),
            lastResponseAt = instantOrNull(schema.lastResponseAt),
        )
    }
}

/**
 * One turn of a session as the router wrote it down: what was asked, and how answering it
 * ended. This is what a rewind or a fork names.
 *
 * Its [id] is not the `turn_id` socket events carry.
 */
public data class AgentResponse(
    val id: String,
    val sessionId: String,
    /** What the person asked. Empty for a turn the agent started on its own, like a greeting. */
    val said: String,
    val status: Status,
    /** What went wrong, for a failed turn. */
    val error: String,
    val createdAt: Instant,
    val finishedAt: Instant?,
) {
    public enum class Status {
        Running,
        Completed,
        Failed,
        /** Interrupted by the caller, which is not a failure: what was said still counts. */
        Cancelled,
        /** A status this SDK has never heard of. */
        Unknown,
    }

    internal companion object {
        fun of(schema: ResponseSchema): AgentResponse = AgentResponse(
            id = schema.id,
            sessionId = schema.sessionId,
            said = schema.said.orEmpty(),
            status = when (schema.status) {
                ResponseSchema.Status.running -> Status.Running
                ResponseSchema.Status.completed -> Status.Completed
                ResponseSchema.Status.failed -> Status.Failed
                ResponseSchema.Status.cancelled -> Status.Cancelled
                else -> Status.Unknown
            },
            error = schema.error.orEmpty(),
            createdAt = instant(schema.createdAt),
            finishedAt = instantOrNull(schema.finishedAt),
        )
    }
}

/** One thing that happened inside a turn: the question, a thought, a tool, the answer. */
public data class ResponseItem(
    /** The response this belongs to, which is what rewinding to it names. */
    val responseId: String,
    /** Where in its response it happened. */
    val ordinal: Int,
    val kind: Kind,
    val text: String,
    val toolName: String,
    /** What the kind carries that text cannot: a tool's arguments, a guardrail's reason. */
    val payload: JsonObject,
    val at: Instant,
) {
    public enum class Kind {
        Said,
        Thought,
        ToolCall,
        ToolResult,
        Answer,
        Blocked,
        Error,
        /** A kind this SDK has never heard of. */
        Unknown,
    }

    internal companion object {
        fun of(schema: ItemSchema): ResponseItem = ResponseItem(
            responseId = schema.responseId,
            ordinal = schema.ordinal,
            kind = when (schema.kind) {
                ItemSchema.Kind.said -> Kind.Said
                ItemSchema.Kind.thought -> Kind.Thought
                ItemSchema.Kind.tool_call -> Kind.ToolCall
                ItemSchema.Kind.tool_result -> Kind.ToolResult
                ItemSchema.Kind.answer -> Kind.Answer
                ItemSchema.Kind.blocked -> Kind.Blocked
                ItemSchema.Kind.error -> Kind.Error
                else -> Kind.Unknown
            },
            text = schema.text.orEmpty(),
            toolName = schema.toolName.orEmpty(),
            payload = JsonObject(schema.payload.orEmpty()),
            at = instant(schema.at),
        )
    }
}

/** Somebody talking to an agent before they have signed up. */
public data class Guest(
    val id: String,
    /** A Stream user token with role guest, which chat and video connect with as well. */
    val token: String,
    val name: String,
    val custom: JsonObject,
    /** When the token stops working, or null when the router did not say. */
    val expiresAt: Instant?,
) {
    internal companion object {
        fun of(schema: GuestSchema): Guest = Guest(
            id = schema.id,
            token = schema.token,
            name = schema.name.orEmpty(),
            custom = JsonObject(schema.custom.orEmpty()),
            expiresAt = instantOrNull(schema.expiresAt),
        )
    }
}

/** An image handed to the model alongside a question: an HTTP(S) URL or a data URI. */
public data class ImageSource(
    val url: String,
    val detail: Detail? = null,
) {
    public enum class Detail { Auto, Low, High }

    internal val schema: ImageSchema
        get() = ImageSchema(
            url = url,
            detail = when (detail) {
                Detail.Auto -> ImageSchema.Detail.auto
                Detail.Low -> ImageSchema.Detail.low
                Detail.High -> ImageSchema.Detail.high
                null -> null
            },
        )
}

/**
 * What to change about the models for one conversation, over whatever its config decided.
 *
 * Null leaves the config's choice alone.
 */
public data class ModelOverwrites(
    val llm: String? = null,
    val stt: String? = null,
    val tts: String? = null,
    val sts: String? = null,
    val subagent: String? = null,
    val search: String? = null,
    val thinking: Thinking? = null,
    /** Zero is a real request for a deterministic model, which null is not. */
    val temperature: Double? = null,
    val maxOutputTokens: Int? = null,
    val verbosity: Verbosity? = null,
) {
    public enum class Thinking { None, Minimal, Low, Medium, High }

    public enum class Verbosity { Low, Medium, High }

    internal val schema: OverwritesSchema
        get() = OverwritesSchema(
            llm = llm,
            stt = stt,
            tts = tts,
            sts = sts,
            subagent = subagent,
            search = search,
            thinking = thinking?.let { OverwritesSchema.Thinking.valueOf(it.name.lowercase()) },
            temperature = temperature,
            maxOutputTokens = maxOutputTokens,
            verbosity = verbosity?.let { OverwritesSchema.Verbosity.valueOf(it.name.lowercase()) },
        )
}

/** An empty map is the same as none: it is left off the wire. */
internal fun Map<String, JsonElement>.orNull(): Map<String, JsonElement>? = ifEmpty { null }
