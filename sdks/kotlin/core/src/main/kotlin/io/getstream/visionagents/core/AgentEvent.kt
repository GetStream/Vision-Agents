package io.getstream.visionagents.core

import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject

/** Who said something. */
public data class Participant(
    val id: String,
    val userId: String,
    val name: String,
) {
    /** What to show for this participant: their name, or their user id when they have none. */
    public val display: String get() = name.ifEmpty { userId }
}

/**
 * One event on a session's socket.
 *
 * The fields are kept as they arrived so that an event added to the router after this SDK
 * shipped still reaches the caller, with [kind] null and [type] naming it. Switching on [kind]
 * covers what is known; `event["whatever"]` covers the rest.
 */
public data class AgentEvent(
    /** The router's own name for this event, always present. */
    val type: String,
    /** The event's fields, flattened as the router sends them, without `type`. */
    val fields: JsonObject = JsonObject(emptyMap()),
) {
    /** This event as one of the kinds the SDK knows, or null for one it does not. */
    public val kind: Kind? get() = Kind.entries.firstOrNull { it.wire == type }

    /** One field, or [JsonNull] when the event has none by that name. */
    public operator fun get(key: String): JsonElement = fields[key] ?: JsonNull

    /** What was said, transcribed or generated, depending on the event. */
    public val text: String get() = string("text")

    /** The turn this belongs to, or the empty string for events outside a turn. */
    public val turnId: String get() = string("turn_id")

    /** What went wrong, for `error` and for the events that carry a failure of their own. */
    public val errorText: String get() = string("error")

    /** Who this is about, or null for the events that are about nobody. */
    public val participant: Participant?
        get() {
            val fields = this["participant"] as? JsonObject ?: return null
            return Participant(
                id = fields.string("id"),
                userId = fields.string("user_id"),
                name = fields.string("name"),
            )
        }

    /** A tool the model wants run, or null when this event is not a tool call. */
    public val toolCall: ToolCall?
        get() = if (kind != Kind.ToolCall) null else ToolCall(
            id = string("id"),
            name = string("name"),
            arguments = string("arguments"),
            commandId = string("command_id"),
            turnId = turnId,
        )

    /** A string field, or the empty string. */
    public fun string(key: String): String = fields.string(key)

    /** The events the router documents. A frame outside this set is still delivered. */
    public enum class Kind(public val wire: String) {
        Joined("joined"),
        ParticipantJoined("participant_joined"),
        ParticipantLeft("participant_left"),
        Hearing("hearing"),
        Heard("heard"),
        Decision("decision"),
        Responding("responding"),
        ResponseDelta("response_delta"),
        Responded("responded"),
        Blocked("blocked"),
        Spoke("spoke"),
        Turn("turn"),
        Delegated("delegated"),
        TaskSettled("task_settled"),
        TaskCancelled("task_cancelled"),
        ToolCall("tool_call"),
        ToolCancel("tool_cancel"),
        ToolStarted("tool_started"),
        ToolRan("tool_ran"),
        Transferred("transferred"),
        Pressed("pressed"),
        LookedUp("looked_up"),
        Backchannel("backchannel"),
        Interrupted("interrupted"),
        OverlapDecided("overlap_decided"),
        ConversationCompacted("conversation_compacted"),
        ConversationUpdated("conversation_updated"),
        CommandAccepted("command_accepted"),
        CommandStopped("command_stopped"),
        ModelsChanged("models_changed"),
        Error("error"),
        Left("left"),
    }

    /** A request from the model to run one of the caller's functions. */
    public data class ToolCall(
        val id: String,
        val name: String,
        /** The arguments as the model wrote them, which is a JSON object encoded as a string. */
        val arguments: String,
        /** Repeated on the answer, so a result cannot be adopted by another command or turn. */
        val commandId: String,
        val turnId: String,
    ) {
        /** The arguments decoded, or an empty object if the model wrote something else. */
        public val argumentValues: JsonObject
            get() = try {
                wire.parseToJsonElement(arguments).jsonObject
            } catch (_: SerializationException) {
                JsonObject(emptyMap())
            } catch (_: IllegalArgumentException) {
                JsonObject(emptyMap())
            }
    }

    public companion object {
        /**
         * Reads one frame, or null for one that is not a JSON object with a type.
         *
         * A frame this SDK cannot read is skipped rather than ending the conversation, which is
         * what the router does with a command it cannot read.
         */
        public fun decode(frame: String): AgentEvent? {
            val fields = try {
                wire.parseToJsonElement(frame) as? JsonObject
            } catch (_: SerializationException) {
                null
            } ?: return null
            val type = (fields["type"] as? JsonPrimitive)?.takeIf { it.isString }?.content
                ?: return null
            return AgentEvent(type, JsonObject(fields - "type"))
        }
    }
}

private fun JsonObject.string(key: String): String =
    (this[key] as? JsonPrimitive)?.contentOrNull.orEmpty()
