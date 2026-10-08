package io.getstream.visionagents.core

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * One thing a client can do to a running conversation over its socket.
 *
 * These are the commands `readCommands` in the router accepts, but `respond`: asking is
 * [Responses.create]. Everything else a caller might want is a request rather than a frame.
 */
public sealed interface Command {
    /** Speak this without going through the model. */
    public data class Say(val text: String) : Command

    /**
     * Abandon the reply in flight, or the command named, which is how a durable command is
     * stopped wherever it got to.
     */
    public data class Interrupt(val commandId: String = "") : Command

    /** Replace the system prompt, from the next turn on. */
    public data class Instructions(val instructions: String) : Command

    /**
     * Answer a tool call. One of [output] or [error] says how it went.
     *
     * [commandId] and [turnId] repeat the call's own, so the router can tell which command or
     * turn the answer belongs to.
     */
    public data class ToolResult(
        val id: String,
        val output: String = "",
        val error: String = "",
        val commandId: String = "",
        val turnId: String = "",
    ) : Command

    /** End the session. */
    public data object Close : Command

    /** The frame as the router reads it. */
    public fun encode(): String = frame().toString()
}

private fun Command.frame(): JsonObject = when (this) {
    is Command.Say -> buildJsonObject {
        put("type", "say")
        put("text", text)
    }
    Command.Close -> buildJsonObject { put("type", "close") }
    is Command.Interrupt -> buildJsonObject {
        put("type", "interrupt")
        if (commandId.isNotEmpty()) put("command_id", commandId)
    }
    is Command.Instructions -> buildJsonObject {
        put("type", "instructions")
        put("instructions", instructions)
    }
    is Command.ToolResult -> buildJsonObject {
        put("type", "tool_result")
        put("tool_call_id", id)
        // The router treats the empty string as absent, so sending it and sending nothing
        // are the same thing.
        put("output", output)
        put("error", error)
        if (commandId.isNotEmpty()) put("command_id", commandId)
        if (turnId.isNotEmpty()) put("turn_id", turnId)
    }
}
