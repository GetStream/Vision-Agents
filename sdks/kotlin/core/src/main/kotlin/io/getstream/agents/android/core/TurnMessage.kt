package io.getstream.agents.android.core

import kotlin.time.Instant
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine

/**
 * A message of a conversation kept in no channel, an incognito session's, in the shape a kept one
 * has in Stream Chat, so an app shows both alike with Stream's AI components.
 *
 * Each of the person's turns is a message, and the agent's turns between two of them are one
 * reply, as the router writes a reply whose tools it waited on: their text, the latest UI, and
 * each round of thinking and each tool call as a step, generating until the agent is done with
 * it. A call waiting in the session's `approvals` asks its user on its step, as a kept reply's
 * does.
 */
public data class TurnMessage(
    /** Its first turn's id. */
    val id: String,
    /** Whether the agent wrote it, rather than the person. */
    val isAgentReply: Boolean,
    val text: String,
    /** When its first turn began. */
    val at: Instant,
    /** Whether the agent is still writing it. */
    val isGenerating: Boolean,
    /**
     * The reply's steps, in order, as a kept reply's attachments hold them: an `ai_reasoning` for
     * each round of thinking, numbered from `r1` across the reply, and an `ai_tool_call` for each
     * call. Stream Chat's `AIMessagePart.parts` reads them as `type to fields`.
     */
    val steps: List<Step>,
    /** The whole of each round of thinking, by its step's id. */
    val reasoning: Map<String, String>,
    /**
     * The reply's custom data as a kept one holds it, in plain Kotlin values: its UI under
     * `ai_ui`, which Stream Chat's `AIUIDocument.from` reads, or nothing.
     */
    val customData: Map<String, Any?>,
) {
    /** A step of a reply, as an attachment's type and its fields. */
    public data class Step(val type: String, val fields: Map<String, Any?>)

    public companion object {
        /**
         * The messages of [conversation], oldest first.
         *
         * @param approvals The calls waiting for the person, each of which asks on its step.
         * @param userId The person the session is with, whom a waiting call asks: the user
         * `setUser` named, which an `AIToolApprover` matches against `target_user_id`.
         * @param isConnected Whether the session's socket still carries the conversation. The
         * latest reply is written only while it does.
         */
        public fun messages(
            conversation: Conversation,
            approvals: List<ToolApprovalRequest> = emptyList(),
            userId: String,
            isConnected: Boolean,
        ): List<TurnMessage> {
            val groups = mutableListOf<MutableList<Turn>>()
            for (turn in conversation.turns) {
                val last = groups.lastOrNull()
                if (turn.isAgent && last?.first()?.isAgent == true) last += turn else groups += mutableListOf(turn)
            }
            return groups.mapIndexed { index, turns ->
                // Only the latest reply can still be written: one the person asked over is as it
                // was left.
                val isLatest = index == groups.lastIndex
                of(turns, isLatest && isConnected && !turns.last().isDone, approvals, userId)
            }
        }

        private fun of(turns: List<Turn>, isGenerating: Boolean, approvals: List<ToolApprovalRequest>, userId: String): TurnMessage {
            val first = turns.first()
            if (!first.isAgent) {
                return TurnMessage(first.id, false, first.text, first.at, false, emptyList(), emptyMap(), emptyMap())
            }
            val steps = mutableListOf<Step>()
            val reasoning = mutableMapOf<String, String>()
            for (step in turns.flatMap { it.steps }) {
                when (step) {
                    is TurnStep.Reasoning -> {
                        // Numbered across the reply, which a tool's result carries on in a turn of its own.
                        val id = "r${reasoning.size + 1}"
                        reasoning[id] = step.text
                        val fields = buildMap {
                            put("id", id)
                            put("status", if (step.isStreaming) "streaming" else "completed")
                            step.duration?.let { put("duration_ms", it.inWholeMilliseconds) }
                        }
                        steps += Step("ai_reasoning", fields)
                    }
                    is TurnStep.ToolCall -> {
                        val waiting = approvals.firstOrNull { it.id == step.id }
                            ?.takeIf { step.status == TurnStep.ToolCall.Status.Running }
                        val fields = buildMap {
                            put("id", step.id)
                            put("name", step.name)
                            put("status", if (waiting != null) "awaiting_approval" else step.status.wire)
                            if (step.summary.isNotEmpty()) put("summary", step.summary)
                            step.duration?.let { put("duration_ms", it.inWholeMilliseconds) }
                            if (waiting != null) {
                                put("target_user_id", userId)
                                put("approval", approval(waiting))
                            }
                        }
                        steps += Step("ai_tool_call", fields)
                    }
                }
            }
            val ui = turns.lastOrNull { it.ui != null }?.ui
            return TurnMessage(
                id = first.id,
                isAgentReply = true,
                text = turns.map { it.text }.filter { it.isNotEmpty() }.joinToString("\n\n"),
                at = first.at,
                isGenerating = isGenerating,
                steps = steps,
                reasoning = reasoning,
                customData = if (ui == null) emptyMap() else mapOf("ai_ui" to ui.customData),
            )
        }

        /** The question a waiting call asks, as its step in Stream Chat holds it. */
        private fun approval(waiting: ToolApprovalRequest): Map<String, Any?> = buildMap {
            put("title", waiting.approval.title)
            waiting.approval.message?.takeIf { it.isNotEmpty() }?.let { put("message", it) }
            waiting.reason.takeIf { it.isNotEmpty() }?.let { put("reason", it) }
            waiting.approval.allowTitle?.takeIf { it.isNotEmpty() }?.let { put("allow_title", it) }
            waiting.approval.declineTitle?.takeIf { it.isNotEmpty() }?.let { put("decline_title", it) }
        }
    }
}

/**
 * The conversation as messages, for a session kept in no channel, an incognito one, to show with
 * Stream's AI components; a kept session's messages are its channel's ([chat]). A new list each
 * time the conversation, the calls waiting in [AgentSession.approvals] or the socket changes. A
 * waiting call asks the user `setUser` named.
 */
public val AgentSession.turnMessages: Flow<List<TurnMessage>>
    get() = combine(conversation, approvals, isConnected) { conversation, approvals, isConnected ->
        TurnMessage.messages(conversation, approvals, backend.user?.id.orEmpty(), isConnected)
    }
