package io.getstream.agents.androidassistant

import io.getstream.agents.android.core.AgentEvent
import io.getstream.agents.android.core.Conversation
import io.getstream.agents.android.core.TurnMessage
import io.getstream.chat.android.ai.compose.conversation.AIMessageRole
import io.getstream.chat.android.ai.compose.parts.AIToolCallPart
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** Frames as the router's session socket writes them. */
private fun Conversation.after(vararg frames: String): Conversation =
    frames.fold(this) { conversation, frame -> conversation.reduce(requireNotNull(AgentEvent.decode(frame))) }

class IncognitoMessageTest {
    @Test
    fun `an incognito reply shows its thinking and tool calls as Stream's components read a kept one`() {
        val conversation = Conversation().said("Plan a weekend in Lisbon").after(
            """{"type":"responding","turn_id":"t1"}""",
            """{"type":"reasoning_delta","turn_id":"t1","text":"They want Lisbon's weather."}""",
            """{"type":"tool_started","tool_call_id":"call_1","tool":"weather","turn_id":"t1","started_at":"2026-11-19T10:00:01Z"}""",
            """{"type":"tool_ran","tool_call_id":"call_1","turn_id":"t1","tool":"weather","error":""}""",
            """{"pending_work":true,"type":"responded","turn_id":"t1","text":""}""",
            """{"type":"responding","turn_id":"t2"}""",
            """{"type":"reasoning_delta","turn_id":"t2","text":"Sunny, 24."}""",
            """{"type":"response_delta","turn_id":"t2","text":"Sunny and 24°C."}""",
            """{"pending_work":false,"type":"responded","turn_id":"t2","text":"Sunny and 24°C."}""",
        )

        val (question, reply) = TurnMessage.messages(conversation, userId = "jlahey", isConnected = true)
            .map(AgentMessage::Unkept)

        assertEquals(AIMessageRole.User, question.role)
        assertEquals("Plan a weekend in Lisbon", question.text)
        assertEquals(AIMessageRole.Assistant, reply.role)
        assertEquals("Sunny and 24°C.", reply.text)
        assertEquals(listOf("r1", "call_1", "r2"), reply.parts.map { it.id })
        assertEquals("weather", reply.parts[1].toolCall?.name)
        assertEquals(AIToolCallPart.Status.Completed, reply.parts[1].toolCall?.status)
        assertEquals("They want Lisbon's weather.", reply.reasoningText(assertNotNull(reply.parts[0].reasoning)))
        assertEquals("Sunny, 24.", reply.reasoningText(assertNotNull(reply.parts[2].reasoning)))
    }

    @Test
    fun `a call waiting for the person shows its approval card to them`() {
        val message = TurnMessage(
            id = "t1",
            isAgentReply = true,
            text = "",
            at = kotlin.time.Instant.fromEpochSeconds(0),
            isGenerating = true,
            steps = listOf(
                TurnMessage.Step(
                    "ai_tool_call",
                    mapOf(
                        "id" to "call_9",
                        "name" to "add_reminder",
                        "status" to "awaiting_approval",
                        "target_user_id" to "jlahey",
                        "approval" to mapOf("title" to "Add a reminder?", "allow_title" to "Add"),
                    ),
                ),
            ),
            reasoning = emptyMap(),
            customData = emptyMap(),
        )

        val call = assertNotNull(AgentMessage.Unkept(message).parts.single().toolCall)

        assertEquals(AIToolCallPart.Status.AwaitingApproval, call.status)
        assertEquals("jlahey", call.targetUserId)
        assertEquals("Add a reminder?", call.approval?.title)
        assertEquals("Add", call.approval?.allowTitle)
        assertTrue(AgentMessage.Unkept(message).isGenerating)
    }
}
