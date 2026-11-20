package io.getstream.agents.android.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Instant

/** Frames exactly as `frameOf` in the router writes them. */
private fun event(json: String): AgentEvent = requireNotNull(AgentEvent.decode(json))

private val start = Instant.fromEpochSeconds(1_000)

private fun at(seconds: Double): Instant = start + (seconds * 1000).toLong().milliseconds

private fun Conversation.at(seconds: Double, frame: String): Conversation = reduce(event(frame), at(seconds))

/**
 * A question that needs a search, asked in an incognito session: the model thinks, calls the
 * search, and takes its result up in a turn of its own.
 */
private fun searched(): Conversation = Conversation()
    .said("Is the Belem Tower open on Sunday?", at(0.0))
    .at(0.0, """{"type":"responding","turn_id":"t1","prompt":"Is the Belem Tower open on Sunday?"}""")
    .at(2.0, """{"type":"reasoning_delta","turn_id":"t1","text":"Look up its hours."}""")
    .at(3.0, """{"type":"tool_started","tool_call_id":"call_1","tool":"search","turn_id":"t1","started_at":"2026-11-19T10:00:03Z"}""")
    .at(3.25, """{"type":"tool_updated","id":"call_1","turn_id":"t1","tool":"search","text":"Reading 3 pages","verbatim":false}""")
    .at(3.5, """{"type":"tool_ran","tool_call_id":"call_1","turn_id":"t1","tool":"search","arguments":"{}","result":"10:00-18:30","files":null,"error":""}""")
    .at(3.5, """{"pending_work":true,"type":"responded","turn_id":"t1","text":"","time_to_first_token_ms":0}""")
    .at(4.0, """{"type":"responding","turn_id":"t2","prompt":""}""")
    .at(5.0, """{"type":"reasoning_delta","turn_id":"t2","text":"Sundays are open."}""")
    .at(6.0, """{"type":"response_delta","turn_id":"t2","text":"Yes, from 10:00."}""")

private val reminder = AgentTool(
    "add_reminder",
    "Adds a reminder.",
    approval = AgentTool.Approval(
        title = "Add a reminder?",
        message = "Juno wants to add one.",
        reasonArgument = "why",
        allowTitle = "Add",
        declineTitle = "Not now",
    ),
) { "added" }

class TurnMessageTest {
    @Test
    fun `an incognito reply is one message across its tools' turns, with every step`() {
        val conversation = searched()
            .at(7.0, """{"pending_work":false,"type":"responded","turn_id":"t2","text":"Yes, from 10:00.","time_to_first_token_ms":900}""")

        val messages = TurnMessage.messages(conversation, userId = "jlahey", isConnected = true)

        assertEquals(listOf(false, true), messages.map { it.isAgentReply })
        assertEquals("Is the Belem Tower open on Sunday?", messages[0].text)
        val reply = messages[1]
        assertEquals("t1", reply.id)
        assertEquals("Yes, from 10:00.", reply.text)
        assertFalse(reply.isGenerating)
        assertEquals(
            listOf(
                TurnMessage.Step("ai_reasoning", mapOf("id" to "r1", "status" to "completed", "duration_ms" to 3000L)),
                TurnMessage.Step(
                    "ai_tool_call",
                    mapOf(
                        "id" to "call_1",
                        "name" to "search",
                        "status" to "completed",
                        "summary" to "Reading 3 pages",
                        "duration_ms" to 500L,
                    ),
                ),
                // Numbered across the reply, as Stream Chat's copy of a kept one is.
                TurnMessage.Step("ai_reasoning", mapOf("id" to "r2", "status" to "completed", "duration_ms" to 2000L)),
            ),
            reply.steps,
        )
        assertEquals(mapOf("r1" to "Look up its hours.", "r2" to "Sundays are open."), reply.reasoning)
        assertEquals(emptyMap(), reply.customData)
    }

    @Test
    fun `the agent's text across a reply's turns is joined by a blank line`() {
        val conversation = Conversation()
            .at(0.0, """{"type":"responding","turn_id":"t1"}""")
            .at(1.0, """{"type":"response_delta","turn_id":"t1","text":"Let me look."}""")
            .at(2.0, """{"pending_work":true,"type":"responded","turn_id":"t1","text":"Let me look."}""")
            .at(3.0, """{"type":"responding","turn_id":"t2"}""")
            .at(4.0, """{"pending_work":false,"type":"responded","turn_id":"t2","text":"Sunny and 24°C."}""")

        val reply = TurnMessage.messages(conversation, userId = "jlahey", isConnected = true).single()

        assertEquals("Let me look.\n\nSunny and 24°C.", reply.text)
    }

    @Test
    fun `only the latest reply is being written, and only while the session is connected`() {
        val conversation = searched()

        val connected = TurnMessage.messages(conversation, userId = "jlahey", isConnected = true)
        val disconnected = TurnMessage.messages(conversation, userId = "jlahey", isConnected = false)

        assertTrue(connected.last().isGenerating)
        assertEquals("completed", connected.last().steps.last().fields["status"])
        assertFalse(disconnected.last().isGenerating)

        val askedAgain = TurnMessage.messages(conversation.said("And on Monday?"), userId = "jlahey", isConnected = true)
        assertEquals(listOf(false, false, false), askedAgain.map { it.isGenerating })
    }

    @Test
    fun `a call waiting for the person asks the session's user on its step`() {
        val conversation = Conversation()
            .said("Remind me to call Bob at 5.")
            .at(0.0, """{"type":"responding","turn_id":"t1","prompt":"Remind me to call Bob at 5."}""")
            .at(1.0, """{"type":"tool_started","tool_call_id":"call_9","tool":"add_reminder","turn_id":"t1","started_at":"2026-11-19T10:00:03Z"}""")
        val call = event(
            """{"type":"tool_call","id":"call_9","name":"add_reminder","arguments":"{\"why\":\"You asked to call Bob.\"}","request_id":"m1","turn_id":"t1"}""",
        ).toolCall!!
        val waiting = ToolApprovalRequest(call, reminder.approval!!, reminder)

        val messages = TurnMessage.messages(conversation, listOf(waiting), userId = "jlahey", isConnected = true)
        val step = messages.last().steps.single().fields

        assertEquals("awaiting_approval", step["status"])
        assertEquals("jlahey", step["target_user_id"])
        assertEquals(
            mapOf(
                "title" to "Add a reminder?",
                "message" to "Juno wants to add one.",
                "reason" to "You asked to call Bob.",
                "allow_title" to "Add",
                "decline_title" to "Not now",
            ),
            step["approval"],
        )

        // Once answered, it is a call like any other.
        val answered = TurnMessage.messages(conversation, userId = "jlahey", isConnected = true)
        val after = answered.last().steps.single().fields
        assertEquals("running", after["status"])
        assertNull(after["approval"])
        assertNull(after["target_user_id"])
    }

    @Test
    fun `a reply's UI is its latest turn's, as a kept reply holds it`() {
        val conversation = Conversation()
            .at(0.0, """{"type":"responding","turn_id":"t1","prompt":"Plan Ohrid"}""")
            .at(1.0, """{"type":"ui_completed","turn_id":"t1","format":"openui","content":"root = Card([hdr])\nhdr = CardHeader(\"Ohrid\")"}""")

        val reply = TurnMessage.messages(conversation, userId = "jlahey", isConnected = true).single()

        assertEquals(mapOf("ai_ui" to conversation.turns.single().ui!!.customData), reply.customData)
        assertEquals("complete", (reply.customData["ai_ui"] as Map<*, *>)["status"])
    }
}
