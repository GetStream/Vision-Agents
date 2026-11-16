package io.getstream.agents.android.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Frames exactly as `frameOf` in the router writes them, so that a change to the wire format on
 * that side fails here rather than in somebody's app.
 */
private fun event(json: String): AgentEvent = requireNotNull(AgentEvent.decode(json))

private fun Conversation.after(vararg frames: String): Conversation =
    frames.fold(this) { conversation, frame -> conversation.reduce(event(frame)) }

class ConversationTest {
    @Test
    fun `a reply arrives one delta at a time and the final text wins`() {
        var conversation = Conversation().after("""{"type":"responding","turn_id":"t1","participant":{"id":"","user_id":"","name":""},"prompt":"hi"}""")
        assertEquals(Conversation.State.Responding, conversation.state)
        assertEquals(1, conversation.turns.size)

        conversation = conversation.after(
            """{"type":"response_delta","turn_id":"t1","text":"Hel"}""",
            """{"type":"response_delta","turn_id":"t1","text":"lo"}""",
        )
        assertEquals("Hello", conversation.turns.last().text)

        conversation = conversation.after(
            """{"pending_work":false,"type":"responded","turn_id":"t1","text":"Hello there.","time_to_first_token_ms":90}""",
        )
        assertEquals(listOf("Hello there."), conversation.turns.map { it.text })
        assertTrue(conversation.turns.last().isAgent)
        assertEquals(Conversation.State.Idle, conversation.state)
    }

    @Test
    fun `a delta for a turn nobody saw begin still lands`() {
        val conversation = Conversation().after("""{"type":"response_delta","turn_id":"t9","text":"mid"}""")

        assertEquals(listOf("mid"), conversation.turns.map { it.text })
        assertTrue(conversation.turns.last().isAgent)
    }

    @Test
    fun `a spoken turn with no final text keeps what was streamed`() {
        val conversation = Conversation().after(
            """{"type":"responding","turn_id":"t1"}""",
            """{"type":"response_delta","turn_id":"t1","text":"Sure."}""",
            """{"type":"responded","turn_id":"t1","text":""}""",
        )

        assertEquals(listOf("Sure."), conversation.turns.map { it.text })
    }

    @Test
    fun `what was heard on a call becomes a participant turn`() {
        val conversation = Conversation().after(
            """{"type":"heard","participant":{"id":"p1","user_id":"u1","name":"Alice"},"text":"what are your hours","language":"en"}""",
        )

        val turn = conversation.turns.single()
        assertEquals("what are your hours", turn.text)
        val speaker = assertIs<Turn.Speaker.Person>(turn.speaker)
        assertEquals("Alice", speaker.participant?.display)
    }

    @Test
    fun `typing something is not shown twice when the router echoes it`() {
        val conversation = Conversation()
            .said("what are your hours")
            .after("""{"type":"heard","participant":{},"text":"what are your hours"}""")

        assertEquals(1, conversation.turns.size)
    }

    @Test
    fun `a delegated skill is named while it runs`() {
        var conversation = Conversation().after("""{"type":"delegated","task_id":"k1","skill":"lookup_order","prompt":"","turn_id":"t1"}""")
        assertEquals(Conversation.State.Working(listOf("lookup_order")), conversation.state)

        conversation = conversation.after(
            """{"type":"task_settled","evidence":null,"task_id":"k1","skill":"lookup_order","text":"done","question":"","elapsed_ms":12,"error":""}""",
        )
        assertEquals(Conversation.State.Responding, conversation.state)
    }

    @Test
    fun `two skills at once both show until both settle`() {
        val conversation = Conversation().after(
            """{"type":"delegated","task_id":"k1","skill":"think"}""",
            """{"type":"delegated","task_id":"k2","skill":"recall"}""",
            """{"type":"task_settled","task_id":"k1","skill":"think"}""",
        )

        assertEquals(Conversation.State.Working(listOf("recall")), conversation.state)
    }

    @Test
    fun `the conversation ends when the agent leaves`() {
        val conversation = Conversation().after("""{"type":"left","at":"2026-09-02T17:05:00Z"}""")

        assertEquals(Conversation.State.Ended, conversation.state)
    }

    @Test
    fun `a reported failure is kept rather than thrown`() {
        val conversation = Conversation().after("""{"type":"error","context":"tts","error":"the voice is unknown"}""")

        assertEquals("the voice is unknown", conversation.failure)
    }

    @Test
    fun `the model failing mid-reply ends the turn and keeps the failure`() {
        val conversation = Conversation().after(
            """{"type":"responding","turn_id":"t1"}""",
            """{"type":"response_delta","turn_id":"t1","text":"Let me"}""",
            """{"type":"error","context":"llm","error":"the model is overloaded"}""",
        )

        assertEquals(Conversation.State.Idle, conversation.state)
        assertEquals("the model is overloaded", conversation.failure)
        assertEquals(listOf("Let me"), conversation.turns.map { it.text })
    }

    @Test
    fun `the speech model failing ends the turn too`() {
        val conversation = Conversation().after(
            """{"type":"responding","turn_id":"t1"}""",
            """{"type":"error","context":"sts","error":"the session dropped"}""",
        )

        assertEquals(Conversation.State.Idle, conversation.state)
    }

    @Test
    fun `the model failing while skills work ends the turn`() {
        val conversation = Conversation().after(
            """{"type":"delegated","task_id":"k1","skill":"lookup_order","turn_id":"t1"}""",
            """{"type":"error","context":"llm","error":"the model is overloaded"}""",
        )

        assertEquals(Conversation.State.Idle, conversation.state)
        assertEquals("the model is overloaded", conversation.failure)
    }

    @Test
    fun `a failure elsewhere leaves the reply going`() {
        val responding = Conversation().after(
            """{"type":"responding","turn_id":"t1"}""",
            """{"type":"error","context":"tool","error":"tool_result output is not a string or parts"}""",
            """{"type":"error","context":"tts","error":"the voice is unknown"}""",
        )
        val working = Conversation().after(
            """{"type":"delegated","task_id":"k1","skill":"think"}""",
            """{"type":"error","context":"command","error":"unknown command \"wave\""}""",
        )

        assertEquals(Conversation.State.Responding, responding.state)
        assertEquals("the voice is unknown", responding.failure)
        assertEquals(Conversation.State.Working(listOf("think")), working.state)
    }

    @Test
    fun `a failure after the conversation ended leaves it ended`() {
        val conversation = Conversation().after(
            """{"type":"left","at":"2026-09-02T17:05:00Z"}""",
            """{"type":"error","context":"llm","error":"agent: not joined"}""",
        )

        assertEquals(Conversation.State.Ended, conversation.state)
        assertEquals("agent: not joined", conversation.failure)
    }

    @Test
    fun `the turn after a failure runs as any other and clears the failure`() {
        val conversation = Conversation().after(
            """{"type":"responding","turn_id":"t1"}""",
            """{"type":"error","context":"llm","error":"the model is overloaded"}""",
            """{"type":"responding","turn_id":"t2"}""",
        )
        assertEquals(Conversation.State.Responding, conversation.state)
        assertNull(conversation.failure)

        val answered = conversation.after("""{"type":"responded","turn_id":"t2","text":"Here it is."}""")
        assertEquals(Conversation.State.Idle, answered.state)
        assertEquals(listOf("", "Here it is."), answered.turns.map { it.text })
    }

    @Test
    fun `a failure lasts until the agent hears, answers or speaks`() {
        val failed = Conversation().after("""{"type":"error","context":"stt","error":"the transcriber is gone"}""")

        val stillFailed = failed.after(
            """{"type":"hearing","participant":{"id":"p1","user_id":"u1","name":""},"text":"wh","language":"en"}""",
            """{"type":"response_delta","turn_id":"t1","text":"Hi"}""",
            """{"type":"responded","turn_id":"t1","text":"Hi"}""",
            """{"type":"interrupted","turn_id":"t1"}""",
        )
        assertEquals("the transcriber is gone", stillFailed.failure)
    }

    @Test
    fun `hearing someone clears the failure`() {
        val conversation = Conversation().after(
            """{"type":"error","context":"stt","error":"the transcriber is gone"}""",
            """{"type":"heard","participant":{"id":"p1","user_id":"u1","name":""},"text":"hello","language":"en"}""",
        )

        assertNull(conversation.failure)
        assertEquals(listOf("hello"), conversation.turns.map { it.text })
    }

    @Test
    fun `starting to answer clears the failure`() {
        val conversation = Conversation().after(
            """{"type":"error","context":"llm","error":"the model is overloaded"}""",
            """{"type":"responding","turn_id":"t2","participant":{"id":"","user_id":"","name":""},"prompt":"hi"}""",
        )

        assertNull(conversation.failure)
        assertEquals(Conversation.State.Responding, conversation.state)
    }

    @Test
    fun `speaking clears the failure`() {
        val conversation = Conversation().after(
            """{"type":"responding","turn_id":"t1"}""",
            """{"type":"error","context":"tts","error":"the voice is unknown"}""",
            """{"type":"spoke","turn_id":"t1","synthesis_id":"s1","characters":5,"audio_duration_ms":400,"time_to_first_byte_ms":90}""",
        )

        assertNull(conversation.failure)
        assertEquals(Conversation.State.Responding, conversation.state)
    }

    @Test
    fun `an interruption leaves the agent waiting to be spoken to`() {
        val conversation = Conversation().after(
            """{"type":"responding","turn_id":"t1"}""",
            """{"type":"interrupted","turn_id":"t1","participant":{"id":"p1","user_id":"u1","name":""}}""",
        )

        assertEquals(Conversation.State.Idle, conversation.state)
    }

    @Test
    fun `an event this SDK has never heard of changes nothing`() {
        val before = Conversation().after("""{"type":"responding","turn_id":"t1"}""")

        val after = before.after("""{"type":"astonished","turn_id":"t1","degree":11}""")

        assertEquals(before, after)
        assertNull(event("""{"type":"astonished"}""").kind)
    }
}
