package io.getstream.agents.androidassistant

import io.getstream.agents.android.core.AgentEvent
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** Frames as the router's session socket writes them. */
private fun event(json: String): AgentEvent = requireNotNull(AgentEvent.decode(json))

class UnansweredTest {
    @Test
    fun `a reply the model could not write leaves the question unanswered, without the raw error`() {
        val unanswered = null.unansweredAfter(
            event("""{"type":"error","context":"llm","error":"gemini: 402 Payment Required"}"""),
        )

        assertEquals(UNANSWERED, unanswered)
    }

    @Test
    fun `a failed reply that says nothing still leaves the question unanswered`() {
        val unanswered = null.unansweredAfter(event("""{"type":"error","context":"llm","error":""}"""))

        assertEquals(UNANSWERED, unanswered)
    }

    @Test
    fun `a failure outside the reply does not mark the question unanswered`() {
        val unanswered = null.unansweredAfter(
            event("""{"type":"error","context":"compaction","error":"the summary was too long"}"""),
        )

        assertNull(unanswered)
    }

    @Test
    fun `the reply that ends a failed turn keeps why it failed`() {
        // The router reports the failure and then ends the turn with an empty reply.
        val unanswered = UNANSWERED
            .unansweredAfter(event("""{"type":"responded","turn_id":"t1","text":"","pending_work":false}"""))

        assertEquals(UNANSWERED, unanswered)
    }

    @Test
    fun `the next reply starting clears the failure`() {
        val unanswered = UNANSWERED
            .unansweredAfter(event("""{"type":"responding","turn_id":"t2","participant":null,"prompt":"hi"}"""))

        assertNull(unanswered)
    }
}
