package io.getstream.agents.android.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The UI a reply's turn streamed on the socket, shown on its message in Stream Chat until the
 * message's own `ai_ui` settles.
 */
class LiveUITest {
    private fun event(json: String): AgentEvent = requireNotNull(AgentEvent.decode(json))

    private fun Conversation.after(vararg frames: String): Conversation =
        frames.fold(this) { conversation, frame -> conversation.reduce(event(frame)) }

    private fun streaming(): Conversation = Conversation().after(
        """{"type":"responding","turn_id":"t1"}""",
        """{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"root = Card([hdr])\nhdr = CardHeader(\"Kaneo\")\n"}""",
    )

    private val streamedUi: Map<String, Any?> = mapOf(
        "v" to 1,
        "format" to "openui",
        "status" to "streaming",
        "content" to "root = Card([hdr])\nhdr = CardHeader(\"Kaneo\")",
    )

    private val live = LiveUI()

    @Test
    fun `a reply being written shows the socket's UI for its turn`() {
        assertTrue(live.observe(streaming()))
        assertFalse(live.observe(streaming()))

        val reply = mapOf(
            "ai_generated" to true,
            "generating" to true,
            "support_runtime" to mapOf("role" to "assistant", "turn_id" to "t1"),
        )
        val overlaid = live.overlayCustom(listOf("question" to emptyMap(), "m1" to reply))

        assertEquals(setOf("m1"), overlaid.keys)
        assertEquals(streamedUi, overlaid["m1"]?.get("ai_ui"))
        assertEquals(true, overlaid["m1"]?.get("generating"))
    }

    @Test
    fun `a reply whose own UI is complete shows its own`() {
        live.observe(streaming())
        val settled = mapOf(
            "ai_generated" to true,
            "generating" to true,
            "turn_id" to "t1",
            "ai_ui" to mapOf(
                "v" to 1,
                "format" to "openui",
                "status" to "complete",
                "content" to "root = Card([hdr])\nhdr = CardHeader(\"Kaneo Cliffside Kitchen\")",
            ),
        )

        assertTrue(live.overlayCustom(listOf("m1" to settled)).isEmpty())
    }

    @Test
    fun `a reply still streaming its own UI shows the socket's`() {
        live.observe(streaming())
        val behind = mapOf(
            "ai_generated" to true,
            "generating" to true,
            "turn_id" to "t1",
            "ai_ui" to mapOf("v" to 1, "format" to "openui", "status" to "streaming", "content" to "root = Card([hdr])"),
        )

        assertEquals(streamedUi, live.overlayCustom(listOf("m1" to behind))["m1"]?.get("ai_ui"))
    }

    @Test
    fun `the newest reply being written takes the latest UI before its turn is known`() {
        live.observe(streaming())
        val earlier = mapOf("ai_generated" to true, "generating" to false)
        val newest = mapOf("ai_generated" to true, "generating" to true)

        val overlaid = live.overlayCustom(listOf("m0" to earlier, "m1" to newest))

        assertEquals(setOf("m1"), overlaid.keys)
        assertEquals("root = Card([hdr])\nhdr = CardHeader(\"Kaneo\")", live.latest?.content)
        assertEquals(live.latest, live["t1"])
    }

    @Test
    fun `a new reply never shows the UI of the reply before it`() {
        val conversation = streaming()
            .said("and for four people?")
            .after("""{"type":"responding","turn_id":"t2"}""")
        live.observe(conversation)
        val named = mapOf(
            "ai_generated" to true,
            "generating" to true,
            "support_runtime" to mapOf("role" to "assistant", "turn_id" to "t2"),
        )
        val unnamed = mapOf("ai_generated" to true, "generating" to true)

        assertTrue(live.overlayCustom(listOf("m2" to named)).isEmpty())
        assertTrue(live.overlayCustom(listOf("m2" to unnamed)).isEmpty())
    }

    @Test
    fun `a UI with nothing finished leaves the reply its text`() {
        live.observe(
            Conversation().after("""{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"root = Card(["}"""),
        )
        val reply = mapOf("ai_generated" to true, "generating" to true, "turn_id" to "t1")

        assertTrue(live.overlayCustom(listOf("m1" to reply)).isEmpty())
    }

    @Test
    fun `a delta that finishes no statement changes nothing shown`() {
        var conversation = streaming()
        live.observe(conversation)

        conversation = conversation.after(
            """{"type":"ui_delta","turn_id":"t1","format":"openui","delta":"tip = Callout(\"info\""}""",
        )
        val unfinished = live.observe(conversation)
        conversation = conversation.after(
            """{"type":"ui_delta","turn_id":"t1","format":"openui","delta":", \"Book\", \"Terrace\")\n"}""",
        )
        val finished = live.observe(conversation)

        assertFalse(unfinished)
        assertTrue(finished)
    }

    @Test
    fun `a reply Chat says is written shows what Chat stored`() {
        live.observe(streaming())
        // Such as a refused reply, which Chat keeps with no UI.
        val written = mapOf("ai_generated" to true, "generating" to false, "turn_id" to "t1")

        assertTrue(live.overlayCustom(listOf("m1" to written)).isEmpty())
    }

    @Test
    fun `the person's messages are left alone`() {
        live.observe(streaming())

        assertTrue(live.overlayCustom(listOf("q1" to mapOf("turn_id" to "t1", "generating" to true))).isEmpty())
    }
}
