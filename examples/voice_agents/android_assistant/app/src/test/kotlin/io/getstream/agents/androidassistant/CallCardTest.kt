package io.getstream.agents.androidassistant

import io.getstream.agents.android.core.AgentEvent
import io.getstream.agents.android.core.Conversation
import io.getstream.agents.android.core.TurnUI
import io.getstream.chat.android.ai.compose.generativeui.AIUIDocument
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame

/** Frames as the router's session socket writes them. */
private fun Conversation.after(vararg frames: String): Conversation =
    frames.fold(this) { conversation, frame -> conversation.reduce(requireNotNull(AgentEvent.decode(frame))) }

/** The chat's session with a card on screen, as it is when voice mode opens. */
private val chat = Conversation().after(
    """{"type":"responding","turn_id":"t1"}""",
    """{"type":"ui_completed","turn_id":"t1","format":"openui","content":"${"$"}party = 2\nroot = Card([hdr])\nhdr = CardHeader(\"Kaneo\")"}""",
    """{"type":"responded","turn_id":"t1","text":"Kaneo has a table."}""",
)

class CallCardTest {
    private val before = chat.turns.mapTo(HashSet()) { it.id }

    @Test
    fun `a change on the call to the chat's card shows the card changed`() {
        val conversation = chat.after(
            """{"type":"responding","turn_id":"t2"}""",
            """{"type":"ui_completed","turn_id":"t2","format":"openui","mode":"patch","content":"${"$"}party = 4"}""",
        )

        val card = assertNotNull(callCard(conversation, before) { null })

        assertEquals("t2", card.turn.id)
        assertEquals("\$party = 4\nroot = Card([hdr])\nhdr = CardHeader(\"Kaneo\")", card.turn.ui?.content)
    }

    @Test
    fun `a new card on the call shows as written`() {
        val conversation = chat.after(
            """{"type":"responding","turn_id":"t2"}""",
            """{"type":"ui_completed","turn_id":"t2","format":"openui","content":"root = Card([hdr])\nhdr = CardHeader(\"Mira\")"}""",
        )

        val card = assertNotNull(callCard(conversation, before) { null })

        assertEquals("root = Card([hdr])\nhdr = CardHeader(\"Mira\")", card.turn.ui?.content)
        assertNull(card.stored)
    }

    @Test
    fun `the chat's own card is not the call's`() {
        val conversation = chat.after("""{"type":"responding","turn_id":"t2"}""")

        assertNull(callCard(conversation, before) { null })
    }

    @Test
    fun `a change to a card only Stream Chat has shows Chat's merged copy`() {
        // As after a reload: the chat's card is only in Stream Chat, so the socket has nothing to
        // merge the change onto, and the router's merged copy on the reply is the one to show.
        val conversation = Conversation().after(
            """{"type":"responding","turn_id":"t2"}""",
            """{"type":"ui_completed","turn_id":"t2","format":"openui","mode":"patch","content":"${"$"}party = 4"}""",
        )
        val merged = TurnUI("openui", isStreaming = false, content = "\$party = 4\nroot = Card([hdr])\nhdr = CardHeader(\"Kaneo\")")
        val stored = assertNotNull(AIUIDocument.from(mapOf(AIUIDocument.CUSTOM_DATA_KEY to merged.customData)))

        val card = assertNotNull(callCard(conversation, emptySet()) { turnId -> stored.takeIf { turnId == "t2" } })

        assertEquals("t2", card.turn.id)
        assertSame(stored, card.document)
    }

    @Test
    fun `a change Stream Chat stored with no card under it is no card`() {
        // The router stores the patch as written when it had no card to merge it onto.
        val conversation = Conversation().after(
            """{"type":"responding","turn_id":"t2"}""",
            """{"type":"ui_completed","turn_id":"t2","format":"openui","mode":"patch","content":"${"$"}party = 4"}""",
        )
        val patch = TurnUI("openui", isStreaming = false, content = "\$party = 4")
        val stored = assertNotNull(AIUIDocument.from(mapOf(AIUIDocument.CUSTOM_DATA_KEY to patch.customData)))

        assertNull(callCard(conversation, emptySet()) { stored })
    }

    @Test
    fun `a change Stream Chat stored that empties the root is no card`() {
        // The router stores the patch as written when the card under it is not OpenUI, so a patch
        // that sets the root to null must not hide the older card Stream Chat has.
        val conversation = Conversation().after(
            """{"type":"responding","turn_id":"t2"}""",
            """{"type":"responded","turn_id":"t2","text":"Kaneo has a table."}""",
            """{"type":"responding","turn_id":"t3"}""",
            """{"type":"ui_completed","turn_id":"t3","format":"openui","mode":"patch","content":"root = null"}""",
        )
        val card = TurnUI("openui", isStreaming = false, content = "root = Card([hdr])\nhdr = CardHeader(\"Kaneo\")")
        val patch = TurnUI("openui", isStreaming = false, content = "root = null")
        val storedCard = assertNotNull(AIUIDocument.from(mapOf(AIUIDocument.CUSTOM_DATA_KEY to card.customData)))
        val storedPatch = assertNotNull(AIUIDocument.from(mapOf(AIUIDocument.CUSTOM_DATA_KEY to patch.customData)))

        assertNull(callCard(conversation, setOf("t2")) { storedPatch })
        val shown = assertNotNull(callCard(conversation, emptySet()) { turnId -> if (turnId == "t2") storedCard else storedPatch })
        assertEquals("t2", shown.turn.id)
        assertSame(storedCard, shown.document)
    }
}
