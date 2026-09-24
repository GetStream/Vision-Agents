package io.getstream.visionagents.core

import java.util.concurrent.CopyOnWriteArrayList
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.flow.first

/**
 * Tests that need a router running.
 *
 * The Kotlin answer to `@pytest.mark.integration`: skipped unless `VISION_AGENTS_URL` is set, so
 * the ordinary test run stays offline and fast.
 *
 *     VISION_AGENTS_URL=http://localhost:8080 VISION_AGENTS_CUSTOMER_ID=examples ./gradlew :core:test
 *
 * `VISION_AGENTS_AGENT` names an agent config to talk to; without one the router's defaults answer.
 */
@EnabledIfEnvironmentVariable(named = "VISION_AGENTS_URL", matches = ".+")
class LiveTest {
    private val agents = VisionAgents(
        url = System.getenv("VISION_AGENTS_URL"),
        customerId = System.getenv("VISION_AGENTS_CUSTOMER_ID") ?: "acme",
    )
    private val options = SessionOptions(agent = System.getenv("VISION_AGENTS_AGENT")?.ifEmpty { null })

    @AfterTest
    fun stop() {
        agents.close()
    }

    @Test
    fun `a text session joins no call and opens a socket`() = runBlocking {
        val chat = agents.chat(options)

        assertTrue(chat.session.isText)
        assertEquals("", chat.session.callId)
        assertEquals(Session.State.Live, chat.session.state)
        assertTrue(chat.isConnected.value)
        assertEquals(chat.id, agents.sessions.get(chat.id).id)
        chat.close()
    }

    @Test
    fun `asking something gets an answer, streamed into the transcript`() = runBlocking {
        val chat = agents.chat(options)

        chat.send("What is two plus two? Answer in one short sentence.")

        val done = withTimeout(30.seconds) {
            chat.conversation.first { it.state == Conversation.State.Idle && it.turns.size >= 2 }
        }
        assertTrue(done.turns.last().isAgent)
        assertTrue(done.turns.last().text.isNotEmpty())
        assertNull(chat.failure.value)
        chat.close()
    }

    @Test
    fun `a tool on this side is called and answered`() = runBlocking {
        val asked = CopyOnWriteArrayList<String>()
        val lookup = AgentTool(
            name = "lookup_order",
            description = "Look up one of the caller's orders by its order number.",
            parameters = AgentTool.strings(mapOf("order_id" to "the order number"), required = listOf("order_id")),
        ) { arguments ->
            asked += arguments["order_id"]?.jsonPrimitive?.content.orEmpty()
            "Order A-1042: 2 wool throws, 78.00, delivered 14 August, unopened."
        }
        val chat = agents.chat(options.copy(tools = listOf(lookup)))

        chat.send("Use the lookup_order tool to look up order A-1042 and tell me what is in it.")

        until(45) { asked.isNotEmpty() }
        assertEquals("A-1042", asked.first().uppercase())
        chat.close()
    }

    @Test
    fun `a rewound session carries on from the response kept, and a fork branches off it`() = runBlocking {
        val chat = agents.chat(options)

        chat.send("My name is Ada. Reply with one word.")
        withTimeout(30.seconds) { chat.conversation.first { it.state == Conversation.State.Idle && it.turns.size >= 2 } }
        chat.send("What is my name? Reply with one word.")
        withTimeout(30.seconds) { chat.conversation.first { it.state == Conversation.State.Idle && it.turns.size >= 4 } }
        until(10) { chat.responses.list().size == 2 }

        val kept = chat.responses.list().first()
        assertTrue(kept.said.contains("Ada"))
        assertTrue(chat.responses.items(kept.id).isNotEmpty())

        chat.responses.rewind(kept)
        assertEquals(listOf(kept.id), chat.responses.list().map { it.id })

        val fork = chat.fork(ForkOptions(responseId = kept.id))
        assertNotEquals(chat.id, fork.id)
        assertEquals(chat.id, fork.forkedFrom)
        agents.sessions.close(fork.id)
        chat.close()
    }

    @Test
    fun `a response created over http is a handle on a turn the socket then streams`() = runBlocking {
        val chat = agents.chat(options.copy(title = "kotlin live ${System.nanoTime()}"))

        val response = chat.responses.create("Say hello in one word.")

        assertEquals(chat.id, response.sessionId)
        withTimeout(30.seconds) { chat.conversation.first { it.turns.any { turn -> turn.isAgent && turn.text.isNotEmpty() } } }
        until(15) { chat.responses.list().any { it.id == response.id && it.status != AgentResponse.Status.Running } }
        chat.close()
    }

    @Test
    fun `a titled conversation is found by its title`() = runBlocking {
        val title = "kotlin search ${System.nanoTime()}"
        val session = agents.sessions.create(options.copy(title = title))

        val found = agents.sessions.search(title)

        assertTrue(found.any { it.id == session.id })
        assertTrue(agents.sessions.query(SessionQuery(limit = 50)).any { it.id == session.id })
        agents.sessions.close(session.id)
    }

    @Test
    fun `a guest can hold a conversation of their own`() = runBlocking {
        val guest = agents.guestUser(GuestOptions(name = "Kotlin guest"))
        assertTrue(guest.id.isNotEmpty() && guest.token.isNotEmpty())

        agents.setUser(guest)
        val session = agents.sessions.create(options)

        assertTrue(agents.sessions.query().any { it.id == session.id })
        agents.sessions.close(session.id)
    }
}
