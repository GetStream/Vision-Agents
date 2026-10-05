package io.getstream.visionagents.core

import java.util.concurrent.CopyOnWriteArrayList
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

private const val SESSION = """{"id":"s1","call_id":"","call_type":"default","user_id":"u1","agent_id":"a1","state":"live","modality":"text","created_at":"2026-09-24T04:03:19Z","text":true}"""
private const val KEPT = """{"id":"s1","call_id":"","call_type":"default","user_id":"u1","agent_id":"a1","state":"live","modality":"text","created_at":"2026-09-24T04:03:19Z","text":true,"conversation_id":"messaging:agent-s1"}"""
private const val ASKED = """{"id":"r1","session_id":"s1","status":"running","created_at":"2026-09-24T04:03:19Z"}"""

/** Socket tests run in real time, because what they wait for is a real connection. */
class AgentSessionTest {
    private val router = TestRouter().apply {
        answer = { if (it.path.endsWith("/responses")) Reply(202, ASKED) else Reply(201, SESSION) }
    }
    private val agents = VisionAgents(url = router.url, customerId = "acme")

    @AfterTest
    fun stop() {
        agents.close()
        router.close()
    }

    @Test
    fun `the handshake carries the device's headers and no credential in the url`() = runBlocking {
        agents.setUser(User("jlahey"), "unused")
        val chat = agents.chat()

        val handshake = router.handshakes.single()
        assertEquals("/v1/agents/sessions/s1/events", handshake.path)
        assertEquals("jwt", handshake.header("Stream-Auth-Type"))
        assertEquals("acme", handshake.header("X-Customer-Id"))
        assertEquals("jlahey", handshake.header("X-Stream-User-Id"))
        assertEquals(mapOf("decisions" to listOf("false")), handshake.query)
        assertTrue(chat.isConnected.value)
        chat.close()
    }

    @Test
    fun `what is typed shows at once and the reply streams into the same transcript`() = runBlocking {
        val chat = agents.agent("docs").chat()

        val asked = chat.responses.create("What are your hours?")
        val request = router.requests.last()
        assertEquals("POST /v1/agents/sessions/s1/responses", "${request.method} ${request.path}")
        assertEquals("""{"text":"What are your hours?"}""", request.body)
        assertEquals("r1", asked.id)
        assertEquals(listOf("What are your hours?"), chat.conversation.value.turns.map { it.text })
        router.script.send(Script.Send("""{"type":"responding","turn_id":"t1","prompt":"What are your hours?"}"""))
        router.script.send(Script.Send("""{"type":"response_delta","turn_id":"t1","text":"Nine "}"""))
        router.script.send(Script.Send("""{"type":"response_delta","turn_id":"t1","text":"to five."}"""))
        router.script.send(Script.Send("""{"pending_work":false,"type":"responded","turn_id":"t1","text":"Nine to five.","time_to_first_token_ms":80}"""))

        val done = chat.conversation.first { it.state == Conversation.State.Idle && it.turns.size == 2 }
        assertEquals(listOf("What are your hours?", "Nine to five."), done.turns.map { it.text })
        assertTrue(done.turns.last().isAgent)
        chat.close()
    }

    @Test
    fun `a question in a conversation kept in chat is a command, unless it shows the agent something`() = runBlocking {
        router.answer = { if (it.path.endsWith("/responses")) Reply(202, ASKED) else Reply(201, KEPT) }
        val chat = agents.chat()

        chat.responses.create("first")
        chat.responses.create("what is this", listOf(ImageSource("https://x/y.png")))

        val (asked, shown) = router.requests.filter { it.path.endsWith("/responses") }
            .map { Json.parseToJsonElement(it.body).jsonObject }
        assertTrue(Regex("[A-Za-z0-9_-]{1,128}").matches(asked["command_id"]!!.jsonPrimitive.content))
        assertNull(shown["command_id"])
        chat.close()
    }

    @Test
    fun `a tool call is answered with nobody collecting events, repeating its turn and command`() = runBlocking {
        val asked = CopyOnWriteArrayList<String>()
        val lookup = AgentTool(
            "lookup_order",
            "Look up an order.",
            AgentTool.strings(mapOf("order_id" to "the order")),
            executor = AgentTool.Executor.Client,
            displayTitle = "Looking up your order",
        ) {
            asked += it["order_id"]!!.jsonPrimitive.content
            "Order A-1042: two wool throws."
        }
        val chat = agents.chat(SessionOptions(tools = listOf(lookup)))

        val declared = Json.parseToJsonElement(router.requests.single().body).jsonObject["tools"].toString()
        assertEquals(
            """[{"description":"Look up an order.","name":"lookup_order","display_title":"Looking up your order",""" +
                """"executor":"client","parameters":{"type":"object","properties":{"order_id":{"type":"string","description":"the order"}},"required":[]}}]""",
            declared,
        )

        router.script.send(
            Script.Send("""{"type":"tool_call","id":"c1","name":"lookup_order","arguments":"{\"order_id\":\"A-1042\"}","command_id":"","turn_id":"t1"}"""),
        )

        val answer = Json.parseToJsonElement(router.nextSent()).jsonObject
        assertEquals(JsonPrimitive("tool_result"), answer["type"])
        assertEquals(JsonPrimitive("c1"), answer["tool_call_id"])
        assertEquals(JsonPrimitive("Order A-1042: two wool throws."), answer["output"])
        assertEquals(JsonPrimitive("t1"), answer["turn_id"])
        assertEquals(listOf("A-1042"), asked)
        chat.close()
    }

    @Test
    fun `a tool that throws tells the model it did not work, since it is mid-sentence waiting`() = runBlocking {
        val broken = AgentTool("lookup_order", "Look up an order.") { error("the orders database is down") }
        val chat = agents.chat(SessionOptions(tools = listOf(broken)))

        router.script.send(Script.Send("""{"type":"tool_call","id":"c1","name":"lookup_order","arguments":"","command_id":"","turn_id":""}"""))

        val answer = Json.parseToJsonElement(router.nextSent()).jsonObject
        assertEquals(JsonPrimitive("the orders database is down"), answer["error"])
        assertEquals(JsonPrimitive(""), answer["output"])
        chat.close()
    }

    @Test
    fun `a tool this device never declared is left for whoever owns it`() = runBlocking {
        val chat = agents.chat()

        router.script.send(Script.Send("""{"type":"tool_call","id":"c1","name":"somebody_elses","arguments":"{}"}"""))
        chat.say("still there?")

        assertEquals("""{"type":"say","text":"still there?"}""", router.nextSent())
        chat.close()
    }

    @Test
    fun `a cancelled tool call stops the tool rather than answering it`() = runBlocking {
        val started = CompletableDeferred<Unit>()
        val stopped = CompletableDeferred<Unit>()
        val slow = AgentTool("slow", "Takes forever.") {
            started.complete(Unit)
            try {
                awaitCancellation()
            } finally {
                stopped.complete(Unit)
            }
        }
        val chat = agents.chat(SessionOptions(tools = listOf(slow)))

        router.script.send(Script.Send("""{"type":"tool_call","id":"c1","name":"slow","arguments":"{}"}"""))
        started.await()
        router.script.send(Script.Send("""{"type":"tool_cancel","id":"c1","command_id":"","turn_id":""}"""))
        stopped.await()
        chat.interrupt()

        assertEquals("""{"type":"interrupt"}""", router.nextSent())
        chat.close()
    }

    @Test
    fun `a frame that cannot be read is skipped and the conversation carries on`() = runBlocking {
        val chat = agents.chat()
        val seen = async(start = CoroutineStart.UNDISPATCHED) { chat.events().first { it.type == "astonished" } }
        until { router.handshakes.isNotEmpty() }

        router.script.send(Script.Send("not json"))
        router.script.send(Script.Send("""{"type":"astonished","degree":11}"""))

        assertEquals(JsonPrimitive(11), seen.await()["degree"])
        assertTrue(chat.isConnected.value)
        chat.close()
    }

    @Test
    fun `a session the router ends closes the socket without a failure`() = runBlocking {
        val chat = agents.chat()

        router.script.send(Script.Send("""{"type":"left","at":"2026-09-24T04:03:20Z"}"""))
        router.script.send(Script.Close(1000, "the session ended"))

        until { !chat.isConnected.value }
        assertNull(chat.failure.value)
        assertEquals(Conversation.State.Ended, chat.conversation.value.state)
    }

    @Test
    fun `a socket that closes for any other reason says why`() = runBlocking {
        val chat = agents.chat()

        router.script.send(Script.Close(1011, "internal error"))

        until { chat.failure.value != null }
        val failure = assertIs<AgentsException.SocketClosed>(chat.failure.value)
        assertEquals(1011, failure.code)
        assertEquals("internal error", failure.reason)
        assertFalse(chat.isConnected.value)
    }

    @Test
    fun `closing tells the router to end the session and is safe twice`() = runBlocking {
        val chat = agents.chat()

        chat.close()
        chat.close()

        assertEquals("""{"type":"close"}""", router.nextSent())
        assertFalse(chat.isConnected.value)
        assertNull(chat.failure.value)
        assertEquals(Conversation.State.Ended, chat.conversation.value.state)
    }

    @Test
    fun `deleting asks the router to delete the session and ends the conversation here`() = runBlocking {
        router.answer = { if (it.method == "DELETE") Reply(204) else Reply(201, SESSION) }
        val chat = agents.chat()

        chat.delete()

        assertEquals("DELETE /v1/agents/sessions/s1", router.requests.last().let { "${it.method} ${it.path}" })
        assertFalse(chat.isConnected.value)
        assertEquals(Conversation.State.Ended, chat.conversation.value.state)
    }

    @Test
    fun `updating renames the session and the handle holds the new one`() = runBlocking {
        val renamed = SESSION.replace("\"text\":true", "\"text\":true,\"title\":\"Hours\"")
        router.answer = { if (it.method == "PATCH") Reply(200, renamed) else Reply(201, SESSION) }
        val chat = agents.chat()

        val updated = chat.update(title = "Hours")

        val request = router.requests.last()
        assertEquals("PATCH /v1/agents/sessions/s1", "${request.method} ${request.path}")
        assertEquals("""{"title":"Hours"}""", request.body)
        assertEquals("Hours", updated.title)
        assertEquals("Hours", chat.session.title)
        chat.close()
    }

    @Test
    fun `a tool declared twice is refused before a session is opened`() = runBlocking {
        val tool = AgentTool("t", "a tool") { "" }

        assertFailsWith<AgentsException.Configuration> { agents.chat(SessionOptions(tools = listOf(tool, tool))) }
        assertTrue(router.requests.isEmpty())
    }

    @Test
    fun `attaching follows a session that is already open`() = runBlocking {
        router.answer = { Reply(200, SESSION) }

        val chat = agents.attach("s1")

        assertEquals("GET /v1/agents/sessions/s1", router.requests.single().let { "${it.method} ${it.path}" })
        until { router.handshakes.isNotEmpty() }
        assertEquals("s1", chat.id)
        chat.close()
    }
}
