package io.getstream.visionagents.android.core

import java.util.concurrent.CopyOnWriteArrayList
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Clock
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

private const val SESSION = """{"id":"s1","call_id":"","call_type":"default","user_id":"u1","agent_id":"a1","state":"live","modality":"text","created_at":"2026-09-24T04:03:19Z","text":true}"""
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
        val body = Json.parseToJsonElement(request.body).jsonObject
        assertEquals(setOf("text", "request_id"), body.keys)
        assertEquals(JsonPrimitive("What are your hours?"), body["text"])
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
    fun `every question carries a fresh request id, one showing the agent something too`() = runBlocking {
        val chat = agents.chat()

        chat.responses.create("first")
        chat.responses.create("first")
        chat.responses.create("what is this", listOf(ImageSource("data:image/png;base64,AAAA")))

        val bodies = router.requests.filter { it.path.endsWith("/responses") }
            .map { Json.parseToJsonElement(it.body).jsonObject }
        val ids = bodies.map { it["request_id"]!!.jsonPrimitive.content }
        assertTrue(ids.all { Regex("[A-Za-z0-9_-]{1,128}").matches(it) })
        assertEquals(3, ids.toSet().size)
        chat.close()
    }

    @Test
    fun `one answer can be capped and told how long to think, still as a text question`() = runBlocking {
        val chat = agents.chat()

        chat.responses.create("think hard", maxOutputTokens = 512, reasoningEffort = "high")

        val body = Json.parseToJsonElement(router.requests.last { it.path.endsWith("/responses") }.body).jsonObject
        assertEquals(setOf("text", "request_id", "max_output_tokens", "reasoning_effort"), body.keys)
        assertEquals(JsonPrimitive(512), body["max_output_tokens"])
        assertEquals(JsonPrimitive("high"), body["reasoning_effort"])
        chat.close()
    }

    @Test
    fun `a tool call is answered with nobody collecting events, repeating its turn and request`() = runBlocking {
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
            Script.Send("""{"type":"tool_call","id":"c1","name":"lookup_order","arguments":"{\"order_id\":\"A-1042\"}","request_id":"r1","turn_id":"t1"}"""),
        )

        val answer = Json.parseToJsonElement(router.nextSent()).jsonObject
        assertEquals(JsonPrimitive("tool_result"), answer["type"])
        assertEquals(JsonPrimitive("c1"), answer["tool_call_id"])
        assertEquals(JsonPrimitive("Order A-1042: two wool throws."), answer["output"])
        assertEquals(JsonPrimitive("r1"), answer["request_id"])
        assertEquals(JsonPrimitive("t1"), answer["turn_id"])
        assertEquals(listOf("A-1042"), asked)
        chat.close()
    }

    @Test
    fun `a tool that throws tells the model it did not work, since it is mid-sentence waiting`() = runBlocking {
        val broken = AgentTool("lookup_order", "Look up an order.") { error("the orders database is down") }
        val chat = agents.chat(SessionOptions(tools = listOf(broken)))

        router.script.send(Script.Send("""{"type":"tool_call","id":"c1","name":"lookup_order","arguments":"","request_id":"","turn_id":""}"""))

        val answer = Json.parseToJsonElement(router.nextSent()).jsonObject
        assertEquals(JsonPrimitive("the orders database is down"), answer["error"])
        assertEquals(JsonPrimitive(""), answer["output"])
        chat.close()
    }

    @Test
    fun `a tool's pictures go back beside its words`() = runBlocking {
        val photograph = AgentTool.withPictures("photograph_room", "Photograph the room.") {
            ToolOutput("the kitchen", listOf(ImageSource("data:image/png;base64,iVBORw0KGgo=")))
        }
        val chat = agents.chat(SessionOptions(tools = listOf(photograph)))

        router.script.send(
            Script.Send("""{"type":"tool_call","id":"c1","name":"photograph_room","arguments":"{}","request_id":"r1","turn_id":"t1"}"""),
        )

        val answer = Json.parseToJsonElement(router.nextSent()).jsonObject
        assertEquals(JsonPrimitive("c1"), answer["tool_call_id"])
        assertEquals(
            Json.parseToJsonElement(
                """[{"type":"text","text":"the kitchen"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]""",
            ),
            answer["output"],
        )
        assertEquals(JsonPrimitive(""), answer["error"])
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
    fun `a request for frames with no camera on is left for the video worker that owns it`() = runBlocking {
        val voice = agents.chat()

        router.script.send(Script.Send("""{"type":"tool_call","id":"c1","name":"get_video_frames","arguments":"{}"}"""))
        voice.say("still there?")

        assertEquals("""{"type":"say","text":"still there?"}""", router.nextSent())
        voice.close()
    }

    @Test
    fun `a camera on a call answers the request to look with what it showed lately`() = runBlocking {
        val voice = agents.chat()
        val shown = Clock.System.now()
        voice.cameraFrames = CameraFrames().apply { record(byteArrayOf(0xff.toByte(), 0xd8.toByte(), 0xff.toByte()), shown) }

        router.script.send(
            Script.Send("""{"type":"tool_call","id":"c1","name":"get_video_frames","arguments":"{\"limit\":2}","request_id":"r1","turn_id":"t1"}"""),
        )

        val answer = Json.parseToJsonElement(router.nextSent()).jsonObject
        assertEquals(JsonPrimitive("c1"), answer["tool_call_id"])
        assertEquals(JsonPrimitive("r1"), answer["request_id"])
        assertEquals(JsonPrimitive(""), answer["error"])
        val parts = answer["output"]!!.jsonArray
        assertEquals(2, parts.size, "what the frame is, then the frame")
        assertEquals(
            Json.parseToJsonElement(
                """{"clock":"device_unix_ms","frames":[{"captured_at_ms":${shown.toEpochMilliseconds()},"frame_id":"camera-1"}],"source":"camera"}""",
            ),
            Json.parseToJsonElement(parts[0].jsonObject["text"]!!.jsonPrimitive.content),
        )
        assertEquals(
            JsonPrimitive("data:image/jpeg;base64,/9j/"),
            parts[1].jsonObject["image_url"]!!.jsonObject["url"],
        )
        voice.close()
    }

    @Test
    fun `a camera that showed nothing lately says so as the answer, not as a failure`() = runBlocking {
        val voice = agents.chat()
        voice.cameraFrames = CameraFrames().apply { record(byteArrayOf(1), Clock.System.now() - 60.seconds) }

        router.script.send(Script.Send("""{"type":"tool_call","id":"c1","name":"get_video_frames","arguments":"{}"}"""))

        val answer = Json.parseToJsonElement(router.nextSent()).jsonObject
        assertTrue(answer["output"]!!.jsonPrimitive.content.contains("point it at what they mean again"))
        assertEquals(JsonPrimitive(""), answer["error"])
        voice.close()
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
        router.script.send(Script.Send("""{"type":"tool_cancel","id":"c1","request_id":"","turn_id":""}"""))
        stopped.await()
        chat.interrupt()

        assertEquals("""{"type":"interrupt"}""", router.nextSent())
        chat.close()
    }

    private val refundCall =
        """{"type":"tool_call","id":"c1","name":"issue_refund","arguments":"{\"order_id\":\"A-1042\",\"reason\":\"it arrived unopened\"}","request_id":"m1","turn_id":"t1"}"""

    private fun refund(ran: MutableList<String>) = AgentTool(
        "issue_refund",
        "Refund an order.",
        approval = AgentTool.Approval("Refund order A-1042?", reasonArgument = "reason", allowTitle = "Refund"),
    ) {
        ran += it["order_id"]!!.jsonPrimitive.content
        "Refunded ${it["order_id"]!!.jsonPrimitive.content}"
    }

    @Test
    fun `a tool that asks first declares its question and waits for the person before it runs`() = runBlocking {
        val ran = CopyOnWriteArrayList<String>()
        val chat = agents.chat(SessionOptions(tools = listOf(refund(ran))))

        val declared = Json.parseToJsonElement(router.requests.single().body).jsonObject["tools"].toString()
        assertTrue(""""approval":{"title":"Refund order A-1042?","allow_title":"Refund","reason_argument":"reason"}""" in declared, declared)

        router.script.send(Script.Send(refundCall))
        val waiting = chat.approvals.first { it.isNotEmpty() }.single()
        assertEquals("Refund order A-1042?", waiting.approval.title)
        assertEquals("it arrived unopened", waiting.reason)
        assertEquals(emptyList(), ran)

        chat.decide("c1", allowed = true)

        assertEquals(emptyList(), chat.approvals.value)
        assertEquals(
            """{"type":"tool_approval","tool_call_id":"c1","allowed":true,"request_id":"m1","turn_id":"t1"}""",
            router.nextSent(),
        )
        val result = Json.parseToJsonElement(router.nextSent()).jsonObject
        assertEquals(JsonPrimitive("tool_result"), result["type"])
        assertEquals(JsonPrimitive("Refunded A-1042"), result["output"])
        assertEquals(JsonPrimitive("m1"), result["request_id"])
        assertEquals(listOf("A-1042"), ran)
        chat.close()
    }

    @Test
    fun `a declined call never runs, the model is told so, and answering again does nothing`() = runBlocking {
        val ran = CopyOnWriteArrayList<String>()
        val chat = agents.chat(SessionOptions(tools = listOf(refund(ran))))

        router.script.send(Script.Send(refundCall))
        chat.approvals.first { it.isNotEmpty() }
        chat.decide("c1", allowed = false, summary = "Kept the order")
        chat.decide("c1", allowed = true)

        assertEquals(
            """{"type":"tool_approval","tool_call_id":"c1","allowed":false,"summary":"Kept the order","request_id":"m1","turn_id":"t1"}""",
            router.nextSent(),
        )
        val result = Json.parseToJsonElement(router.nextSent()).jsonObject
        assertEquals(JsonPrimitive("tool_result"), result["type"])
        assertEquals(JsonPrimitive(""), result["output"])
        assertEquals(JsonPrimitive("The person declined this call."), result["error"])
        chat.say("still there?")
        assertEquals("""{"type":"say","text":"still there?"}""", router.nextSent())
        assertEquals(emptyList(), ran)
        chat.close()
    }

    @Test
    fun `declining a sign-in tells the router the call is not allowed`() = runBlocking {
        val chat = agents.chat(SessionOptions())

        chat.declineLogin("c9")

        assertEquals("""{"type":"tool_approval","tool_call_id":"c9","allowed":false}""", router.nextSent())
        chat.close()
    }

    @Test
    fun `a cancelled call stops waiting for the person`() = runBlocking {
        val chat = agents.chat(SessionOptions(tools = listOf(refund(CopyOnWriteArrayList()))))

        router.script.send(Script.Send(refundCall))
        chat.approvals.first { it.isNotEmpty() }
        router.script.send(Script.Send("""{"type":"tool_cancel","id":"c1","request_id":"m1","turn_id":"t1"}"""))

        chat.approvals.first { it.isEmpty() }
        chat.decide("c1", allowed = true)
        chat.say("still there?")
        assertEquals("""{"type":"say","text":"still there?"}""", router.nextSent())
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
    fun `a socket the router refuses to open says what the router said`() = runBlocking {
        router.refuseHandshake = Reply(
            403,
            envelope("permission", "forbidden", "not your session"),
            mapOf("X-Request-Id" to "req-3"),
        )

        val refused = assertFailsWith<AgentsException.Http> { agents.chat() }

        assertEquals(403, refused.status)
        assertEquals("permission", refused.type)
        assertEquals("forbidden", refused.code)
        assertEquals("not your session", refused.reason)
        assertEquals("https://getstream.io/agents/docs/api/errors/#forbidden", refused.docUrl)
        assertEquals("req-3", refused.requestId)
    }

    @Test
    fun `a socket refused for its credential is a status too, not a transport failure`() = runBlocking {
        router.refuseHandshake = Reply(401, envelope("authentication", "unauthenticated", "token expired"))

        val refused = assertFailsWith<AgentsException.Http> { agents.chat() }

        assertEquals(401, refused.status)
        assertEquals("unauthenticated", refused.code)
        assertEquals("token expired", refused.reason)
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
