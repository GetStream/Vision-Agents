package io.getstream.visionagents.core

import java.io.File
import java.nio.file.Files
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Instant
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put

/** A session as the router answers one: every required field and a handful of the optional ones. */
private fun sessionJson(id: String = "s1", state: String = "live", text: Boolean = true) =
    """{"id":"$id","call_id":"","call_type":"default","user_id":"u1","agent_id":"a1","state":"$state",""" +
        """"created_at":"2026-09-24T04:03:19.979138Z","text":$text,"agent":"docs","title":"Billing",""" +
        """"custom":{"tenant":"acme"},"closed_at":"2026-09-24T04:03:20Z","some_field_added_later":1}"""

private fun responseJson(id: String, status: String = "completed") =
    """{"id":"$id","session_id":"s1","said":"hi","status":"$status","created_at":"2026-09-24T04:03:19Z"}"""

private fun itemJson(response: String, ordinal: Int, kind: String = "answer") =
    """{"response_id":"$response","ordinal":$ordinal,"kind":"$kind","text":"t$ordinal","at":"2026-09-24T04:03:19.5Z"}"""

class VisionAgentsTest {
    private val router = TestRouter()
    private val agents = VisionAgents(url = router.url, customerId = "acme")

    @AfterTest
    fun stop() {
        agents.close()
        router.close()
    }

    @Test
    fun `every request says it comes from a device, even to a router with nothing in front of it`() = runTest {
        router.answer = { Reply(200, "[]") }

        agents.sessions.query()

        val arrived = router.requests.single()
        assertEquals("jwt", arrived.header("Stream-Auth-Type"))
        assertEquals("acme", arrived.header("X-Customer-Id"))
        assertNull(arrived.header("Authorization"))
        assertNull(arrived.header("X-Stream-User-Id"))
    }

    @Test
    fun `a user set on a router reached by customer id is named rather than proven`() = runTest {
        router.answer = { Reply(200, "[]") }
        agents.setUser(User("jlahey", "Jim Lahey"), "unused")

        agents.sessions.query()

        assertEquals("jlahey", router.requests.single().header("X-Stream-User-Id"))
    }

    @Test
    fun `behind a verifying deployment the token is the credential, asked for once`() = runTest {
        val asked = AtomicInteger()
        val keyed = VisionAgents(url = router.url, apiKey = "key-1")
        keyed.setUser(User("jlahey")) { "token-${asked.incrementAndGet()}" }
        router.answer = { Reply(200, "[]") }

        keyed.sessions.query()
        keyed.sessions.query()

        assertEquals(listOf("Bearer token-1", "Bearer token-1"), router.requests.map { it.header("Authorization") })
        assertEquals("key-1", router.requests.first().header("X-Api-Key"))
        assertEquals("jwt", router.requests.first().header("Stream-Auth-Type"))
        assertNull(router.requests.first().header("X-Customer-Id"))
        keyed.close()
    }

    @Test
    fun `an expired token is asked for again and the request retried once`() = runTest {
        val asked = AtomicInteger()
        val keyed = VisionAgents(url = router.url, apiKey = "key-1")
        keyed.setUser(User("jlahey")) { "token-${asked.incrementAndGet()}" }
        router.answer = { if (it.header("Authorization") == "Bearer token-1") Reply(401, """{"error":"expired"}""") else Reply(200, "[]") }

        keyed.sessions.query()

        assertEquals(listOf("Bearer token-1", "Bearer token-2"), router.requests.map { it.header("Authorization") })
        keyed.close()
    }

    @Test
    fun `an api key with nobody to act for is refused before any request`() = runTest {
        val keyed = VisionAgents(url = router.url, apiKey = "key-1")

        assertFailsWith<AgentsException.Configuration> { keyed.sessions.query() }
        assertTrue(router.requests.isEmpty())
        keyed.close()
    }

    @Test
    fun `a session request carries what was set and nothing the schema defaults`() = runTest {
        router.answer = { Reply(201, sessionJson()) }

        val session = agents.agent("docs").sessions.create(
            SessionOptions(
                title = "Billing",
                project = "Health",
                incognito = false,
                custom = buildJsonObject { put("tenant", "acme") },
                modelOverwrites = ModelOverwrites(thinking = ModelOverwrites.Thinking.High, temperature = 0.0),
            ),
        )

        val body = Json.parseToJsonElement(router.requests.single().body).jsonObject
        assertEquals(
            setOf("text", "agent", "title", "project", "incognito", "custom", "model_overwrites"),
            body.keys,
        )
        assertEquals(JsonPrimitive(true), body["text"])
        assertEquals(JsonPrimitive("docs"), body["agent"])
        assertEquals("""{"thinking":"high","temperature":0.0}""", body["model_overwrites"].toString())
        assertEquals("s1", session.id)
        assertEquals("Billing", session.title)
        assertEquals(Instant.parse("2026-09-24T04:03:19.979138Z"), session.createdAt)
    }

    @Test
    fun `a voice session names its call and is not a text one`() = runTest {
        router.answer = { Reply(201, sessionJson(text = false)) }

        agents.sessions.create(SessionOptions(configId = "cfg-1"), callId = "call-1")

        val body = Json.parseToJsonElement(router.requests.single().body).jsonObject
        assertEquals(setOf("call_id", "config_id"), body.keys)
    }

    @Test
    fun `listing an agent's sessions narrows to it and sends only the filters asked for`() = runTest {
        router.answer = { Reply(200, "[${sessionJson(state = "closed")}]") }

        val found = agents.agent("docs").sessions.query(
            SessionQuery(state = SessionQuery.State.Closed, createdAfter = Instant.parse("2026-09-01T00:00:00Z"), limit = 10),
        )

        val arrived = router.requests.single()
        assertEquals("/v1/agents/sessions", arrived.path)
        assertEquals(
            mapOf(
                "agent" to listOf("docs"),
                "state" to listOf("closed"),
                "created_after" to listOf("2026-09-01T00:00:00Z"),
                "limit" to listOf("10"),
            ),
            arrived.query,
        )
        assertEquals(Session.State.Ended, found.single().state)
    }

    @Test
    fun `searching sends the words, and an empty box falls through to the list`() = runTest {
        router.answer = { Reply(200, "[]") }

        agents.sessions.search("billing \"q3\"")
        agents.sessions.search("")

        assertEquals("/v1/agents/sessions/search", router.requests[0].path)
        assertEquals(listOf("billing \"q3\""), router.requests[0].query["q"])
        assertNull(router.requests[1].query["q"])
    }

    @Test
    fun `an id is a path segment, however it is spelled`() = runTest {
        router.answer = { Reply(200, sessionJson()) }

        agents.sessions.get("a/b c")

        assertEquals("/v1/agents/sessions/a%2Fb%20c", router.requests.single().path)
    }

    @Test
    fun `closing, rewinding and forking are the requests the router takes`() = runTest {
        router.answer = { arrived ->
            when {
                arrived.path.endsWith("/fork") -> Reply(201, sessionJson(id = "s2"))
                else -> Reply(204)
            }
        }

        agents.sessions.responses("s1").rewind("r1")
        val fork = agents.sessions.fork("s1", ForkOptions(responseId = "r1"))
        agents.sessions.close("s2")

        val (rewind, forked, closed) = router.requests
        assertEquals("POST /v1/agents/sessions/s1/rewind", "${rewind.method} ${rewind.path}")
        assertEquals("""{"response_id":"r1"}""", rewind.body)
        assertEquals("POST /v1/agents/sessions/s1/fork", "${forked.method} ${forked.path}")
        assertEquals("""{"response_id":"r1"}""", forked.body)
        assertEquals("DELETE /v1/agents/sessions/s2", "${closed.method} ${closed.path}")
        assertEquals("s2", fork.id)
    }

    @Test
    fun `a fork without history says so, and only then`() = runTest {
        router.answer = { Reply(201, sessionJson(id = "s2")) }

        agents.sessions.fork("s1", ForkOptions(withoutHistory = true, title = "again"))

        assertEquals("""{"title":"again","messages":false}""", router.requests.single().body)
    }

    @Test
    fun `a persistent conversation refused a rewind reports what the router said`() = runTest {
        router.answer = { Reply(400, """{"error":"a persistent conversation cannot be rewound; fork it at the response instead"}""") }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.responses("s1").rewind("r1") }

        assertEquals(400, refused.status)
        assertTrue(refused.reason.contains("fork it"))
    }

    @Test
    fun `a device that has used up its day is told when it can ask again`() = runTest {
        router.answer = { Reply(429, """{"error":"daily limit"}""", mapOf("Retry-After" to "3600")) }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.responses("s1").create("hi") }

        assertEquals(3600L, refused.retryAfterSeconds)
    }

    @Test
    fun `a server-side path refused is recognisable as one`() = runTest {
        router.answer = { Reply(403, """{"error":"server-side only"}""") }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.get("s1") }

        assertTrue(refused.isServerSideOnly)
    }

    @Test
    fun `an error body that is not json is still reported, cut short`() = runTest {
        router.answer = { Reply(500, "store: query sessions: " + "x".repeat(2000)) }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.query() }

        assertTrue(refused.reason.startsWith("store: query sessions"))
        assertEquals(512, refused.reason.length)
    }

    @Test
    fun `a router that is not there is a transport failure, not a status`() = runTest {
        val nowhere = VisionAgents(url = "http://127.0.0.1:1", customerId = "acme")

        assertFailsWith<AgentsException.Transport> { nowhere.sessions.query() }
        nowhere.close()
    }

    @Test
    fun `an answer that is not what the spec says is unreadable`() = runTest {
        router.answer = { Reply(200, """{"not":"a list"}""") }

        assertFailsWith<AgentsException.Unreadable> { agents.sessions.query() }
    }

    @Test
    fun `asking something returns a handle on the turn it starts`() = runTest {
        router.answer = { Reply(202, responseJson("r1", status = "running")) }

        val response = agents.sessions.responses("s1").create("What changed?", listOf(ImageSource("data:image/png;base64,AAAA")))

        assertEquals("""{"text":"What changed?","images":[{"url":"data:image/png;base64,AAAA"}]}""", router.requests.single().body)
        assertEquals(AgentResponse.Status.Running, response.status)
        assertEquals("r1", response.id)
    }

    @Test
    fun `a status or kind the SDK never heard of is unknown rather than a crash`() = runTest {
        router.answer = { arrived ->
            if (arrived.path.endsWith("/items")) Reply(200, "[${itemJson("r1", 0, kind = "pondered")}]")
            else Reply(200, "[${responseJson("r1", status = "paused")}]")
        }

        val responses = agents.sessions.responses("s1")

        assertEquals(AgentResponse.Status.Unknown, responses.list().single().status)
        assertEquals(ResponseItem.Kind.Unknown, responses.items().single().kind)
    }

    @Test
    fun `unwinding reads every page and stops at the short one`() = runTest {
        router.answer = { arrived ->
            val offset = arrived.query["offset"]?.single()?.toInt() ?: 0
            val count = if (offset == 0) 200 else 3
            Reply(200, (0 until count).joinToString(",", "[", "]") { itemJson("r1", offset + it) })
        }

        val items = agents.sessions.responses("s1").unwind(responseId = "r1").toList()

        assertEquals(203, items.size)
        assertEquals((0 until 203).toList(), items.map { it.ordinal })
        assertEquals(listOf(listOf("0"), listOf("200")), router.requests.map { it.query["offset"] })
        assertEquals(listOf("r1"), router.requests.first().query["response_id"])
    }

    @Test
    fun `search is one round trip under the config named`() = runTest {
        router.answer = {
            Reply(200, """{"provider":"exa","model":"exa-fast","answer":"Yes.","results":[{"url":"https://a","title":"A","score":0.5}]}""")
        }

        val found = agents.router(config = "healthcare").search("pricing", SearchOptions(depth = SearchOptions.Depth.Fast))

        assertEquals("""{"query":"pricing","config_id":"healthcare","options":{"depth":"fast"}}""", router.requests.single().body)
        assertEquals("Yes.", found.answer)
        assertEquals(0.5, found.results.single().score)
    }

    @Test
    fun `a remembered guest is the same guest on the next launch`() = runTest {
        val store = FileGuestStore(File(Files.createTempDirectory("guest").toFile(), "guest.json"))
        router.answer = { Reply(201, """{"id":"guest-1","token":"tok","name":"Guest","expires_at":"2999-01-01T00:00:00Z"}""") }

        val first = agents.guestUser(GuestOptions(name = "Guest"), store)
        val again = agents.guestUser(GuestOptions(name = "Guest"), store)

        assertEquals(first, again)
        assertEquals(1, router.requests.size)
        assertEquals("""{"name":"Guest"}""", router.requests.single().body)
    }

    @Test
    fun `a guest whose token expired comes back as the same guest with a fresh one`() = runTest {
        val store = FileGuestStore(File(Files.createTempDirectory("guest").toFile(), "guest.json"))
        store.write("""{"id":"guest-1","token":"old","expires_at":"2020-01-01T00:00:00Z"}""")
        router.answer = { Reply(201, """{"id":"guest-1","token":"new","expires_at":"2999-01-01T00:00:00Z"}""") }

        val guest = agents.guestUser(store = store)

        assertEquals("""{"id":"guest-1"}""", router.requests.single().body)
        assertEquals("new", guest.token)
    }

    @Test
    fun `not me mints somebody new, and a store holding rubbish is not fatal`() = runTest {
        val store = FileGuestStore(File(Files.createTempDirectory("guest").toFile(), "guest.json"))
        store.write("{truncated")
        router.answer = { Reply(201, """{"id":"guest-${router.requests.size}","token":"t"}""") }

        val first = agents.guestUser(store = store)
        val second = agents.guestUser(GuestOptions(fresh = true), store)

        assertEquals(2, router.requests.size)
        assertFalse(first.id == second.id)
        agents.forgetGuest(store)
        assertNull(store.read())
    }
}
