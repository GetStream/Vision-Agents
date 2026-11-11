package io.getstream.agents.android.core

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
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/** A session as the router answers one: every required field and a handful of the optional ones. */
internal fun sessionJson(id: String = "s1", state: String = "live", callId: String = "", modality: String = "text") =
    """{"id":"$id","call_id":"$callId","call_type":"agent","user_id":"u1","agent_id":"a1","state":"$state",""" +
        """"modality":"$modality","created_at":"2026-09-24T04:03:19.979138Z","text":${callId.isEmpty()},"agent":"docs",""" +
        """"title":"Billing","project_id":"Health","custom":{"tenant":"acme"},"closed_at":"2026-09-24T04:03:20Z",""" +
        """"some_field_added_later":1}"""

internal fun responseJson(id: String, status: String = "completed") =
    """{"id":"$id","session_id":"s1","said":"hi","status":"$status","created_at":"2026-09-24T04:03:19Z"}"""

private fun itemJson(response: String, ordinal: Int, kind: String = "answer") =
    """{"response_id":"$response","ordinal":$ordinal,"kind":"$kind","text":"t$ordinal","at":"2026-09-24T04:03:19.5Z"}"""

internal fun pageJson(items: List<String>, next: String? = null) =
    """{"items":${items.joinToString(",", "[", "]")},"has_more":${next != null}""" +
        (if (next != null) ""","next_cursor":"$next"}""" else "}")

private val EMPTY_PAGE = pageJson(emptyList())

class VisionAgentsTest {
    private val router = TestRouter()
    private val agents = StreamAgents(url = router.url, customerId = "acme")
    private val deviceZone = deviceZoneId

    init {
        // Whatever zone the machine running these is in, a request asserted whole sends none.
        deviceZoneId = { null }
    }

    @AfterTest
    fun stop() {
        deviceZoneId = deviceZone
        agents.close()
        router.close()
    }

    @Test
    fun `naming no router reaches the hosted one`() {
        StreamAgents(apiKey = "key").use { hosted ->
            assertEquals(Backend.DEFAULT_URL, hosted.backend.url)
            assertEquals("https://accelerate-us-east4.gcp.stream-io-api.com", hosted.backend.url)
        }
    }

    @Test
    fun `a region is the hosted router there`() {
        assertEquals("https://accelerate-us-east4.gcp.stream-io-api.com", Backend.regionUrl("us-east4"))
        assertEquals("https://accelerate-europe-west4.gcp.stream-io-api.com", Backend.regionUrl("europe-west4"))
    }

    @Test
    fun `anything but a region name is refused`() {
        for (name in listOf("", "US-EAST4", "us-east4.evil.com/x", "us-east4\n", "us_east4", "useast4", "us-east")) {
            val refused = assertFailsWith<IllegalArgumentException>(name) { Backend.regionUrl(name) }
            assertTrue(name in refused.message.orEmpty(), refused.message)
        }
    }

    @Test
    fun `every request says it comes from a device, even to a router with nothing in front of it`() = runTest {
        router.answer = { Reply(200, EMPTY_PAGE) }

        agents.sessions.query()

        val arrived = router.requests.single()
        assertEquals("jwt", arrived.header("Stream-Auth-Type"))
        assertEquals("acme", arrived.header("X-Customer-Id"))
        assertNull(arrived.header("Authorization"))
        assertNull(arrived.header("X-Stream-User-Id"))
    }

    @Test
    fun `a user set on a router reached by customer id is named rather than proven`() = runTest {
        router.answer = { Reply(200, EMPTY_PAGE) }
        agents.setUser(User("jlahey", "Jim Lahey"), "unused")

        agents.sessions.query()

        assertEquals("jlahey", router.requests.single().header("X-Stream-User-Id"))
    }

    @Test
    fun `a key beside a customer id is Stream's, and the router is still reached by customer id`() = runTest {
        val local = StreamAgents(url = router.url, customerId = "acme", apiKey = "stream-key")
        local.setUser(User("jlahey"), "token-for-jlahey")
        router.answer = { Reply(200, EMPTY_PAGE) }

        local.sessions.query()

        val arrived = router.requests.single()
        assertEquals("acme", arrived.header("X-Customer-Id"))
        assertEquals("jlahey", arrived.header("X-Stream-User-Id"))
        assertNull(arrived.query["api_key"])
        assertNull(arrived.header("Authorization"))
        assertEquals("stream-key", local.backend.streamCredentials().apiKey)
        local.close()
    }

    @Test
    fun `behind a verifying deployment the token is the credential, asked for once`() = runTest {
        val asked = AtomicInteger()
        val keyed = StreamAgents(url = router.url, apiKey = "key-1")
        keyed.setUser(User("jlahey")) { "token-${asked.incrementAndGet()}" }
        router.answer = { Reply(200, EMPTY_PAGE) }

        keyed.sessions.query()
        keyed.sessions.query()

        assertEquals(listOf("Bearer token-1", "Bearer token-1"), router.requests.map { it.header("Authorization") })
        assertEquals(listOf("key-1"), router.requests.first().query["api_key"])
        assertNull(router.requests.first().header("X-Api-Key"))
        assertEquals("jwt", router.requests.first().header("Stream-Auth-Type"))
        assertNull(router.requests.first().header("X-Customer-Id"))
        keyed.close()
    }

    @Test
    fun `an expired token is asked for again and the request retried once`() = runTest {
        val asked = AtomicInteger()
        val keyed = StreamAgents(url = router.url, apiKey = "key-1")
        keyed.setUser(User("jlahey")) { "token-${asked.incrementAndGet()}" }
        router.answer = {
            if (it.header("Authorization") == "Bearer token-1") Reply(401, envelope("authentication", "unauthenticated", "expired"))
            else Reply(200, EMPTY_PAGE)
        }

        keyed.sessions.query()

        assertEquals(listOf("Bearer token-1", "Bearer token-2"), router.requests.map { it.header("Authorization") })
        keyed.close()
    }

    @Test
    fun `an api key with nobody to act for is refused before any request`() = runTest {
        val keyed = StreamAgents(url = router.url, apiKey = "key-1")

        assertFailsWith<AgentsException.Configuration> { keyed.sessions.query() }
        assertTrue(router.requests.isEmpty())
        keyed.close()
    }

    @Test
    fun `a session request carries what was set and nothing the schema defaults`() = runTest {
        router.answer = { Reply(201, sessionJson()) }

        val session = agents.agent("docs").sessions.create(
            SessionOptions(
                id = "order-1042",
                title = "Billing",
                projectId = "Health",
                incognito = false,
                custom = buildJsonObject { put("tenant", "acme") },
                modelOverwrites = ModelOverwrites(thinking = ModelOverwrites.Thinking.High, temperature = 0.0),
                greeting = Greeting("Hello.", Greeting.Mode.Variation),
                returnUrl = "com.example.app:/connected",
            ),
        )

        val body = Json.parseToJsonElement(router.requests.single().body).jsonObject
        assertEquals(
            setOf(
                "id", "agent", "title", "project_id", "incognito", "custom", "model_overwrites", "greeting",
                "return_url",
            ),
            body.keys,
        )
        assertEquals(JsonPrimitive("com.example.app:/connected"), body["return_url"])
        assertEquals(JsonPrimitive("order-1042"), body["id"])
        assertEquals(JsonPrimitive("docs"), body["agent"])
        assertEquals("""{"temperature":0.0,"thinking":"high"}""", body["model_overwrites"].toString())
        assertEquals("""{"text":"Hello.","mode":"variation"}""", body["greeting"].toString())
        assertEquals("s1", session.id)
        assertEquals("Billing", session.title)
        assertEquals("Health", session.projectId)
        assertEquals(Session.Modality.Text, session.modality)
        assertEquals(Instant.parse("2026-09-24T04:03:19.979138Z"), session.createdAt)
    }

    @Test
    fun `a session is in the device's zone unless it names its own`() = runTest {
        router.answer = { Reply(201, sessionJson()) }
        deviceZoneId = { "Asia/Kolkata" }

        agents.sessions.create()
        agents.agent("docs").sessions.create(SessionOptions(title = "Billing"))
        agents.sessions.create(SessionOptions(timezone = "Europe/Skopje"))

        val (device, agent, named) = router.requests.map { Json.parseToJsonElement(it.body).jsonObject }
        assertEquals("""{"timezone":"Asia/Kolkata"}""", device.toString())
        assertEquals(JsonPrimitive("Asia/Kolkata"), agent["timezone"])
        assertEquals(JsonPrimitive("Europe/Skopje"), named["timezone"])
    }

    @Test
    fun `a session that asks for no zone sends none, whatever the device is in`() = runTest {
        router.answer = { Reply(201, sessionJson()) }
        deviceZoneId = { "Asia/Kolkata" }

        agents.sessions.create(SessionOptions(timezone = null))
        agents.sessions.create(SessionOptions(timezone = ""))

        assertEquals(listOf("{}", "{}"), router.requests.map { it.body })
    }

    @Test
    fun `a device set to a bare offset sends no zone, but one the app names is sent as it is`() = runTest {
        router.answer = { Reply(201, sessionJson()) }
        deviceZoneId = { "GMT+05:30" }

        agents.sessions.create()
        // The router is the one that knows which zones it knows, so what the app names is its to refuse.
        agents.sessions.create(SessionOptions(timezone = "GMT+05:30"))

        val (device, named) = router.requests.map { it.body }
        assertEquals("{}", device)
        assertEquals("""{"timezone":"GMT+05:30"}""", named)
    }

    @Test
    fun `only a device zone named the IANA way is taken`() {
        val taken = listOf<String?>("Europe/Skopje", "America/Argentina/Buenos_Aires", "Etc/GMT-5", "UTC", "GMT")
        val left = listOf<String?>("GMT+05:30", "GMT-08:00", "EST", "Local", "", null)

        for (id in taken + left) {
            deviceZoneId = { id }
            assertEquals(id.takeIf { it in taken }, SessionOptions().timezone, "the device in $id")
        }
    }

    @Test
    fun `memory is the signed-in user's, narrowed by the labels asked for`() = runTest {
        router.answer = { Reply(201, sessionJson()) }
        agents.setUser(User("jlahey"), "unused")

        agents.sessions.create(
            SessionOptions(memory = SessionOptions.Memory(filter = mapOf("topic" to "billing"), appId = "web")),
        )
        agents.sessions.create(SessionOptions(memory = SessionOptions.Memory(), incognito = true))

        val (narrowed, incognito) = router.requests.map { Json.parseToJsonElement(it.body).jsonObject }
        assertEquals("""{"app_id":"web","filter":{"topic":"billing"},"user_id":"jlahey"}""", narrowed["memory"].toString())
        // The router keeps nothing of an incognito session, so the two are sent as asked and it decides.
        assertEquals(setOf("incognito", "memory"), incognito.keys)
        assertEquals("""{"user_id":"jlahey"}""", incognito["memory"].toString())
    }

    @Test
    fun `a session without memory sends none, whoever is signed in`() = runTest {
        router.answer = { Reply(201, sessionJson()) }
        agents.setUser(User("jlahey"), "unused")

        agents.sessions.create()

        assertEquals("{}", router.requests.single().body)
    }

    @Test
    fun `a modality the SDK never heard of is unknown`() = runTest {
        router.answer = { Reply(200, sessionJson(modality = "telepathy")) }

        assertEquals(Session.Modality.Unknown, agents.sessions.get("s1").modality)
    }

    @Test
    fun `a voice session asks to start voice and is on its own call`() = runTest {
        router.answer = { Reply(201, sessionJson(callId = "s1")) }

        val session = agents.sessions.create(SessionOptions(configId = "cfg-1", startVoice = true))

        val body = Json.parseToJsonElement(router.requests.single().body).jsonObject
        assertEquals(setOf("config_id", "start_voice"), body.keys)
        assertEquals(JsonPrimitive(true), body["start_voice"])
        assertEquals("s1", session.callId)
    }

    @Test
    fun `starting and stopping voice moves a session onto its call and off it`() = runTest {
        router.answer = { Reply(200, sessionJson(callId = if (it.method == "POST") "s1" else "")) }

        val started = agents.sessions.startVoice("s1")
        val stopped = agents.sessions.stopVoice("s1")

        assertEquals(
            listOf("POST /v1/agents/sessions/s1/voice", "DELETE /v1/agents/sessions/s1/voice"),
            router.requests.map { "${it.method} ${it.path}" },
        )
        assertEquals("s1", started.callId)
        assertEquals("", stopped.callId)
    }

    @Test
    fun `listing an agent's sessions with no filter sends just the agent`() = runTest {
        router.answer = { Reply(200, pageJson(listOf(sessionJson(state = "ended", modality = "voice")), next = "c2")) }

        val found = agents.agent("docs").sessions.query()
        agents.sessions.query()

        val arrived = router.requests.first()
        assertEquals("POST /v1/agents/sessions/query", "${arrived.method} ${arrived.path}")
        assertEquals("""{"filter":{"agent":{"${'$'}eq":"docs"}}}""", arrived.body)
        assertEquals("""{"filter":{}}""", router.requests.last().body)
        assertEquals(Session.State.Ended, found.items.single().state)
        assertEquals(Session.Modality.Voice, found.items.single().modality)
        assertTrue(found.hasMore)
        assertEquals("c2", found.nextCursor)
    }

    @Test
    fun `a project filter is sent beside the agent, as one project or as one of several`() = runTest {
        router.answer = { Reply(200, EMPTY_PAGE) }
        val sessions = agents.agent("docs").sessions

        sessions.query(filter = SessionFilter.ProjectId(eq = "a"))
        sessions.query(filter = SessionFilter.ProjectIdIn(listOf("a", "b")))

        val (one, several) = router.requests.map { it.body }
        assertEquals("""{"filter":{"agent":{"${'$'}eq":"docs"},"project_id":{"${'$'}eq":"a"}}}""", one)
        assertEquals("""{"filter":{"agent":{"${'$'}eq":"docs"},"project_id":{"${'$'}in":["a","b"]}}}""", several)
    }

    @Test
    fun `an or of two projects, and a nested and, are sent as they are with the agent beside them`() = runTest {
        router.answer = { Reply(200, EMPTY_PAGE) }
        val sessions = agents.agent("docs").sessions
        val either = SessionFilter.Or(SessionFilter.ProjectId("a"), SessionFilter.ProjectId("b"))

        sessions.query(filter = either)
        sessions.query(filter = SessionFilter.And(either, SessionFilter.ProjectIdIn(listOf("a", "c"))))

        val (or, nested) = router.requests.map { it.body }
        assertEquals(
            """{"filter":{"${'$'}or":[{"project_id":{"${'$'}eq":"a"}},{"project_id":{"${'$'}eq":"b"}}],"agent":{"${'$'}eq":"docs"}}}""",
            or,
        )
        assertEquals(
            """{"filter":{"${'$'}and":[{"${'$'}or":[{"project_id":{"${'$'}eq":"a"}},{"project_id":{"${'$'}eq":"b"}}]},""" +
                """{"project_id":{"${'$'}in":["a","c"]}}],"agent":{"${'$'}eq":"docs"}}}""",
            nested,
        )
    }

    @Test
    fun `an agent's sessions are that agent's whatever the filter names, and any agent's from the top`() = runTest {
        router.answer = { Reply(200, EMPTY_PAGE) }

        agents.agent("docs").sessions.query(filter = SessionFilter.Agent("billing"))
        agents.sessions.query(filter = SessionFilter.Agent("billing"))

        val (scoped, open) = router.requests.map { it.body }
        assertEquals("""{"filter":{"agent":{"${'$'}eq":"docs"}}}""", scoped)
        assertEquals("""{"filter":{"agent":{"${'$'}eq":"billing"}}}""", open)
    }

    @Test
    fun `a filter serialises to the wire's own keys`() {
        val filter = SessionFilter.And(
            SessionFilter.Or(SessionFilter.ProjectId("a"), SessionFilter.ProjectIdIn(listOf("b", "c"))),
            SessionFilter.Text("billing"),
        )

        assertEquals(
            """{"${'$'}and":[{"${'$'}or":[{"project_id":{"${'$'}eq":"a"}},{"project_id":{"${'$'}in":["b","c"]}}]},{"text":{"${'$'}q":"billing"}}]}""",
            filter.toJson().toString(),
        )
    }

    @Test
    fun `each sort sends its field and direction, and no sort sends none`() = runTest {
        router.answer = { Reply(200, EMPTY_PAGE) }
        val sessions = agents.sessions

        sessions.query(sort = listOf(SessionSort(SessionSortField.LastMessageAt)))
        sessions.query(sort = listOf(SessionSort(SessionSortField.LastMessageAt, direction = 1)))
        sessions.query(sort = listOf(SessionSort(SessionSortField.Pinned)))
        sessions.query(sort = listOf(SessionSort(SessionSortField.Pinned), SessionSort(SessionSortField.LastMessageAt, 1)))
        sessions.query(filter = SessionFilter.Text("pricing"), sort = listOf(SessionSort(SessionSortField.Relevance)))

        assertEquals(
            listOf(
                """{"filter":{},"sort":[{"field":"last_message_at","direction":-1}]}""",
                """{"filter":{},"sort":[{"field":"last_message_at","direction":1}]}""",
                """{"filter":{},"sort":[{"field":"pinned","direction":-1}]}""",
                """{"filter":{},"sort":[{"field":"pinned","direction":-1},{"field":"last_message_at","direction":1}]}""",
                """{"filter":{"text":{"${'$'}q":"pricing"}},"sort":[{"field":"relevance","direction":-1}]}""",
            ),
            router.requests.map { it.body },
        )
        assertFailsWith<AgentsException.Configuration> {
            sessions.query(sort = listOf(SessionSort(SessionSortField.LastMessageAt, direction = 0)))
        }
    }

    @Test
    fun `the next page is asked for with the cursor of the one before`() = runTest {
        router.answer = { Reply(200, pageJson(listOf(sessionJson()), next = "c2")) }
        val sessions = agents.agent("docs").sessions
        val filter = SessionFilter.ProjectId("a")

        val first = sessions.query(filter = filter, limit = 20)
        sessions.query(filter = filter, limit = 20, cursor = first.nextCursor)

        val (opening, following) = router.requests.map { it.body }
        assertEquals("""{"filter":{"agent":{"${'$'}eq":"docs"},"project_id":{"${'$'}eq":"a"}},"limit":20}""", opening)
        assertEquals(
            """{"cursor":"c2","filter":{"agent":{"${'$'}eq":"docs"},"project_id":{"${'$'}eq":"a"}},"limit":20}""",
            following,
        )
    }

    @Test
    fun `searching sends the words, and an empty box falls through to the list`() = runTest {
        router.answer = { Reply(200, EMPTY_PAGE) }

        val found = agents.sessions.search("billing \"q3\"")
        agents.sessions.search("")

        assertEquals("/v1/agents/sessions/query", router.requests[0].path)
        assertEquals("""{"filter":{"text":{"${'$'}q":"billing \"q3\""}}}""", router.requests[0].body)
        assertEquals("""{"filter":{}}""", router.requests[1].body)
        assertFalse(found.hasMore)
        assertNull(found.nextCursor)
    }

    @Test
    fun `a search within a project sets the words beside the filter, and pages`() = runTest {
        router.answer = { Reply(200, EMPTY_PAGE) }
        val sessions = agents.agent("docs").sessions

        sessions.search("billing", filter = SessionFilter.ProjectId("a"))
        sessions.search("billing", filter = SessionFilter.Or(SessionFilter.ProjectId("a"), SessionFilter.ProjectId("b")), limit = 10, cursor = "c1")

        val (within, paged) = router.requests.map { it.body }
        assertEquals(
            """{"filter":{"agent":{"${'$'}eq":"docs"},"project_id":{"${'$'}eq":"a"},"text":{"${'$'}q":"billing"}}}""",
            within,
        )
        assertEquals(
            """{"cursor":"c1","filter":{"${'$'}or":[{"project_id":{"${'$'}eq":"a"}},{"project_id":{"${'$'}eq":"b"}}],""" +
                """"agent":{"${'$'}eq":"docs"},"text":{"${'$'}q":"billing"}},"limit":10}""",
            paged,
        )
    }

    @Test
    fun `a search refuses a filter that already holds words`() = runTest {
        assertFailsWith<AgentsException.Configuration> {
            agents.sessions.search("billing", filter = SessionFilter.Text("pricing"))
        }
        assertFailsWith<AgentsException.Configuration> {
            agents.sessions.search("billing", filter = SessionFilter.And(SessionFilter.ProjectId("a"), SessionFilter.Text("pricing")))
        }
        assertTrue(router.requests.isEmpty())
    }

    @Test
    fun `pinning and unpinning send only the pin, and a pinned session says when`() = runTest {
        val pinned = sessionJson().dropLast(1) + ""","pinned_at":"2026-11-10T09:00:00Z"}"""
        router.answer = { if (it.body.contains("true")) Reply(200, pinned) else Reply(200, sessionJson()) }

        val held = agents.sessions.pin("s1")
        val released = agents.sessions.unpin("s1")
        agents.sessions.update("s1", title = "Billing", pinned = true)

        val (pin, unpin, both) = router.requests
        assertEquals("PATCH /v1/agents/sessions/s1", "${pin.method} ${pin.path}")
        assertEquals("""{"pinned":true}""", pin.body)
        assertEquals("PATCH /v1/agents/sessions/s1", "${unpin.method} ${unpin.path}")
        assertEquals("""{"pinned":false}""", unpin.body)
        assertEquals("""{"pinned":true,"title":"Billing"}""", both.body)
        assertEquals(Instant.parse("2026-11-10T09:00:00Z"), held.pinnedAt)
        assertNull(released.pinnedAt)
    }

    @Test
    fun `a page of sessions carries which are pinned`() = runTest {
        val pinned = sessionJson(id = "s2").dropLast(1) + ""","pinned_at":"2026-11-10T09:00:00Z"}"""
        router.answer = { Reply(200, pageJson(listOf(pinned, sessionJson()))) }

        val found = agents.sessions.query(sort = listOf(SessionSort(SessionSortField.Pinned)))

        assertEquals(listOf(Instant.parse("2026-11-10T09:00:00Z"), null), found.items.map { it.pinnedAt })
    }

    @Test
    fun `an id is a path segment, however it is spelled`() = runTest {
        router.answer = { Reply(200, sessionJson()) }

        agents.sessions.get("a/b c")

        assertEquals("/v1/agents/sessions/a%2Fb%20c", router.requests.single().path)
    }

    @Test
    fun `stopping, deleting, rewinding and forking are the requests the router takes`() = runTest {
        router.answer = { arrived ->
            when {
                arrived.path.endsWith("/fork") -> Reply(201, sessionJson(id = "s2"))
                else -> Reply(204)
            }
        }

        agents.sessions.responses("s1").rewind("r1")
        val fork = agents.sessions.fork("s1", ForkOptions(responseId = "r1", projectId = "Health"))
        agents.sessions.close("s2")
        agents.sessions.delete("s2")

        val (rewind, forked, stopped, deleted) = router.requests
        assertEquals("POST /v1/agents/sessions/s1/rewind", "${rewind.method} ${rewind.path}")
        assertEquals("""{"response_id":"r1"}""", rewind.body)
        assertEquals("POST /v1/agents/sessions/s1/fork", "${forked.method} ${forked.path}")
        assertEquals("""{"project_id":"Health","response_id":"r1"}""", forked.body)
        assertEquals("POST /v1/agents/sessions/s2/stop", "${stopped.method} ${stopped.path}")
        assertEquals("DELETE /v1/agents/sessions/s2", "${deleted.method} ${deleted.path}")
        assertEquals("s2", fork.id)
    }

    @Test
    fun `an update sends only what changes and returns the session as it now is`() = runTest {
        router.answer = { Reply(200, sessionJson(state = "ended")) }

        val updated = agents.sessions.update("s1", title = "Billing", custom = buildJsonObject {})

        val arrived = router.requests.single()
        assertEquals("PATCH /v1/agents/sessions/s1", "${arrived.method} ${arrived.path}")
        assertEquals("""{"custom":{},"title":"Billing"}""", arrived.body)
        assertEquals("Billing", updated.title)
        assertEquals(Session.State.Ended, updated.state)
    }

    @Test
    fun `a fork without history says so, and only then`() = runTest {
        router.answer = { Reply(201, sessionJson(id = "s2")) }

        agents.sessions.fork("s1", ForkOptions(withoutHistory = true, title = "again"))

        assertEquals("""{"messages":false,"title":"again"}""", router.requests.single().body)
    }

    @Test
    fun `a persistent conversation refused a rewind reports what the router said`() = runTest {
        router.answer = {
            Reply(400, envelope("invalid_request", "invalid_request", "a persistent conversation cannot be rewound; fork it at the response instead"))
        }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.responses("s1").rewind("r1") }

        assertEquals(400, refused.status)
        assertTrue(refused.reason.contains("fork it"))
    }

    @Test
    fun `a device that has used up its day is told when it can ask again`() = runTest {
        router.answer = { Reply(429, envelope("rate_limited", "rate_limited", "daily limit"), mapOf("Retry-After" to "3600")) }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.responses("s1").create("hi") }

        assertEquals(3600L, refused.retryAfterSeconds)
    }

    @Test
    fun `a server-side path refused is recognisable as one`() = runTest {
        router.answer = { Reply(403, envelope("permission", "server_side_only", "server-side only")) }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.get("s1") }

        assertTrue(refused.isServerSideOnly)
        assertEquals("server_side_only", refused.code)
    }

    @Test
    fun `a refusal carries what the router said and the request id to quote`() = runTest {
        router.answer = {
            Reply(404, envelope("not_found", "session_not_found", "no session s1"), mapOf("X-Request-Id" to "req-1"))
        }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.get("s1") }

        assertEquals(404, refused.status)
        assertEquals("not_found", refused.type)
        assertEquals("session_not_found", refused.code)
        assertEquals("no session s1", refused.reason)
        assertEquals("https://getstream.io/agents/docs/api/errors/#session_not_found", refused.docUrl)
        assertEquals("req-1", refused.requestId)
        assertTrue(refused.message.orEmpty().contains("req-1"))
    }

    @Test
    fun `a type or code the SDK never heard of is kept as it came`() = runTest {
        router.answer = { Reply(402, envelope("payment_required", "credits_exhausted", "top up")) }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.query() }

        assertEquals("payment_required", refused.type)
        assertEquals("credits_exhausted", refused.code)
        assertEquals("top up", refused.reason)
    }

    @Test
    fun `a proxy's page in place of the router's error is the reason, with no type or code`() = runTest {
        router.answer = { Reply(502, "<html>bad gateway</html>\n", mapOf("X-Request-Id" to "req-2")) }

        val refused = assertFailsWith<AgentsException.Http> { agents.sessions.query() }

        assertEquals(502, refused.status)
        assertEquals("<html>bad gateway</html>", refused.reason)
        assertNull(refused.type)
        assertNull(refused.code)
        assertNull(refused.docUrl)
        assertEquals("req-2", refused.requestId)
    }

    @Test
    fun `an older router's error string and an empty body are reported as they came`() = runTest {
        router.answer = { if (it.path.endsWith("/s1")) Reply(400, """{"error":"bad id"}""") else Reply(503) }

        val old = assertFailsWith<AgentsException.Http> { agents.sessions.get("s1") }
        val empty = assertFailsWith<AgentsException.Http> { agents.sessions.query() }

        assertEquals("""{"error":"bad id"}""", old.reason)
        assertNull(old.type)
        assertNull(old.code)
        assertEquals("Service Unavailable", empty.reason)
        assertNull(empty.type)
        assertNull(empty.requestId)
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
        val nowhere = StreamAgents(url = "http://127.0.0.1:1", customerId = "acme")

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

        val body = Json.parseToJsonElement(router.requests.single().body).jsonObject
        assertEquals(setOf("text", "images", "request_id"), body.keys)
        assertEquals("""[{"url":"data:image/png;base64,AAAA"}]""", body["images"].toString())
        assertEquals(AgentResponse.Status.Running, response.status)
        assertEquals("r1", response.id)
    }

    @Test
    fun `a status or kind the SDK never heard of is unknown rather than a crash`() = runTest {
        router.answer = { arrived ->
            if (arrived.path.endsWith("/items")) Reply(200, pageJson(listOf(itemJson("r1", 0, kind = "pondered"))))
            else Reply(200, pageJson(listOf(responseJson("r1", status = "paused"))))
        }

        val responses = agents.sessions.responses("s1")

        assertEquals(AgentResponse.Status.Unknown, responses.list().items.single().status)
        assertEquals(ResponseItem.Kind.Unknown, responses.items().items.single().kind)
    }

    @Test
    fun `listing turns pages by cursor`() = runTest {
        router.answer = { Reply(200, pageJson(listOf(responseJson("r2")), next = "c3")) }

        val page = agents.sessions.responses("s1").list(limit = 1, cursor = "c2")

        assertEquals(mapOf("limit" to listOf("1"), "cursor" to listOf("c2")), router.requests.single().query)
        assertEquals("r2", page.items.single().id)
        assertEquals("c3", page.nextCursor)
    }

    @Test
    fun `unwinding follows the cursor until there is no more`() = runTest {
        router.answer = { arrived ->
            if (arrived.query["cursor"] == null) Reply(200, pageJson((0 until 200).map { itemJson("r1", it) }, next = "c1"))
            else Reply(200, pageJson((200 until 203).map { itemJson("r1", it) }))
        }

        val items = agents.sessions.responses("s1").unwind(responseId = "r1").toList()

        assertEquals(203, items.size)
        assertEquals((0 until 203).toList(), items.map { it.ordinal })
        assertEquals(listOf(null, listOf("c1")), router.requests.map { it.query["cursor"] })
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
    fun `a minted guest has its refresh secret remembered`() = runTest {
        val store = FileGuestStore(File(Files.createTempDirectory("guest").toFile(), "guest.json"))
        router.answer = { Reply(201, """{"id":"guest-1","token":"tok","refresh_secret":"s1"}""") }

        agents.guestUser(store = store)

        assertEquals("s1", Json.parseToJsonElement(store.read()!!).jsonObject["refresh_secret"]?.jsonPrimitive?.content)
    }

    @Test
    fun `a guest whose token expired comes back with their refresh secret`() = runTest {
        val store = FileGuestStore(File(Files.createTempDirectory("guest").toFile(), "guest.json"))
        store.write("""{"id":"guest-1","token":"old","expires_at":"2020-01-01T00:00:00Z","refresh_secret":"s1"}""")
        router.answer = { Reply(201, """{"id":"guest-1","token":"new","expires_at":"2999-01-01T00:00:00Z","refresh_secret":"s2"}""") }

        val guest = agents.guestUser(store = store)

        assertEquals("""{"id":"guest-1","refresh_secret":"s1"}""", router.requests.single().body)
        assertEquals("new", guest.token)
        assertEquals("s2", Json.parseToJsonElement(store.read()!!).jsonObject["refresh_secret"]?.jsonPrimitive?.content)
    }

    @Test
    fun `a spent secret is answered with somebody new, who is remembered instead`() = runTest {
        val store = FileGuestStore(File(Files.createTempDirectory("guest").toFile(), "guest.json"))
        store.write("""{"id":"guest-1","token":"old","expires_at":"2020-01-01T00:00:00Z","refresh_secret":"spent"}""")
        router.answer = { Reply(201, """{"id":"guest-2","token":"new","expires_at":"2999-01-01T00:00:00Z","refresh_secret":"s9"}""") }

        val guest = agents.guestUser(store = store)
        val again = agents.guestUser(store = store)

        assertEquals("guest-2", guest.id)
        assertEquals(guest, again)
        assertEquals(1, router.requests.size)
        val stored = Json.parseToJsonElement(store.read()!!).jsonObject
        assertEquals("guest-2", stored["id"]?.jsonPrimitive?.content)
        assertEquals("s9", stored["refresh_secret"]?.jsonPrimitive?.content)
    }

    @Test
    fun `an expired guest with no secret does not name an id it cannot prove`() = runTest {
        val store = FileGuestStore(File(Files.createTempDirectory("guest").toFile(), "guest.json"))
        store.write("""{"id":"guest-1","token":"old","expires_at":"2020-01-01T00:00:00Z"}""")
        router.answer = { Reply(201, """{"id":"guest-2","token":"new"}""") }

        val guest = agents.guestUser(GuestOptions(name = "Guest"), store)

        assertEquals("""{"name":"Guest"}""", router.requests.single().body)
        assertEquals("guest-2", guest.id)
    }

    @Test
    fun `a device with only the app key asks for a guest anonymously`() = runTest {
        router.answer = { Reply(201, """{"id":"guest-1","token":"t"}""") }

        val guest = StreamAgents(url = router.url, apiKey = "key").use { it.guestUser() }

        assertEquals("guest-1", guest.id)
        val arrived = router.requests.single()
        assertEquals("anonymous", arrived.header("Stream-Auth-Type"))
        assertNull(arrived.header("Authorization"))
        assertEquals(listOf("key"), arrived.query["api_key"])
    }

    @Test
    fun `a router admitting no anonymous callers says to mint the guest on the backend`() = runTest {
        router.answer = {
            Reply(401, """{"code":2,"message":"anonymous access is not available on the acceleration api","StatusCode":401}""")
        }

        val refused = StreamAgents(url = router.url, apiKey = "key").use {
            assertFailsWith<AgentsException.Configuration> { it.guestUser() }
        }

        assertTrue(refused.message.orEmpty().contains("setUser"))
        assertEquals(1, router.requests.size)
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
