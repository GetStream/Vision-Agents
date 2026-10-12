package io.getstream.visionagents.core

import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.util.Base64
import java.util.UUID
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable

/** The spec the router is built from, beside this SDK in the repo. */
private val spec = File("../../../acceleration/api/openapi.yaml")

/**
 * Every operation the spec marks `x-client-accessible: true`, as a method and a path pattern.
 *
 * Read a method block at a time, so a mark on one operation is never credited to its neighbour.
 */
private fun clientAccessible(): List<Pair<String, Regex>> {
    val methods = setOf("get", "put", "post", "patch", "delete")
    val accessible = mutableListOf<Pair<String, Regex>>()
    var path = ""
    var method = ""
    var marked = false
    fun close() {
        if (marked && method.isNotEmpty()) {
            accessible += method.uppercase() to Regex(path.replace(Regex("\\{[^}]+}"), "[^/]+"))
        }
        method = ""
        marked = false
    }
    for (line in spec.readLines()) {
        val indent = line.takeWhile { it == ' ' }.length
        when {
            line.isNotEmpty() && indent <= 2 -> {
                close()
                if (indent == 2 && line.trimStart().startsWith("/")) path = line.trim().removeSuffix(":")
            }
            indent == 4 && line.trim().removeSuffix(":") in methods -> {
                close()
                method = line.trim().removeSuffix(":")
            }
            line == "      x-client-accessible: true" -> marked = true
        }
    }
    close()
    return accessible
}

/**
 * What a device holding this SDK can reach: a conversation, and nothing that is the app's
 * backend's to do.
 */
class PostureTest {
    private val router = TestRouter()
    private val agents = VisionAgents(url = router.url, customerId = "acme")

    @AfterTest
    fun stop() {
        agents.close()
        router.close()
    }

    @Test
    fun `everything the SDK can ask for is something the spec lets a device ask for`() = runTest {
        router.answer = { arrived ->
            when {
                arrived.path == "/v1/search" ->
                    Reply(200, """{"provider":"exa","model":"exa-fast","answer":"Yes.","results":[]}""")
                arrived.path == "/v1/agents/guests" -> Reply(201, """{"id":"guest-1","token":"t"}""")
                arrived.path.endsWith("/query") -> Reply(200, pageJson(emptyList()))
                arrived.path.endsWith("/responses") && arrived.method == "POST" -> Reply(202, responseJson("r1"))
                arrived.path.endsWith("/responses") || arrived.path.endsWith("/items") -> Reply(200, pageJson(emptyList()))
                arrived.path.endsWith("/voice") -> Reply(200, sessionJson())
                arrived.path.endsWith("/rewind") || arrived.path.endsWith("/stop") || arrived.method == "DELETE" -> Reply(204)
                else -> Reply(200, sessionJson())
            }
        }
        val sessions = agents.sessions

        sessions.create()
        sessions.query()
        sessions.search("billing")
        sessions.get("s1")
        sessions.update("s1", title = "Billing")
        sessions.startVoice("s1")
        sessions.stopVoice("s1")
        sessions.fork("s1")
        sessions.responses("s1").create("hi")
        sessions.responses("s1").list()
        sessions.responses("s1").items()
        sessions.responses("s1").rewind("r1")
        sessions.close("s1")
        sessions.delete("s1")
        agents.router().search("pricing")
        agents.guestUser()

        val accessible = clientAccessible()
        assertTrue(accessible.any { (method, path) -> method == "POST" && path.matches("/v1/agents/sessions") })
        assertTrue(accessible.none { (_, path) -> path.matches("/v1/agents/calls/x/token") })
        val refused = router.requests.filter { arrived ->
            accessible.none { (method, path) -> method == arrived.method && path.matches(arrived.path) }
        }
        assertEquals(emptyList(), refused.map { "${it.method} ${it.path}" }, "server-side only")
        assertTrue(router.requests.all { it.header("Stream-Auth-Type") == "jwt" })
    }
}

/**
 * The same promise against a router that verifies credentials, which needs one running in
 * `api_key` mode and a key minted for it with `router keys create`; the JavaScript README has
 * the commands.
 */
@EnabledIfEnvironmentVariable(named = "INTEGRATION_ROUTER_KEY", matches = ".+")
class KeyedRouterPostureTest {
    private val url = System.getenv("INTEGRATION_ROUTER_URL") ?: "http://localhost:8099"
    private val key = System.getenv("INTEGRATION_ROUTER_KEY").orEmpty()
    private val secret = System.getenv("INTEGRATION_ROUTER_SECRET").orEmpty()
    private val agents = VisionAgents(url = url, apiKey = key).apply {
        val id = "kotlin-device-${UUID.randomUUID()}"
        setUser(User(id), sign("""{"user_id":"$id","exp":${System.currentTimeMillis() / 1000 + 3600}}"""))
    }

    @AfterTest
    fun stop() {
        agents.close()
    }

    @Test
    fun `a device may not mint tokens or configure agents`() = runTest {
        val serverOnly = listOf(
            "POST" to "/v1/agents/calls/default:anything/token",
            "POST" to "/v1/agents/chat-token",
            "POST" to "/v1/agents/guests/claim",
            "GET" to "/v1/agents/configs",
        )
        for ((method, path) in serverOnly) {
            val (status, code) = send(method, path)

            assertEquals(403, status, "$method $path answered $status")
            assertEquals("server_side_only", code)
        }
    }

    @Test
    fun `a device may still list its own conversations`() = runTest {
        agents.sessions.query()
    }

    /** Sends what the SDK sends for this device, to a path the SDK has no method for. */
    private suspend fun send(method: String, path: String): Pair<Int, String?> {
        val request = HttpRequest.newBuilder(URI.create(agents.backend.url(*path.trim('/').split('/').toTypedArray())))
            .method(method, if (method == "POST") HttpRequest.BodyPublishers.ofString("{}") else HttpRequest.BodyPublishers.noBody())
            .header("Content-Type", "application/json")
        agents.backend.headers().forEach { (name, value) -> request.header(name, value) }
        val response = HttpClient.newHttpClient().send(request.build(), HttpResponse.BodyHandlers.ofString())
        val code = Json.parseToJsonElement(response.body()).jsonObject["error"]?.jsonObject?.get("code")?.jsonPrimitive?.content
        return response.statusCode() to code
    }

    /** An HS256 token, which is what the app's own backend would have minted. */
    private fun sign(claims: String): String {
        val encode = Base64.getUrlEncoder().withoutPadding()
        val unsigned = listOf("""{"alg":"HS256","typ":"JWT"}""", claims)
            .joinToString(".") { encode.encodeToString(it.toByteArray()) }
        val mac = Mac.getInstance("HmacSHA256").apply { init(SecretKeySpec(secret.toByteArray(), "HmacSHA256")) }
        return unsigned + "." + encode.encodeToString(mac.doFinal(unsigned.toByteArray()))
    }
}
