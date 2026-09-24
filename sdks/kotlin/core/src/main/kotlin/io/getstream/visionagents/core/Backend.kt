package io.getstream.visionagents.core

import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.websocket.WebSockets
import io.ktor.http.URLBuilder
import io.ktor.http.URLProtocol
import io.ktor.http.appendPathSegments
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import okhttp3.OkHttpClient

/**
 * Somebody a device is acting for, as Stream knows them.
 *
 * Only the id reaches the router; the token is what proves it. The name and image are carried
 * so chat and video have something to show without a second lookup.
 */
public data class User(
    val id: String,
    val name: String = "",
    val image: String = "",
)

/**
 * Hands over a token for the user, and a fresh one when asked again.
 *
 * A provider rather than a string, because a token expires and an hour-long conversation
 * should not end when it does. It is asked once, then again after a 401.
 */
public fun interface TokenProvider {
    public suspend fun token(): String
}

/**
 * Where the router is and who is asking.
 *
 * A phone holds no secret worth having, so there is no API secret here. A router with nothing
 * in front of it is reached by [customerId]; a deployment verifying tokens is reached by
 * [apiKey] and the token [setUser] is given, which the app's own backend minted.
 *
 * Constructing one does no I/O. Pass an [OkHttpClient] to share the app's connection pool,
 * proxy and certificate pinning; [close] releases the HTTP client built around it.
 */
public class Backend(
    url: String,
    public val customerId: String = "",
    public val apiKey: String = "",
    okHttpClient: OkHttpClient? = null,
) : AutoCloseable {
    /** The router's base URL, with no trailing slash. */
    public val url: String = url.trimEnd('/')

    internal val http: HttpClient = HttpClient(OkHttp) {
        expectSuccess = false
        install(WebSockets)
        engine {
            if (okHttpClient != null) preconfigured = okHttpClient
        }
    }

    private val lock = Mutex()
    @Volatile private var identity: Identity? = null
    @Volatile private var cached: String? = null

    /** Who this is acting for, or null until [setUser]. */
    public val user: User? get() = identity?.user

    /**
     * Says who this device is acting for, and how to prove it.
     *
     * Against a router reached by customer id the token is not read, and the user id is what
     * names the end user; behind a verifying deployment the token is the whole of it.
     */
    public fun setUser(user: User, token: TokenProvider) {
        if (user.id.isEmpty()) throw AgentsException.Configuration("a user needs an id")
        identity = Identity(user, token)
        cached = null
    }

    /** Says who this device is acting for, with a token that will not be refreshed. */
    public fun setUser(user: User, token: String) {
        setUser(user, TokenProvider { token })
    }

    /** Acts as a guest from here on. */
    public fun setUser(guest: Guest) {
        setUser(User(guest.id, guest.name), guest.token)
    }

    /** Forgets the user, which is what signing out is. */
    public fun clearUser() {
        identity = null
        cached = null
    }

    /**
     * The headers every request and every socket handshake carries.
     *
     * `Stream-Auth-Type: jwt` says this caller is somebody's device rather than their backend.
     * It is what makes the router refuse the paths that configure an agent, and it is sent
     * even to a router with nothing in front of it, which would otherwise take the caller for
     * a backend.
     */
    internal suspend fun headers(): Map<String, String> {
        val headers = linkedMapOf("Stream-Auth-Type" to "jwt")
        val who = identity
        if (apiKey.isNotEmpty()) {
            if (who == null) {
                throw AgentsException.Configuration(
                    "an api key needs a user token to go with it; call setUser first",
                )
            }
            headers["X-Api-Key"] = apiKey
            headers["Authorization"] = "Bearer ${token(who)}"
            return headers
        }
        if (customerId.isEmpty()) {
            throw AgentsException.Configuration("pass a customerId, or an apiKey and setUser")
        }
        headers["X-Customer-Id"] = customerId
        if (who != null) headers["X-Stream-User-Id"] = who.user.id
        return headers
    }

    /**
     * Drops the token held, so the next request asks the provider again. Reports whether
     * there was a provider to ask, which is whether a retry could go any differently.
     */
    internal fun expireToken(): Boolean {
        cached = null
        return apiKey.isNotEmpty() && identity != null
    }

    /** The URL for a path under the router, with its segments encoded. */
    internal fun url(vararg segments: String, query: List<Pair<String, String>> = emptyList()): String =
        URLBuilder(url).apply {
            appendPathSegments(segments.toList(), encodeSlash = true)
            query.forEach { (name, value) -> parameters.append(name, value) }
        }.buildString()

    /**
     * The socket URL for a path under the router.
     *
     * Credentials go in the handshake's headers rather than the query string, which a phone
     * can do and a browser cannot: a token in a URL ends up in every log that URL passes
     * through.
     */
    internal fun socketUrl(vararg segments: String, query: List<Pair<String, String>>): String =
        URLBuilder(url(*segments, query = query)).apply {
            protocol = if (protocol == URLProtocol.HTTPS) URLProtocol.WSS else URLProtocol.WS
        }.buildString()

    override fun close() {
        http.close()
    }

    private suspend fun token(who: Identity): String {
        cached?.let { return it }
        return lock.withLock {
            // Asked once however many requests were waiting for it.
            cached ?: who.token.token().also { if (identity === who) cached = it }
        }
    }

    private class Identity(val user: User, val token: TokenProvider)
}
