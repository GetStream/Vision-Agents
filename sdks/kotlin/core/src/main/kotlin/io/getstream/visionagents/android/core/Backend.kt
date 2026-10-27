package io.getstream.visionagents.android.core

import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.websocket.WebSockets
import io.ktor.http.URLBuilder
import io.ktor.http.URLProtocol
import io.ktor.http.appendPathSegments
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
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

/** What Stream's own Chat and Video clients connect with: the app's key, the user and their token. */
public data class StreamCredentials(
    val apiKey: String,
    val user: User,
    val token: String,
)

/**
 * Where the router is and who is asking.
 *
 * A phone holds no secret worth having, so there is no API secret here. A router with nothing
 * in front of it is reached by [customerId]; a deployment verifying tokens is reached by
 * [apiKey] and the token [setUser] is given, which the app's own backend minted. Beside a
 * customer id the key is Stream's alone: the router is still reached by customer id, and the key
 * is what Chat and Video connect with.
 *
 * The key, user and token are the one identity for the router, Stream Chat and Stream Video.
 * Nothing connects when the user is set; each Stream client is built the first time a session
 * asks for it, once per key and user, and [VisionAgents.disconnect] closes it.
 *
 * Constructing one does no I/O. Pass an [OkHttpClient] to share the app's connection pool,
 * proxy and certificate pinning; [close] releases the HTTP client built around it.
 */
public class Backend(
    url: String = DEFAULT_URL,
    public val customerId: String = "",
    public val apiKey: String = "",
    okHttpClient: OkHttpClient? = null,
) : AutoCloseable {
    public companion object {
        /** Stream's hosted router, which is where a client goes when nothing says otherwise. */
        public const val DEFAULT_URL: String = "https://accelerate.gcp.stream-io-api.com"

        /**
         * The shape of a region name, such as us-east4 or europe-west4. Only the shape is
         * checked: which regions are served is the hosted router's to say, so offering another
         * needs no release of this SDK.
         */
        private val REGION = Regex("[a-z]+-[a-z]+[0-9]+")

        /**
         * Stream's hosted router held to one [region], such as us-east4: requests to it are
         * served there and nowhere else, even when that region is full or down.
         *
         * @throws IllegalArgumentException when [region] is not shaped like a region name.
         */
        public fun regionUrl(region: String): String {
            require(REGION.matches(region)) { "\"$region\" is not a region, such as us-east4" }
            return "https://accelerate-$region.gcp.stream-io-api.com"
        }
    }

    /** The router's base URL, with no trailing slash. */
    public val url: String = url.trimEnd('/')

    internal val http: HttpClient = (okHttpClient ?: OkHttpClient()).let { client ->
        HttpClient(OkHttp) {
            expectSuccess = false
            install(WebSockets)
            engine {
                preconfigured = client
                webSocketFactory = Handshakes(client)
            }
        }
    }

    private val lock = Mutex()
    @Volatile private var identity: Identity? = null
    @Volatile private var cached: String? = null

    /** Opens shared clients outside any one caller, so a session going away does not cancel another's. */
    private val opening = CoroutineScope(SupervisorJob())
    /** The app's own clients by kind, kept whoever signs in. */
    private val given = ConcurrentHashMap<String, Any>()
    /** The clients built here, by kind, key and user. */
    private val built = mutableMapOf<String, Built>()

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
     * The key, user and token for Stream Chat, Stream Video or any other Stream product.
     *
     * [refresh] is for a token the caller was told has expired: the one held is fetched again,
     * unless somebody else already did, so two clients refreshing at once fetch one token.
     */
    public suspend fun streamCredentials(refresh: Boolean = false): StreamCredentials {
        val who = identity
        if (apiKey.isEmpty() || who == null) {
            throw AgentsException.Configuration(
                "Stream Chat and Video need the app's Stream key and a user: build VisionAgents " +
                    "with apiKey and call setUser",
            )
        }
        val token = token(who, stale = if (refresh) cached ?: "" else null)
        return StreamCredentials(apiKey, who.user, token)
    }

    /** Records a Stream client the app owns, so it is used rather than another being built. */
    internal fun give(kind: String, client: Any) {
        given[kind] = client
    }

    /**
     * The client the app gave for [kind], or the one built for the current key and user.
     *
     * Built once however many sessions ask, keyed before any token is asked for, so a second
     * session mints nothing. One that failed to open is not kept.
     */
    @Suppress("UNCHECKED_CAST")
    internal suspend fun <T : Any> shared(
        kind: String,
        open: suspend (StreamCredentials) -> T,
        disconnect: suspend (T) -> Unit,
    ): T {
        given[kind]?.let { return it as T }
        val who = identity
        if (apiKey.isEmpty() || who == null) {
            throw AgentsException.Configuration(
                "$kind connects to Stream rather than to the router, so it needs the app's Stream " +
                    "key and a user: build VisionAgents with apiKey and call setUser",
            )
        }
        val key = "$kind:$apiKey:${who.user.id}"
        val entry = synchronized(built) {
            built.getOrPut(key) {
                Built(opening.async { open(streamCredentials()) }) { disconnect(it as T) }
            }
        }
        try {
            return entry.opening.await() as T
        } catch (e: Exception) {
            // Only a client that failed to open is forgotten. A caller cancelled while it was
            // opening leaves it to whoever asks next, or a second would be built beside it.
            if (entry.opening.isCompleted) {
                synchronized(built) { if (built[key] === entry) built.remove(key) }
            }
            throw e
        }
    }

    /** Disconnects every Stream client built here. The ones the app gave are left alone. */
    internal suspend fun disconnectStream() {
        val opened = synchronized(built) { built.values.toList().also { built.clear() } }
        for (entry in opened) {
            val client = try {
                entry.opening.await()
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                // One that never opened has nothing to disconnect, and its opener was told why.
                continue
            }
            entry.disconnect(client)
        }
    }

    /**
     * The headers every request and every socket handshake carries.
     *
     * `Stream-Auth-Type: jwt` says this caller is somebody's device rather than their backend.
     * It is what makes the router refuse the paths that configure an agent, and it is sent
     * even to a router with nothing in front of it, which would otherwise take the caller for
     * a backend. The key travels in the query instead, where both Stream's proxy and the router
     * read it.
     */
    internal suspend fun headers(): Map<String, String> {
        val headers = linkedMapOf("Stream-Auth-Type" to "jwt")
        val who = identity
        if (customerId.isNotEmpty()) {
            headers["X-Customer-Id"] = customerId
            if (who != null) headers["X-Stream-User-Id"] = who.user.id
            return headers
        }
        if (apiKey.isEmpty()) {
            throw AgentsException.Configuration("pass a customerId, or an apiKey and setUser")
        }
        if (who == null) {
            throw AgentsException.Configuration(
                "an api key needs a user token to go with it; call setUser first",
            )
        }
        headers["Authorization"] = "Bearer ${token(who)}"
        return headers
    }

    /**
     * Whether this device holds the app's key and nobody to act for yet, which is how a guest is
     * asked for before there is one: `Stream-Auth-Type: anonymous` and no token.
     */
    internal val anonymous: Boolean
        get() = customerId.isEmpty() && apiKey.isNotEmpty() && identity == null

    /**
     * Drops the token held, so the next request asks the provider again. Reports whether
     * there was a provider to ask, which is whether a retry could go any differently.
     */
    internal fun expireToken(): Boolean {
        cached = null
        return customerId.isEmpty() && apiKey.isNotEmpty() && identity != null
    }

    /**
     * The URL for a path under the router, with its segments encoded and the API key, the
     * public half of the credential, in the query.
     */
    internal fun url(vararg segments: String, query: List<Pair<String, String>> = emptyList()): String =
        URLBuilder(url).apply {
            appendPathSegments(segments.toList(), encodeSlash = true)
            if (customerId.isEmpty() && apiKey.isNotEmpty()) parameters.append("api_key", apiKey)
            query.forEach { (name, value) -> parameters.append(name, value) }
        }.buildString()

    /**
     * The socket URL for a path under the router.
     *
     * The token goes in the handshake's headers rather than the query string, which a phone
     * can do and a browser cannot: a token in a URL ends up in every log that URL passes
     * through.
     */
    internal fun socketUrl(vararg segments: String, query: List<Pair<String, String>>): String =
        URLBuilder(url(*segments, query = query)).apply {
            protocol = if (protocol == URLProtocol.HTTPS) URLProtocol.WSS else URLProtocol.WS
        }.buildString()

    override fun close() {
        opening.cancel()
        http.close()
    }

    /** The token held, or a fresh one when the one held is [stale]. */
    private suspend fun token(who: Identity, stale: String? = null): String {
        if (stale == null) cached?.let { return it }
        return lock.withLock {
            // Asked once however many requests were waiting for it.
            cached?.takeIf { it != stale } ?: who.token.token().also { if (identity === who) cached = it }
        }
    }

    private class Identity(val user: User, val token: TokenProvider)

    private class Built(val opening: Deferred<Any>, val disconnect: suspend (Any) -> Unit)
}
