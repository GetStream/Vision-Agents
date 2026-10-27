package io.getstream.visionagents.androidassistant

import io.getstream.visionagents.android.core.AgentsException
import io.getstream.visionagents.android.core.Sessions
import io.getstream.visionagents.android.core.User
import io.getstream.visionagents.android.core.VisionAgents
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * Where the router is, where the token backend is, and who this phone is.
 *
 * There is no sign-in. The router runs in the mode where it trusts the customer and user ids it
 * is given, `ROUTER_AUTH_MODE=proxy`, so these are the whole of the configuration. In front of a
 * real deployment the router is reached by the key and the same token, and nothing else here
 * would change.
 */
object Assistant {
    /** The router. The addresses and the key come from `local.properties`; see the README. */
    val routerUrl: String = BuildConfig.ROUTER_URL

    /**
     * This app's backend, which is `go run ./backend` in swift_assistant. A phone holds no secret
     * to sign a Stream token with, so it asks this for one.
     */
    val backendUrl: String = BuildConfig.BACKEND_URL

    /**
     * The scheme the router sends a sign-in back to, once the person has answered the provider
     * an agent's tool needs, such as Linear or Gmail. The router takes an app's own scheme only
     * in reverse domain form.
     */
    const val CALLBACK_SCHEME: String = "io.getstream.visionagents.androidassistant"

    /** Where the sign-ins a conversation asks for go back to, kept with its session. */
    const val RETURN_URL: String = "$CALLBACK_SCHEME:/connected"

    /** The customer swift_assistant's `go run ./configure` stored the agent under. */
    const val CUSTOMER_ID: String = "examples"

    /**
     * The `STREAM_API_KEY` the router runs with, which the conversations' channels are in.
     */
    val streamApiKey: String = BuildConfig.STREAM_API_KEY

    /** The agent swift_assistant's `go run ./configure` stored, by the name in its `agent.yaml`. */
    const val AGENT_NAME: String = "swift_assistant"

    /**
     * The model a written conversation runs on. Grok streams its reasoning, which is what fills
     * a reply's thinking panel. A call keeps the agent's faster model.
     */
    const val CHAT_MODEL: String = "xai/grok-4.7"

    /**
     * Who this phone is. Every conversation listed is theirs, and they are the same person the
     * iOS app is, so both phones list the same conversations.
     */
    val user: User = User(id = "juno-owner", name = "Alex")

    val agents: VisionAgents by lazy {
        VisionAgents(url = routerUrl, customerId = CUSTOMER_ID, apiKey = streamApiKey).apply {
            setUser(user) { streamToken() }
        }
    }

    /** The agent's conversations: opening one, and listing and searching the ones there were. */
    val sessions: Sessions get() = agents.agent(AGENT_NAME).sessions

    private val http = OkHttpClient()
    private val json = Json { ignoreUnknownKeys = true }

    /** Asks the backend for a Stream user token, which opens the chat. */
    private suspend fun streamToken(): String = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("${backendUrl.trimEnd('/')}/stream-token")
            .post("""{"user_id":"${user.id}"}""".toRequestBody("application/json".toMediaType()))
            .build()
        http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw AgentsException.Unreadable("the backend would not mint a Stream token")
            }
            json.parseToJsonElement(response.body.string()).jsonObject["token"]?.jsonPrimitive?.contentOrNull.orEmpty()
        }
    }
}
