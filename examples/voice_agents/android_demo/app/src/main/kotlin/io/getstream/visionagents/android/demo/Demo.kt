package io.getstream.visionagents.android.demo

import io.getstream.visionagents.android.core.AgentTool
import io.getstream.visionagents.android.core.AgentsException
import io.getstream.visionagents.android.core.User
import io.getstream.visionagents.android.core.VisionAgents
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * Where the router is, where this app's own backend is, and the tools it answers itself.
 *
 * There is no sign-in. The router is running in the mode where it trusts the customer and user
 * ids it is given, `ROUTER_AUTH_MODE=proxy`, so these are the whole of the configuration. In
 * front of a real deployment the router is reached by the key and the same token, and nothing
 * else here would change.
 */
object Demo {
    /** The router. The addresses and the key come from `local.properties`; see the README. */
    val routerUrl: String = BuildConfig.ROUTER_URL

    /**
     * This app's own backend, which is `go run ./backend` in swift_demo. A device holds no
     * secret to sign a Stream token with, so it asks this for one.
     */
    val backendUrl: String = BuildConfig.BACKEND_URL

    /**
     * Whichever customer id you started the router with. `compose.yaml` builds the dashboard
     * with `examples`, and a config belongs to one customer, so anything else here works but
     * will not appear on the dashboard.
     */
    const val CUSTOMER_ID: String = "examples"

    /**
     * The `STREAM_API_KEY` the router runs with: the chat channel a written conversation is
     * kept in and the call the agent is on are both in that Stream app, and this device
     * connects to them there.
     */
    val streamApiKey: String = BuildConfig.STREAM_API_KEY

    /**
     * The agent swift_demo's `go run ./configure` stored, by the name in its `agent.yaml`.
     *
     * The app is told which agent it talks to rather than finding out, because reading the
     * configs is server-side only.
     */
    const val AGENT_NAME: String = "swift_demo"

    /** Who this device is. A call waiting for approval is addressed to them. */
    val user: User = User(id = "demo-caller", name = "Demo caller")

    val agents: VisionAgents by lazy {
        VisionAgents(url = routerUrl, customerId = CUSTOMER_ID, apiKey = streamApiKey).apply {
            setUser(user) { streamToken() }
        }
    }

    private val http = OkHttpClient()
    private val json = Json { ignoreUnknownKeys = true }

    /** Asks the backend for a Stream user token, which joins the agent's call. */
    private suspend fun streamToken(): String = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("${backendUrl.trimEnd('/')}/stream-token")
            .post("""{"user_id":"${user.id}"}""".toRequestBody("application/json".toMediaType()))
            .build()
        http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw AgentsException.Unreadable("the backend would not mint a Stream token")
            }
            json.parseToJsonElement(response.body.string()).jsonObject.string("token")
        }
    }

    /** The tools the agent may call, which run on this phone. */
    val tools: List<AgentTool> by lazy { listOf(lookupOrder, issueRefund) }

    /**
     * Orders the agent can look up.
     *
     * The point of this being here rather than in the backend is that it does not have to
     * leave the phone. A real one would read whatever the signed-in person's session gives it;
     * the agent asks, and only ever sees the answer.
     */
    private val orders = mapOf(
        "A-1042" to "2 Larkspur wool throws, 78.00, paid by card ending 4242, " +
            "delivered on 14 August, unopened",
        "A-1043" to "1 linen apron, 24.00, paid by card ending 4242, " +
            "delivered on 2 September, worn",
    )

    private val lookupOrder = AgentTool(
        name = "lookup_order",
        description = "Look up one of the caller's orders by its order number, such as A-1042.",
        parameters = AgentTool.strings(
            mapOf("order_id" to "the order number, such as A-1042"),
            required = listOf("order_id"),
        ),
        displayTitle = "Looking up your order",
    ) { arguments ->
        val id = arguments.string("order_id").uppercase()
        val order = orders[id] ?: return@AgentTool "There is no order $id on this account."
        "Order $id: $order."
    }

    /**
     * Refunds an order, once the person says it may.
     *
     * The agent decides a refund is owed; the person decides whether it goes ahead. Each call
     * waits in the session's `approvals` until they answer, on the reply's step, where Stream's
     * AI components ask.
     */
    private val issueRefund = AgentTool(
        name = "issue_refund",
        description = "Refund one of the caller's orders to the card it was paid with, once the " +
            "refund skill has said they are owed it.",
        parameters = AgentTool.strings(
            mapOf(
                "order_id" to "the order number, such as A-1042",
                "amount" to "how much to refund, such as 78.00",
                "reason" to "why the caller is owed it, in a few words they would recognise",
            ),
            required = listOf("order_id", "amount", "reason"),
        ),
        displayTitle = "Refunding your order",
        approval = AgentTool.Approval(
            title = "Refund this order?",
            message = "The money goes back to the card the order was paid with.",
            reasonArgument = "reason",
            allowTitle = "Refund",
            declineTitle = "Not now",
        ),
    ) { arguments ->
        val id = arguments.string("order_id").uppercase()
        val amount = arguments.string("amount")
        if (id !in orders) return@AgentTool "There is no order $id on this account."
        "Refunded $amount for order $id to the card ending 4242. It shows within five working days."
    }

    private fun JsonObject.string(key: String): String = this[key]?.jsonPrimitive?.contentOrNull.orEmpty()
}
