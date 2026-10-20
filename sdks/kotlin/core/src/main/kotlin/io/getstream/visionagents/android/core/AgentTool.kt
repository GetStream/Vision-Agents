package io.getstream.visionagents.android.core

import io.getstream.visionagents.android.core.generated.SessionTool
import io.getstream.visionagents.android.core.generated.SessionToolApproval
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject

/**
 * A function of yours the agent can call.
 *
 * The agent runs in the backend but the tool runs here, which is the point: a tool can read the
 * signed-in user's data, or something only the phone knows, without any of it leaving the
 * device. The router asks over the session socket and waits for the answer.
 */
public class AgentTool(
    /** What the model calls it. Must be unique within a session. */
    public val name: String,
    /** What it is for, in words the model reads to decide whether to call it. */
    public val description: String,
    /** A JSON Schema object describing the arguments, or null for a tool that takes none. */
    public val parameters: JsonObject? = null,
    /** Who runs it, as the conversation shows it. Null leaves the router's default, [Executor.Server]. */
    public val executor: Executor? = null,
    /**
     * What a call is doing, in words for the people in the conversation, such as "Checking your
     * location". At most 80 characters, shown on the reply's tool call attachment.
     */
    public val displayTitle: String? = null,
    /**
     * What a person is asked before each call runs, or null to run every call straight away.
     * A call that asks waits in [AgentSession.approvals] until [AgentSession.decide] answers it.
     */
    public val approval: Approval? = null,
    /**
     * Runs the tool. What it returns is given to the model as the result; throwing tells the
     * model the tool failed and why. It is cancelled when the router gives up on the call.
     */
    public val run: suspend (arguments: JsonObject) -> String,
) {
    /** Who runs a tool. Either way, this session answers it over its socket. */
    public enum class Executor {
        Server,
        /**
         * A person's device: in a conversation kept in Stream Chat the call is shown as waiting
         * on the device of the person whose message it answers, arguments included, which
         * everybody in the channel can read.
         */
        Client,
    }

    /**
     * A tool's question to the person whose message a call answers.
     *
     * In a conversation kept in Stream Chat the call's step carries it as `awaiting_approval`,
     * and every channel member can read it: keep anything private out of it. A client tool is
     * asked about only once the router knows the install to address it to; until then, declare
     * it with [Executor.Server].
     */
    public data class Approval(
        /** The question, such as "Share your location?". At most 80 characters. */
        val title: String,
        /** What allowing it shares or does, such as "Only your city is shared." */
        val message: String? = null,
        /**
         * The string argument in which the model says why it wants the call, shown beside the
         * question. Declare it in `parameters` so the model fills it in.
         */
        val reasonArgument: String? = null,
        /** The label of the button that allows the call. */
        val allowTitle: String? = null,
        /** The label of the button that declines it. */
        val declineTitle: String? = null,
    )

    internal val schema: SessionTool
        get() = SessionTool(
            name = name,
            description = description,
            parameters = parameters,
            executor = when (executor) {
                Executor.Server -> SessionTool.Executor.SessionToolExecutorServer
                Executor.Client -> SessionTool.Executor.SessionToolExecutorClient
                null -> null
            },
            displayTitle = displayTitle,
            approval = approval?.let {
                SessionToolApproval(
                    title = it.title,
                    message = it.message,
                    reasonArgument = it.reasonArgument,
                    allowTitle = it.allowTitle,
                    declineTitle = it.declineTitle,
                )
            },
        )

    public companion object {
        /**
         * A JSON Schema object for a tool whose arguments are all strings.
         *
         * A convenience for the common shape, so declaring a tool does not mean writing out a
         * schema by hand. Anything else, write the object yourself.
         *
         *     AgentTool.strings(mapOf("location" to "the city, e.g. Boulder, CO"), required = listOf("location"))
         */
        public fun strings(properties: Map<String, String>, required: List<String> = emptyList()): JsonObject =
            buildJsonObject {
                put("type", "object")
                putJsonObject("properties") {
                    properties.forEach { (name, description) ->
                        putJsonObject(name) {
                            put("type", "string")
                            put("description", description)
                        }
                    }
                }
                putJsonArray("required") { required.forEach { add(it) } }
            }
    }
}
