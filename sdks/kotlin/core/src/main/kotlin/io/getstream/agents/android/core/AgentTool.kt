package io.getstream.agents.android.core

import io.getstream.agents.android.core.generated.SessionTool
import io.getstream.agents.android.core.generated.SessionToolApproval
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialInfo
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.SerialKind
import kotlinx.serialization.descriptors.StructureKind
import kotlinx.serialization.descriptors.elementNames
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import kotlinx.serialization.serializer

/**
 * A function of yours the agent can call.
 *
 * The agent runs in the backend but the tool runs here, which is the point: a tool can read the
 * signed-in user's data, or something only the phone knows, without any of it leaving the
 * device. The router asks over the session socket and waits for the answer.
 */
public class AgentTool private constructor(
    // First, so this constructor and the public one taking words differ on the JVM.
    /**
     * Runs the tool. What it returns is given to the model as the result; throwing tells the
     * model the tool failed and why. It is cancelled when the router gives up on the call.
     */
    public val run: suspend (arguments: JsonObject) -> ToolOutput,
    /** What the model calls it. Must be unique within a session. */
    public val name: String,
    /** What it is for, in words the model reads to decide whether to call it. */
    public val description: String,
    /** A JSON Schema object describing the arguments, or null for a tool that takes none. */
    public val parameters: JsonObject?,
    /** Who runs it, as the conversation shows it. Null leaves the router's default, [Executor.Server]. */
    public val executor: Executor?,
    /**
     * What a call is doing, in words for the people in the conversation, such as "Checking your
     * location". At most 80 characters, shown on the reply's tool call attachment.
     */
    public val displayTitle: String?,
    /**
     * What a person is asked before each call runs, or null to run every call straight away.
     * A call that asks waits in [AgentSession.approvals] until [AgentSession.decide] answers it.
     */
    public val approval: Approval?,
) {
    /** A tool that answers in words. For one that shows pictures too, see [withPictures]. */
    public constructor(
        name: String,
        description: String,
        parameters: JsonObject? = null,
        executor: Executor? = null,
        displayTitle: String? = null,
        approval: Approval? = null,
        run: suspend (arguments: JsonObject) -> String,
    ) : this(
        { ToolOutput(run(it)) }, name, description, parameters, executor, displayTitle, approval,
    )

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

    /**
     * What an argument of a [typed] tool is, in words the model reads to fill it in. On a class,
     * it describes the object wherever the class is an argument.
     */
    @OptIn(ExperimentalSerializationApi::class)
    @SerialInfo
    @Target(AnnotationTarget.PROPERTY, AnnotationTarget.CLASS)
    public annotation class Describe(public val description: String)

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
         * A tool that answers with pictures as well as words, such as a frame from the camera.
         *
         *     AgentTool.withPictures("photograph_room", "Take a photo of the room.") {
         *         ToolOutput("the room", listOf(ImageSource("data:image/jpeg;base64,...")))
         *     }
         */
        public fun withPictures(
            name: String,
            description: String,
            parameters: JsonObject? = null,
            executor: Executor? = null,
            displayTitle: String? = null,
            approval: Approval? = null,
            run: suspend (arguments: JsonObject) -> ToolOutput,
        ): AgentTool =
            AgentTool(run, name, description, parameters, executor, displayTitle, approval)

        /**
         * A tool whose arguments are a `@Serializable` class, which is also its schema.
         *
         * Strings, numbers, booleans, enums, lists and nested `@Serializable` classes are
         * described; a property with no default that is not nullable is required. Arguments
         * that do not decode tell the model why, as a tool that throws does.
         *
         *     @Serializable
         *     data class WeatherLookup(@AgentTool.Describe("a city or region") val location: String)
         *
         *     AgentTool.typed<WeatherLookup>("weather_lookup", "Look up the current weather.") {
         *         weather.today(it.location)
         *     }
         */
        public inline fun <reified T> typed(
            name: String,
            description: String,
            executor: Executor? = null,
            displayTitle: String? = null,
            approval: Approval? = null,
            noinline run: suspend (arguments: T) -> String,
        ): AgentTool = decoding(serializer<T>(), name, description, executor, displayTitle, approval, run)

        @PublishedApi
        internal fun <T> decoding(
            serializer: KSerializer<T>,
            name: String,
            description: String,
            executor: Executor?,
            displayTitle: String?,
            approval: Approval?,
            run: suspend (arguments: T) -> String,
        ): AgentTool {
            require(serializer.descriptor.kind == StructureKind.CLASS) {
                "A typed tool's arguments are a class, not ${serializer.descriptor.serialName}"
            }
            return AgentTool(
                { ToolOutput(run(wire.decodeFromJsonElement(serializer, it))) },
                name, description, schemaOf(serializer.descriptor), executor, displayTitle, approval,
            )
        }

        @OptIn(ExperimentalSerializationApi::class)
        private fun schemaOf(descriptor: SerialDescriptor, description: String? = descriptor.annotations.described): JsonObject =
            buildJsonObject {
                when (descriptor.kind) {
                    PrimitiveKind.STRING, PrimitiveKind.CHAR -> put("type", "string")
                    PrimitiveKind.BOOLEAN -> put("type", "boolean")
                    PrimitiveKind.BYTE, PrimitiveKind.SHORT, PrimitiveKind.INT, PrimitiveKind.LONG -> put("type", "integer")
                    PrimitiveKind.FLOAT, PrimitiveKind.DOUBLE -> put("type", "number")
                    SerialKind.ENUM -> {
                        put("type", "string")
                        putJsonArray("enum") { descriptor.elementNames.forEach { add(it) } }
                    }
                    StructureKind.LIST -> {
                        put("type", "array")
                        put("items", schemaOf(descriptor.getElementDescriptor(0)))
                    }
                    StructureKind.CLASS -> {
                        val elements = 0 until descriptor.elementsCount
                        put("type", "object")
                        putJsonObject("properties") {
                            for (i in elements) {
                                val element = descriptor.getElementDescriptor(i)
                                val described = descriptor.getElementAnnotations(i).described ?: element.annotations.described
                                put(descriptor.getElementName(i), schemaOf(element, described))
                            }
                        }
                        putJsonArray("required") {
                            for (i in elements) {
                                if (!descriptor.isElementOptional(i) && !descriptor.getElementDescriptor(i).isNullable) {
                                    add(descriptor.getElementName(i))
                                }
                            }
                        }
                    }
                    else -> throw IllegalArgumentException("${descriptor.serialName} cannot be a tool argument")
                }
                description?.let { put("description", it) }
            }

        private val List<Annotation>.described: String?
            get() = filterIsInstance<Describe>().firstOrNull()?.description

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

/**
 * What a tool gives the model: words, and pictures beside them, such as a photo the person
 * picked or a frame from the camera.
 *
 * The router hands a result's pictures to the agent's `vision` skill on a call, which looks at
 * them and tells the conversation what it found. With no such worker, as in a chat, a model that
 * sees images is shown them with its next reply alone; one that reads only text is told the tool
 * failed. Each picture is a data URI or an https URL, and one socket message holds at most 5 MB,
 * so shrink a photo before returning it.
 */
public data class ToolOutput(
    /** What the tool says. */
    val text: String = "",
    /** What the tool shows. */
    val images: List<ImageSource> = emptyList(),
)
