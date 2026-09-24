package io.getstream.visionagents.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class WireTest {
    @Test
    fun `an event the SDK does not know still reaches the caller whole`() {
        val event = assertNotNull(AgentEvent.decode("""{"type":"astonished","degree":11,"why":{"because":"yes"}}"""))

        assertEquals("astonished", event.type)
        assertNull(event.kind)
        assertEquals(11, event["degree"].jsonPrimitive.int)
        assertEquals("yes", event["why"].jsonObject["because"]?.jsonPrimitive?.content)
    }

    @Test
    fun `a frame that is not an event is skipped rather than fatal`() {
        assertNull(AgentEvent.decode("not json"))
        assertNull(AgentEvent.decode("""["an","array"]"""))
        assertNull(AgentEvent.decode("""{"no":"type"}"""))
        assertNull(AgentEvent.decode("""{"type":7}"""))
    }

    @Test
    fun `a tool call carries what its answer has to repeat`() {
        val event = assertNotNull(
            AgentEvent.decode(
                """{"type":"tool_call","id":"c1","name":"lookup_order","arguments":"{\"order_id\":\"A-1042\"}","command_id":"cmd","turn_id":"t1"}""",
            ),
        )

        val call = assertNotNull(event.toolCall)
        assertEquals("lookup_order", call.name)
        assertEquals("A-1042", call.argumentValues["order_id"]?.jsonPrimitive?.content)
        assertEquals("cmd", call.commandId)
        assertEquals("t1", call.turnId)
    }

    @Test
    fun `arguments a model mangled decode to nothing rather than throwing`() {
        val call = AgentEvent.ToolCall("c1", "t", "{not json", "", "")

        assertEquals(JsonObject(emptyMap()), call.argumentValues)
    }

    @Test
    fun `commands are the frames readCommands takes`() {
        fun frame(command: Command) = Json.parseToJsonElement(command.encode()).jsonObject

        assertEquals("""{"type":"respond","text":"hi"}""", Command.Respond("hi").encode())
        assertEquals("""{"type":"interrupt"}""", Command.Interrupt().encode())
        assertEquals("""{"type":"interrupt","command_id":"cmd-1"}""", Command.Interrupt("cmd-1").encode())
        assertEquals("""{"type":"close"}""", Command.Close.encode())
        assertEquals("""{"type":"say","text":"one moment"}""", Command.Say("one moment").encode())
        assertEquals("""{"type":"instructions","instructions":"be brief"}""", Command.Instructions("be brief").encode())

        val result = frame(Command.ToolResult("c1", output = "done", turnId = "t1"))
        assertEquals(JsonPrimitive("tool_result"), result["type"])
        assertEquals(JsonPrimitive("c1"), result["tool_call_id"])
        assertEquals(JsonPrimitive("done"), result["output"])
        assertEquals(JsonPrimitive("t1"), result["turn_id"])
        assertNull(result["command_id"])

        val withImage = frame(Command.Respond("what is this", listOf(ImageSource("https://x/y.png", ImageSource.Detail.Low))))
        assertEquals("""[{"url":"https://x/y.png","detail":"low"}]""", withImage["images"].toString())
    }

    @Test
    fun `timestamps are read however many fractional digits Go wrote`() {
        assertEquals(instant("2026-09-24T04:03:19Z"), instant("2026-09-24T04:03:19.000000000Z"))
        assertEquals(979138000, instant("2026-09-24T04:03:19.979138Z").nanosecondsOfSecond)
        assertEquals(instant("2026-09-24T04:03:19.5Z"), instant("2026-09-24T06:03:19.5+02:00"))
        assertFailsWith<AgentsException.Unreadable> { instant("yesterday") }
    }

    @Test
    fun `a tool whose arguments are all strings gets a schema without writing one`() {
        val schema = AgentTool.strings(mapOf("order_id" to "the order number"), required = listOf("order_id"))

        assertEquals(
            """{"type":"object","properties":{"order_id":{"type":"string","description":"the order number"}},"required":["order_id"]}""",
            schema.toString(),
        )
    }
}
