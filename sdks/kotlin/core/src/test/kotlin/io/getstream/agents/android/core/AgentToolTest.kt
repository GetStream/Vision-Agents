package io.getstream.agents.android.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject

@Serializable
@AgentTool.Describe("a place on the map")
private data class Place(val city: String, val country: String? = null)

@Serializable
private enum class Units {
    @SerialName("metric") Metric,
    @SerialName("imperial") Imperial,
}

@Serializable
private data class Forecast(
    @AgentTool.Describe("where to forecast") val place: Place,
    @SerialName("days_ahead") val daysAhead: Int = 1,
    val units: Units,
    val note: String?,
    val tags: List<String> = emptyList(),
    val home: Place?,
    val score: Double,
    val strict: Boolean,
)

class AgentToolTest {
    private val forecast = AgentTool.typed<Forecast>("forecast", "Forecast the weather.") {
        "${it.place.city} ${it.daysAhead} ${it.units} ${it.note} ${it.home}"
    }

    @Test
    fun `a typed tool's schema is its class, required where nothing fills it in`() {
        val place = """{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city"]"""
        assertEquals(
            Json.parseToJsonElement(
                """{"type":"object","properties":{""" +
                    """"place":$place,"description":"where to forecast"},""" +
                    """"days_ahead":{"type":"integer"},""" +
                    """"units":{"type":"string","enum":["metric","imperial"]},""" +
                    """"note":{"type":"string"},""" +
                    """"tags":{"type":"array","items":{"type":"string"}},""" +
                    """"home":$place,"description":"a place on the map"},""" +
                    """"score":{"type":"number"},""" +
                    """"strict":{"type":"boolean"}},""" +
                    """"required":["place","units","score","strict"]}""",
            ).jsonObject,
            forecast.parameters,
        )
    }

    @Test
    fun `a typed tool runs on its decoded arguments, ignoring ones it does not know`() = runBlocking {
        val arguments = Json.parseToJsonElement(
            """{"place":{"city":"Boulder"},"days_ahead":3,"units":"imperial","score":0.5,"strict":true,"mood":"sunny"}""",
        ).jsonObject

        assertEquals(ToolOutput("Boulder 3 Imperial null null"), forecast.run(arguments))
    }
}
