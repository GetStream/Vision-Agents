package io.getstream.agents.androidassistant.components

import android.content.Context
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponent
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponentDefinition
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import java.io.IOException

/**
 * The phone's own views the agent can place in the UI of a reply, such as a trip plan or a table
 * for dinner.
 *
 * What each one is, what the agent fills in and what it reports back are in UIComponents.json,
 * which the agent's instructions are written from too, so the agent knows them as the app does.
 * Each one loads what it shows on the phone: the place cards their photos from Wikipedia, and the
 * calendar the person's own days, which the agent never sees. Each wears the brand Settings chose,
 * through the OpenUI theme around it.
 */
object TripComponents {
    private const val DEFINITIONS = "UIComponents.json"

    /** Every component the app has a view for, as UIComponents.json describes them. */
    fun load(context: Context): List<AIUIComponent> = definitions(context).mapNotNull(::component)

    private fun definitions(context: Context): List<AIUIComponentDefinition> = try {
        val text = context.assets.open(DEFINITIONS).bufferedReader().use { it.readText() }
        (Json.parseToJsonElement(text) as? JsonArray).orEmpty()
            .mapNotNull { AIUIComponentDefinition.fromJsonString(it.toString()) }
    } catch (_: IOException) {
        emptyList()
    } catch (_: SerializationException) {
        emptyList()
    }

    private fun component(definition: AIUIComponentDefinition): AIUIComponent? = when (definition.name) {
        "place_card" -> placeCard(definition)
        "trip_map" -> tripMap(definition)
        "calendar_check" -> calendarCheck(definition)
        "restaurant_card" -> restaurantCard(definition)
        "availability_gauge" -> availabilityGauge(definition)
        "seat_selector" -> seatSelector(definition)
        else -> null
    }
}

// The arguments the agent wrote, as plain Kotlin values, read leniently: a value that is missing
// or of another type reads as null, as it may while the reply streams in.

internal fun Map<String, Any?>.string(key: String): String? = (this[key] as? String)?.takeIf { it.isNotEmpty() }

internal fun Map<*, *>.number(key: String): Double? = (this[key] as? Number)?.toDouble()

internal fun Map<String, Any?>.objects(key: String): List<Map<*, *>> =
    (this[key] as? List<*>).orEmpty().filterIsInstance<Map<*, *>>()
