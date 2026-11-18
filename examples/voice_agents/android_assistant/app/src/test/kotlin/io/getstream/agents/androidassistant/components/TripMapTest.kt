package io.getstream.agents.androidassistant.components

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** A place as the agent writes it in a trip map's arguments. */
private fun place(latitude: Any?, longitude: Any?, name: String = "Ohrid"): Map<String, Any?> =
    mapOf("name" to name, "latitude" to latitude, "longitude" to longitude)

class TripMapTest {
    @Test
    fun `a place on the globe is read with its coordinates`() {
        val place = TripPlace.from(place(41.1172, 20.8019))

        assertEquals(TripPlace("Ohrid", 41.1172, 20.8019), place)
    }

    @Test
    fun `the poles and the antimeridian are still on the globe`() {
        assertNotNull(TripPlace.from(place(90, 180)))
        assertNotNull(TripPlace.from(place(-90, -180)))
    }

    @Test
    fun `a latitude past a pole drops the place`() {
        assertNull(TripPlace.from(place(95, 20.8)))
        assertNull(TripPlace.from(place(-90.5, 20.8)))
    }

    @Test
    fun `a longitude past the antimeridian drops the place`() {
        assertNull(TripPlace.from(place(41.1, 181)))
        assertNull(TripPlace.from(place(41.1, -200.0)))
    }

    @Test
    fun `a coordinate that is not a number drops the place`() {
        assertNull(TripPlace.from(place(Double.NaN, 20.8)))
        assertNull(TripPlace.from(place(41.1, Double.POSITIVE_INFINITY)))
        assertNull(TripPlace.from(place("41.1", 20.8)))
    }

    @Test
    fun `no places have no region`() {
        assertNull(region(emptyList()))
    }

    @Test
    fun `one place is shown in a small region around it`() {
        val bounds = requireNotNull(region(listOf(TripPlace("Ohrid", 41.1172, 20.8019))))

        assertTrue(bounds.latitudeSouth < 41.1172 && 41.1172 < bounds.latitudeNorth)
        assertTrue(bounds.longitudeWest < 20.8019 && 20.8019 < bounds.longitudeEast)
        assertTrue(bounds.latitudeNorth - bounds.latitudeSouth < 0.1)
    }

    @Test
    fun `places in the same spot still have a region`() {
        val bounds = requireNotNull(region(List(3) { TripPlace("Ohrid $it", 41.1172, 20.8019) }))

        assertTrue(bounds.latitudeNorth > bounds.latitudeSouth)
        assertTrue(bounds.longitudeEast > bounds.longitudeWest)
    }

    @Test
    fun `places nearer a pole than the map reaches are shown at its edge`() {
        val north = requireNotNull(region(listOf(TripPlace("Svalbard", 89.5, 15.0), TripPlace("Pole", 90.0, 15.0))))
        val south = requireNotNull(region(listOf(TripPlace("Pole", -90.0, 0.0))))

        assertEquals(85.0, north.latitudeNorth)
        assertTrue(north.latitudeSouth < north.latitudeNorth)
        assertEquals(-85.0, south.latitudeSouth)
        assertTrue(south.latitudeSouth < south.latitudeNorth)
    }

    @Test
    fun `places either side of the antimeridian fit within the map`() {
        val bounds = requireNotNull(region(listOf(TripPlace("Fiji", -17.7, 179.9), TripPlace("Samoa", -13.8, -179.9))))

        assertEquals(-180.0, bounds.longitudeWest)
        assertEquals(180.0, bounds.longitudeEast)
    }
}
