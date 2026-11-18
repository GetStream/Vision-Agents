package io.getstream.agents.androidassistant.components

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

private const val HOUR = 60 * 60 * 1000L
private const val DAY = 24 * HOUR

/** Three days' starts, a day apart, and when the last one ends. */
private val starts = listOf(0L, DAY, 2 * DAY)
private val last = 3 * DAY

class CalendarCheckTest {
    @Test
    fun `a calendar that was not read leaves no day free`() {
        val lookup = CalendarDay.lookup(starts, last, events = null)

        assertEquals(CalendarLookup.Unavailable, lookup)
    }

    @Test
    fun `a calendar read with no events leaves every day free`() {
        val lookup = assertIs<CalendarLookup.Days>(CalendarDay.lookup(starts, last, events = emptyList()))

        assertEquals(listOf(0L, DAY, 2 * DAY), lookup.days.map { it.date.time })
        assertEquals(listOf(emptyList<String>(), emptyList(), emptyList()), lookup.days.map { it.events })
    }

    @Test
    fun `an event is on its own day only`() {
        val dentist = CalendarDay.Companion.Event("Dentist", DAY + 9 * HOUR, DAY + 10 * HOUR, allDay = false)

        val lookup = assertIs<CalendarLookup.Days>(CalendarDay.lookup(starts, last, events = listOf(dentist)))

        assertEquals(listOf(0, 1, 0), lookup.days.map { it.events.size })
        assertEquals(dentist.text, lookup.days[1].events.single())
    }
}
