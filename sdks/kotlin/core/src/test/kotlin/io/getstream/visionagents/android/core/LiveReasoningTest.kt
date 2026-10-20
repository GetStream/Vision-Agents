package io.getstream.visionagents.android.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * The windows of thinking the router puts on a reply's live updates, as `reasoningWindow` in
 * `acceleration/internal/conversation/reasoning.go` writes them.
 */
class LiveReasoningTest {
    private val reasoning = LiveReasoning()

    @Test
    fun `windows in order make the whole thinking`() {
        reasoning.add("The order was ", 0, "r1")
        reasoning.add("delivered unopened.", 14, "r1")

        assertEquals("The order was delivered unopened.", reasoning["r1"])
    }

    @Test
    fun `a repeat of recent thinking adds only what is new`() {
        reasoning.add("Checking the policy", 0, "r1")

        reasoning.add("the policy, which allows 30 days", 9, "r1")
        reasoning.add("policy", 13, "r1")

        assertEquals("Checking the policy, which allows 30 days", reasoning["r1"])
    }

    @Test
    fun `a missed window leaves a paragraph break rather than running on`() {
        reasoning.add("First thought.", 0, "r1")

        reasoning.add("Much later.", 40, "r1")

        assertEquals("First thought.\n\nMuch later.", reasoning["r1"])
    }

    @Test
    fun `somebody joining midway starts from the window they get`() {
        reasoning.add("the rest of it", 300, "r2")

        assertEquals("the rest of it", reasoning["r2"])
        assertNull(reasoning["r1"])
    }

    @Test
    fun `positions count code points as the router does`() {
        reasoning.add("café ", 0, "r1")

        reasoning.add(" \uD83C\uDF38 bloom", 4, "r1")

        assertEquals("café \uD83C\uDF38 bloom", reasoning["r1"])
    }
}
