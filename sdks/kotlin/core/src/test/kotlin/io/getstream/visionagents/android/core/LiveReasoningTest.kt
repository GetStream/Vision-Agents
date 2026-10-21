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
        reasoning.add("The order was ", 0, "reply", "r1")
        reasoning.add("delivered unopened.", 14, "reply", "r1")

        assertEquals("The order was delivered unopened.", reasoning["reply", "r1"])
    }

    @Test
    fun `a repeat of recent thinking adds only what is new`() {
        reasoning.add("Checking the policy", 0, "reply", "r1")

        reasoning.add("the policy, which allows 30 days", 9, "reply", "r1")
        reasoning.add("policy", 13, "reply", "r1")

        assertEquals("Checking the policy, which allows 30 days", reasoning["reply", "r1"])
    }

    @Test
    fun `a missed window leaves a paragraph break rather than running on`() {
        reasoning.add("First thought.", 0, "reply", "r1")

        reasoning.add("Much later.", 40, "reply", "r1")

        assertEquals("First thought.\n\nMuch later.", reasoning["reply", "r1"])
    }

    @Test
    fun `somebody joining midway starts from the window they get`() {
        reasoning.add("the rest of it", 300, "reply", "r2")

        assertEquals("the rest of it", reasoning["reply", "r2"])
        assertNull(reasoning["reply", "r1"])
    }

    @Test
    fun `positions count code points as the router does`() {
        reasoning.add("café ", 0, "reply", "r1")

        reasoning.add(" \uD83C\uDF38 bloom", 4, "reply", "r1")

        assertEquals("café \uD83C\uDF38 bloom", reasoning["reply", "r1"])
    }

    /**
     * Every reply numbers its steps from `r1`, so a later reply's first round is not the earlier
     * one's, whether it thinks for less or for more.
     */
    @Test
    fun `each reply keeps the thinking of its own steps`() {
        reasoning.add("Looking up the order history and the return policy.", 0, "first", "r1")

        reasoning.add("Refund it.", 0, "second", "r1")
        reasoning.add(
            "Checking which of the two orders arrived damaged, then the refund window.",
            0,
            "third",
            "r1",
        )

        assertEquals("Looking up the order history and the return policy.", reasoning["first", "r1"])
        assertEquals("Refund it.", reasoning["second", "r1"])
        assertEquals(
            "Checking which of the two orders arrived damaged, then the refund window.",
            reasoning["third", "r1"],
        )
    }
}
