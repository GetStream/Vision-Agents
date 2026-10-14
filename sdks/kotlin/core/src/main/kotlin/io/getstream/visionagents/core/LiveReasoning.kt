package io.getstream.visionagents.core

/**
 * A reply's thinking, put back together from its live updates.
 *
 * The router stores only the opening of each round of reasoning, as its step's `preview`. The
 * whole of it reaches people watching the reply live: each update carries the next window of
 * the round being streamed, in the message's `reasoning` field. Read every update into one of
 * these (`read(message)` in the chat module does it for a Stream Chat message), from the
 * client's message-updated events rather than the channel's message list, which can fold
 * several updates into one, and show a step by its id.
 */
public class LiveReasoning {
    private class Step(var text: String = "", var end: Int = 0)

    private val steps = mutableMapOf<String, Step>()

    /** The thinking of reasoning step [stepId] so far, or null before any of it arrived. */
    public operator fun get(stepId: String): String? = synchronized(steps) { steps[stepId]?.text }

    /**
     * Adds what [text], which starts [offset] code points into step [stepId]'s thinking, has
     * that is not held yet. A window that repeats recent thinking, for somebody who joined
     * midway, overlaps what is held; one that starts past it means updates were missed, and is
     * kept after a paragraph break rather than run on.
     */
    public fun add(text: String, offset: Int, stepId: String) {
        synchronized(steps) {
            val step = steps.getOrPut(stepId) { Step() }
            val end = offset + text.codePointCount(0, text.length)
            if (end <= step.end) return
            if (offset > step.end && step.text.isNotEmpty()) step.text += "\n\n"
            step.text += text.substring(text.offsetByCodePoints(0, maxOf(0, step.end - offset)))
            step.end = end
        }
    }
}
