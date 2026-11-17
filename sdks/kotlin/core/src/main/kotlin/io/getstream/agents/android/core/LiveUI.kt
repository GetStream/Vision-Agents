package io.getstream.agents.android.core

/**
 * The UI the session's socket streamed for each agent turn, to show on the turn's reply in Stream
 * Chat while it is written.
 *
 * The router writes a reply's UI on its Stream Chat message as `ai_ui` a few times a second, and
 * on the socket as the model writes it. Give every change of the session's conversation to
 * [observe], and show each reply with the custom data [overlay] answers for it, so a card draws at
 * the socket's speed and Chat's copy takes over once it is complete.
 */
public class LiveUI {
    private val byTurn = mutableMapOf<String, TurnUI>()

    /**
     * The UI of the turn the agent is on: the conversation's newest turn's, when it is the
     * agent's.
     */
    public var latest: TurnUI? = null
        get() = synchronized(byTurn) { field }
        private set

    /** The UI the socket streamed for agent turn [turnId], or null. */
    public operator fun get(turnId: String): TurnUI? = synchronized(byTurn) { byTurn[turnId] }

    /**
     * Takes in the UI of [conversation]'s agent turns, and says whether what it shows changed.
     * The person's question and the agent's new turn clear [latest], so a new reply never shows
     * the card of the one before it.
     */
    public fun observe(conversation: Conversation): Boolean =
        synchronized(byTurn) {
            var changed = false
            for (turn in conversation.turns) {
                val ui = turn.ui?.takeIf { turn.isAgent } ?: continue
                if (byTurn[turn.id] == ui) continue
                byTurn[turn.id] = ui
                changed = true
            }
            val newest = conversation.turns.lastOrNull()?.takeIf { it.isAgent }?.ui
            if (newest != latest) {
                latest = newest
                changed = true
            }
            changed
        }

    /**
     * The custom data of each agent reply that shows the socket's UI, by message id, from each
     * message's id and custom data, oldest first.
     *
     * A reply Chat is still writing (`generating`), whose own `ai_ui` is not complete, shows its
     * turn's UI, matched by the `turn_id` the router writes on it. Before its turn id is in Chat,
     * the newest reply being written shows [latest]. Once Chat says a reply is written, what Chat
     * stored shows. A UI with no statement finished yet leaves the reply its text.
     */
    internal fun overlayCustom(messages: List<Pair<String, Map<String, Any?>>>): Map<String, Map<String, Any?>> =
        synchronized(byTurn) {
            if (byTurn.isEmpty()) return emptyMap()
            val generating = messages.lastOrNull { (_, custom) ->
                isAgentReply(custom) && custom["generating"] == true
            }?.first
            buildMap {
                for ((id, custom) in messages) {
                    if (!isAgentReply(custom) || !showsLiveUi(custom)) continue
                    val turnId = turnId(custom)
                    val ui = if (turnId != null) byTurn[turnId] else latest.takeIf { id == generating }
                    if (ui == null || ui.content.isEmpty()) continue
                    put(id, custom + ("ai_ui" to ui.customData))
                }
            }
        }

    private companion object {
        fun isAgentReply(custom: Map<String, Any?>): Boolean =
            custom["ai_generated"] == true || runtime(custom)?.get("role") == "assistant"

        /**
         * Whether a reply shows the socket's UI rather than what Chat stored: while Chat says it
         * is being written and its UI isn't complete there.
         */
        fun showsLiveUi(custom: Map<String, Any?>): Boolean =
            custom["generating"] == true && (custom["ai_ui"] as? Map<*, *>)?.get("status") != "complete"

        fun turnId(custom: Map<String, Any?>): String? =
            (custom["turn_id"] as? String ?: runtime(custom)?.get("turn_id") as? String)?.takeIf { it.isNotEmpty() }

        fun runtime(custom: Map<String, Any?>): Map<*, *>? = custom["support_runtime"] as? Map<*, *>
    }
}
