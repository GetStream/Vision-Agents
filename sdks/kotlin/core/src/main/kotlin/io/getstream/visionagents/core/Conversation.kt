package io.getstream.visionagents.core

import java.util.UUID
import kotlin.time.Clock
import kotlin.time.Instant

/** One line of a conversation. */
public data class Turn(
    /** The router's turn id for an agent turn, and one of our own for a participant's. */
    val id: String,
    val speaker: Speaker,
    val text: String,
    val at: Instant,
) {
    public sealed interface Speaker {
        /** A person, on the call or typing. Null when the router did not say who. */
        public data class Person(val participant: Participant?) : Speaker

        /** The agent. */
        public data object Agent : Speaker
    }

    public val isAgent: Boolean get() = speaker == Speaker.Agent
}

/**
 * A conversation, as the events so far have left it.
 *
 * Immutable and with no network in it, which is the whole design: what a stream of frames
 * means for the transcript is decided in [reduce], and [AgentSession] is only the socket and
 * the `StateFlow` around it. Most new event handling belongs here, and is tested here.
 */
public data class Conversation(
    /**
     * The conversation so far, oldest first. The agent's turn in flight is the last entry and
     * grows as deltas arrive.
     */
    val turns: List<Turn> = emptyList(),
    /** What the agent is doing. */
    val state: State = State.Idle,
    /**
     * What the agent reported going wrong, or null. Errors arrive as events rather than being
     * thrown, because nobody is awaiting the socket.
     */
    val failure: String? = null,
) {
    public sealed interface State {
        /** Waiting to be spoken to. */
        public data object Idle : State

        /** Somebody is talking and being transcribed. */
        public data object Listening : State

        /** The model is answering. */
        public data object Responding : State

        /** Skills are thinking, named here so a view can say what about. */
        public data class Working(val skills: List<String>) : State

        /** The conversation is over. */
        public data object Ended : State
    }

    /** Whatever the caller typed, shown before the router has confirmed hearing it. */
    public fun said(text: String, at: Instant = Clock.System.now()): Conversation =
        copy(turns = turns + Turn(UUID.randomUUID().toString(), Turn.Speaker.Person(null), text, at))

    /**
     * Folds one event into the conversation.
     *
     * An event with no bearing on the transcript, and one this SDK has never heard of, both
     * leave it as it was.
     */
    public fun reduce(event: AgentEvent, at: Instant = Clock.System.now()): Conversation =
        when (event.kind) {
            // A text session echoes nothing back, so what the caller typed is already here. A
            // call transcribes what was spoken, which is the first anyone hears of it.
            AgentEvent.Kind.Heard -> {
                val echoed = turns.lastOrNull()?.let { !it.isAgent && it.text == event.text } ?: false
                val heard = Turn(UUID.randomUUID().toString(), Turn.Speaker.Person(event.participant), event.text, at)
                copy(turns = if (echoed) turns else turns + heard, state = State.Idle)
            }

            AgentEvent.Kind.Hearing -> copy(state = State.Listening)

            AgentEvent.Kind.Responding ->
                copy(turns = turns + Turn(event.turnId, Turn.Speaker.Agent, "", at), state = State.Responding)

            AgentEvent.Kind.ResponseDelta ->
                write(event.turnId, event.text, at) { it + event.text }.copy(state = State.Responding)

            // The final text is authoritative: the deltas are what was being written, this is
            // what was said. An empty one adds nothing, which is what a spoken-only turn is.
            AgentEvent.Kind.Responded ->
                (if (event.text.isEmpty()) this else write(event.turnId, event.text, at) { event.text })
                    .copy(state = State.Idle)

            AgentEvent.Kind.Delegated -> copy(state = State.Working(working + event.string("skill")))

            AgentEvent.Kind.TaskSettled, AgentEvent.Kind.TaskCancelled -> {
                val left = working - event.string("skill")
                copy(state = if (left.isEmpty()) State.Responding else State.Working(left))
            }

            AgentEvent.Kind.Interrupted -> copy(state = State.Idle)

            AgentEvent.Kind.Error -> copy(failure = event.errorText)

            AgentEvent.Kind.Left -> copy(state = State.Ended)

            else -> this
        }

    private val working: List<String>
        get() = (state as? State.Working)?.skills.orEmpty()

    /**
     * Changes the agent turn this event belongs to, starting one if the router sent text for a
     * turn we never saw begin.
     */
    private fun write(turnId: String, text: String, at: Instant, change: (String) -> String): Conversation {
        val index = turns.indexOfLast { it.id == turnId && it.isAgent }
        if (index < 0) return copy(turns = turns + Turn(turnId, Turn.Speaker.Agent, text, at))
        val updated = turns.toMutableList()
        updated[index] = updated[index].copy(text = change(updated[index].text))
        return copy(turns = updated)
    }
}
