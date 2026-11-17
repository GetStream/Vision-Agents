package io.getstream.agents.android.core

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
    /**
     * The interactive UI an agent turn wrote, as it streams in, or null for a turn without any.
     * OpenUI is null until one of its statements is finished.
     */
    val ui: TurnUI? = null,
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
 *
 * A value, compared by what it holds, though not a data class: it also keeps what each turn's UI
 * wrote, which nobody outside sets or reads.
 */
public class Conversation private constructor(
    /**
     * The conversation so far, oldest first. The agent's turn in flight is the last entry and
     * grows as deltas arrive.
     */
    public val turns: List<Turn>,
    /** What the agent is doing. */
    public val state: State,
    /**
     * What the agent reported going wrong, or null. Errors arrive as events rather than being
     * thrown, because nobody is awaiting the socket. It lasts until the agent shows it works
     * again: it next hears, starts answering or speaks.
     */
    public val failure: String?,
    /**
     * What each agent turn's UI block wrote so far, by turn id: the deltas, or a patch before it
     * is merged, until the turn's UI is settled. Kept apart from [Turn.ui], which is what shows
     * and is left unchanged by a delta that shows nothing new, and counted in equality so that
     * such a delta is still kept.
     */
    private val uiWritten: Map<String, UiWritten>,
) {
    public constructor(
        turns: List<Turn> = emptyList(),
        state: State = State.Idle,
        failure: String? = null,
    ) : this(turns, state, failure, emptyMap())

    /** This conversation with the given parts changed, and the UI being written kept. */
    public fun copy(
        turns: List<Turn> = this.turns,
        state: State = this.state,
        failure: String? = this.failure,
    ): Conversation = Conversation(turns, state, failure, uiWritten)

    public operator fun component1(): List<Turn> = turns

    public operator fun component2(): State = state

    public operator fun component3(): String? = failure

    override fun equals(other: Any?): Boolean =
        this === other ||
            other is Conversation &&
            turns == other.turns &&
            state == other.state &&
            failure == other.failure &&
            uiWritten == other.uiWritten

    override fun hashCode(): Int = listOf(turns, state, failure, uiWritten).hashCode()

    override fun toString(): String = "Conversation(turns=$turns, state=$state, failure=$failure)"

    private fun writing(uiWritten: Map<String, UiWritten>): Conversation =
        Conversation(turns, state, failure, uiWritten)

    /** What a turn's UI block wrote, and how. */
    private data class UiWritten(val format: String, val version: String?, val mode: String, val text: String) {
        /** [text], written as the `ui_delta` or `ui_completed` [event] says. */
        constructor(event: AgentEvent, format: String, text: String) :
            this(format, event.string("version").ifEmpty { null }, event.string("mode"), text)
    }

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
        (if (failure != null && event.kind in recovered) copy(failure = null) else this).fold(event, at)

    private fun fold(event: AgentEvent, at: Instant): Conversation =
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

            AgentEvent.Kind.UiDelta -> {
                val delta = event.string("delta")
                val format = event.string("format")
                if (delta.isEmpty() || format.isEmpty()) {
                    this
                } else {
                    val text = uiWritten[event.turnId]?.text.orEmpty() + delta
                    val written = UiWritten(event, format, text)
                    val kept = writing(uiWritten + (event.turnId to written))
                    // Only the line still being written grew, so no statement finished: what shows
                    // is unchanged.
                    val shown = if (format == TurnUI.OPEN_UI && '\n' !in delta) {
                        kept
                    } else {
                        kept.writeUi(event.turnId, written, TurnUI.Keep.Finished, at)
                    }
                    shown.let { if (it.state is State.Working) it else it.copy(state = State.Responding) }
                }
            }

            // The whole of it is authoritative, as a reply's final text is. One with no content
            // keeps what streamed.
            AgentEvent.Kind.UiCompleted -> {
                val format = event.string("format")
                if (format.isEmpty()) {
                    this
                } else {
                    val text = event.string("content").ifEmpty { uiWritten[event.turnId]?.text.orEmpty() }
                    val written = UiWritten(event, format, text)
                    writing(uiWritten - event.turnId).writeUi(event.turnId, written, TurnUI.Keep.All, at)
                }
            }

            // The final text is authoritative: the deltas are what was being written, this is
            // what was said. An empty one adds nothing, which is what a spoken-only turn is.
            AgentEvent.Kind.Responded ->
                settleUi(event.turnId, at)
                    .let { if (event.text.isEmpty()) it else it.write(event.turnId, event.text, at) { event.text } }
                    .copy(state = State.Idle)

            AgentEvent.Kind.Delegated -> copy(state = State.Working(working + event.string("skill")))

            AgentEvent.Kind.TaskSettled, AgentEvent.Kind.TaskCancelled -> {
                val left = working - event.string("skill")
                copy(state = if (left.isEmpty()) State.Responding else State.Working(left))
            }

            AgentEvent.Kind.Interrupted -> settleUi(event.turnId, at).copy(state = State.Idle)

            // The model writing the reply failing is the end of that reply, which may never be
            // reported as responded. Anything else that fails, a tool, the voice or the
            // transcriber, leaves the reply going, and the agent doing what it was doing.
            AgentEvent.Kind.Error ->
                copy(failure = event.errorText, state = if (endsTurn(event)) State.Idle else state)

            AgentEvent.Kind.Left -> copy(state = State.Ended)

            else -> this
        }

    private val working: List<String>
        get() = (state as? State.Working)?.skills.orEmpty()

    /** Whether an error ends the agent's turn in flight: it is the reply model's, and there is one. */
    private fun endsTurn(event: AgentEvent): Boolean =
        (state == State.Responding || state is State.Working) && event.string("context") in replyModels

    private companion object {
        /** The `context` of an error from the model writing the reply, as text or as speech. */
        val replyModels = setOf("llm", "sts")

        /** The events that show the agent works again, so whatever it last reported is over. */
        val recovered = setOf(AgentEvent.Kind.Heard, AgentEvent.Kind.Responding, AgentEvent.Kind.Spoke)
    }

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

    /**
     * Settles the UI of a turn that ended with no `ui_completed`, as an interrupted reply does:
     * the router dropped its UI block with the turn. What was written shows, but for a statement
     * it ended in the middle of, and the UI is complete, since nothing more of it will come.
     */
    private fun settleUi(turnId: String, at: Instant): Conversation {
        val written = uiWritten[turnId] ?: return this
        return writing(uiWritten - turnId).writeUi(turnId, written, TurnUI.Keep.Closed, at)
    }

    /**
     * Shows [written], what the turn's UI block wrote, as the UI of agent turn [turnId], starting
     * the turn if the router sent UI for one we never saw begin. Of OpenUI, only the statements
     * [keep] says count; it is streaming while only the finished ones do, and sets no UI until
     * one counts. A patch is merged onto the UI of the latest earlier turn in the same format;
     * with none in this transcript, such as after a reload, it sets no UI, and Stream Chat's copy,
     * which the router merged, is the one to show.
     */
    private fun writeUi(turnId: String, written: UiWritten, keep: TurnUI.Keep, at: Instant): Conversation {
        val openUi = written.format == TurnUI.OPEN_UI
        var content = if (openUi) TurnUI.kept(written.text, keep) else written.text
        if (openUi && keep != TurnUI.Keep.All && content.isEmpty()) return this
        if (written.mode == "patch" && openUi) {
            val base = turns.lastOrNull {
                it.isAgent && it.id != turnId && it.ui?.format == TurnUI.OPEN_UI && it.ui.content.isNotEmpty()
            }?.ui ?: return this
            content = TurnUI.merge(base.content, content)
        }
        val ui = TurnUI(written.format, written.version, keep == TurnUI.Keep.Finished, content)
        val index = turns.indexOfLast { it.id == turnId && it.isAgent }
        if (index < 0) return copy(turns = turns + Turn(turnId, Turn.Speaker.Agent, "", at, ui))
        val updated = turns.toMutableList()
        updated[index] = updated[index].copy(ui = ui)
        return copy(turns = updated)
    }
}

/**
 * The interactive UI an agent turn writes, such as a card in OpenUI Lang, as the session's socket
 * streams it: `ui_delta` as the model writes it, before stage directions are taken out or
 * anything is spoken, then `ui_completed`.
 *
 * The router writes the same UI on the reply's message in Stream Chat as `ai_ui`, later and less
 * often; [LiveUI] shows this one on the message until that one is complete. A call screen, which
 * shows no channel and speaks none of it, shows this one itself.
 */
public data class TurnUI(
    /** The format it is written in, such as `openui` or `a2ui`. */
    val format: String,
    /** The format's version, when the model named one. */
    val version: String? = null,
    /** Whether the model is still writing it. */
    val isStreaming: Boolean,
    /**
     * The UI to show. While OpenUI is written, that is its finished statements: one still being
     * written shows once a newline ends it, so the UI grows a part at a time rather than changing
     * with every delta, such as a gauge reading 8 before 86. A patch, which changes only some
     * statements of the turn before's OpenUI, is already merged onto it here.
     */
    val content: String,
) {
    /**
     * The UI as a message's `ai_ui` custom field holds it, in plain Kotlin values, the shape
     * Stream Chat's `AIUIDocument.from` reads under that key.
     */
    public val customData: Map<String, Any?>
        get() = buildMap {
            put("v", 1)
            put("format", format)
            put("status", if (isStreaming) "streaming" else "complete")
            put("content", content)
            version?.let { put("version", it) }
        }

    /**
     * Which statements of an OpenUI program count: all of them; those closed, leaving out one a
     * bracket or a string left open; or those finished, also leaving out the line no newline has
     * ended, which is still being written.
     */
    internal enum class Keep { All, Closed, Finished }

    public companion object {
        /** The format OpenUI Lang is written in, the one whose patches are merged. */
        public const val OPEN_UI: String = "openui"

        /**
         * An OpenUI program with [patch] applied, as the router merges one: a statement the patch
         * assigns again is replaced where it was, a new one goes at the end, and a component
         * assigned `null` is removed. A `$variable` assigned `null` keeps the `null`.
         */
        internal fun merge(base: String, patch: String): String {
            if (base.isBlank()) return patch
            if (patch.isBlank()) return base
            // Insertion-ordered, and replacing a name's statement keeps its place.
            val statements = LinkedHashMap<String, String>()
            for (statement in statements(base)) {
                val (name, _) = assignment(statement) ?: continue
                statements[name] = statement
            }
            for (statement in statements(patch)) {
                val (name, value) = assignment(statement) ?: continue
                if (value == "null" && !name.startsWith("$")) {
                    statements.remove(name)
                } else {
                    statements[name] = "$name = $value"
                }
            }
            return statements.values.joinToString("\n")
        }

        /** The statements of [program] that [keep] says count, or all of it as written. */
        internal fun kept(program: String, keep: Keep): String =
            if (keep == Keep.All) program else statements(program, keep).joinToString("\n")

        /**
         * A program's statements: each line, or the lines a bracket or a string left open joined
         * to the one that closes it. Blank and comment lines are left out, and so is what [keep]
         * leaves out.
         */
        private fun statements(program: String, keep: Keep = Keep.All): List<String> {
            val statements = mutableListOf<String>()
            val current = StringBuilder()
            var quote: Char? = null
            var escaped = false
            var depth = 0
            // What follows the last newline is still being written.
            val lines = program.split("\n").let { if (keep == Keep.Finished) it.dropLast(1) else it }
            for (line in lines) {
                if (quote == null && depth == 0) {
                    val trimmed = line.trim(' ', '\t')
                    if (trimmed.isEmpty() || trimmed.startsWith("//") || trimmed.startsWith("#")) continue
                }
                if (current.isNotEmpty()) current.append('\n')
                current.append(line)
                var previous: Char? = null
                for (character in line) {
                    val before = previous
                    previous = character
                    if (escaped) {
                        escaped = false
                        continue
                    }
                    val open = quote
                    if (open != null) {
                        if (character == '\\') escaped = true else if (character == open) quote = null
                        continue
                    }
                    if (character == '/' && before == '/') break
                    when (character) {
                        '"', '\'' -> quote = character
                        '(', '[', '{' -> depth += 1
                        ')', ']', '}' -> depth = maxOf(0, depth - 1)
                    }
                }
                if (quote == null && depth == 0) {
                    val finished = current.trim()
                    if (finished.isNotEmpty()) statements += finished.toString()
                    current.clear()
                }
            }
            val tail = current.trim()
            if (tail.isNotEmpty() && keep == Keep.All) statements += tail.toString()
            return statements
        }

        /** Whether [name] can be assigned: an identifier, or a `$variable`. */
        private fun isName(name: String): Boolean {
            val identifier = name.removePrefix("$")
            val first = identifier.firstOrNull() ?: return false
            return (first == '_' || first.isAsciiLetter()) &&
                identifier.all { it == '_' || it.isAsciiLetter() || it in '0'..'9' }
        }

        private fun Char.isAsciiLetter(): Boolean = this in 'a'..'z' || this in 'A'..'Z'

        /**
         * The name a statement assigns and the value it assigns, or null when it assigns nothing a
         * name can be read from.
         */
        private fun assignment(statement: String): Pair<String, String>? {
            var index = 0
            while (index < statement.length) {
                if (statement[index] != '=') {
                    index += 1
                    continue
                }
                if (index > 0 && statement[index - 1] in "!<>=") {
                    index += 1
                    continue
                }
                if (statement.getOrNull(index + 1) == '=') {
                    index += 2
                    continue
                }
                val name = statement.substring(0, index).trim(' ', '\t')
                if (!isName(name)) return null
                return name to statement.substring(index + 1).trim()
            }
            return null
        }
    }
}
