package io.getstream.agents.android.core

import java.util.UUID
import kotlin.time.Clock
import kotlin.time.Duration
import kotlin.time.Instant
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull

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
    /**
     * What the agent did on its way to the turn's text, in order: each round of the model's
     * thinking on a written reply, and each tool it called. Empty on a participant's turn.
     */
    val steps: List<TurnStep> = emptyList(),
    /**
     * Whether the agent is done with the turn: it answered with no tools or work left to come, or
     * it was interrupted. A turn whose tools are still running is not. Never set on a
     * participant's turn.
     */
    val isDone: Boolean = false,
) {
    public sealed interface Speaker {
        /** A person, on the call or typing. Null when the router did not say who. */
        public data class Person(val participant: Participant?) : Speaker

        /** The agent. */
        public data object Agent : Speaker
    }

    public val isAgent: Boolean get() = speaker == Speaker.Agent

    /**
     * This turn with [delta] added to the round of thinking under way, or to a new one when the
     * model did something else since: every round is numbered from `r1`, as a kept reply's steps
     * are.
     */
    internal fun think(delta: String, began: Instant): Turn {
        val last = steps.lastOrNull()
        if (last is TurnStep.Reasoning && last.isStreaming) {
            return copy(steps = steps.dropLast(1) + last.copy(text = last.text + delta))
        }
        val round = TurnStep.Reasoning(
            id = "r${steps.count { it is TurnStep.Reasoning } + 1}",
            isStreaming = true,
            text = delta,
            at = began,
        )
        return copy(steps = steps + round)
    }

    /** This turn with its round of thinking under way completed, or itself when none is. */
    internal fun settled(at: Instant): Turn {
        val last = steps.lastOrNull()
        if (last !is TurnStep.Reasoning || !last.isStreaming) return this
        return copy(steps = steps.dropLast(1) + last.copy(isStreaming = false, duration = at - last.at))
    }

    /**
     * This turn with [call] begun, or itself when it already shows that call. A round of
     * thinking's id is ours, `r1` and on, so a call the provider gave the same id is another step.
     */
    internal fun called(call: TurnStep.ToolCall): Turn =
        if (steps.any { it is TurnStep.ToolCall && it.id == call.id }) this else copy(steps = steps + call)

    /** This turn with its running call [id] changed as [change] says, or itself when it has none. */
    internal fun changedCall(id: String, change: (TurnStep.ToolCall) -> TurnStep.ToolCall): Turn {
        val index = steps.indexOfFirst {
            it is TurnStep.ToolCall && it.id == id && it.status == TurnStep.ToolCall.Status.Running
        }
        if (index < 0) return this
        val updated = steps.toMutableList()
        updated[index] = change(updated[index] as TurnStep.ToolCall)
        return copy(steps = updated)
    }
}

/** One thing the agent did in a turn: a round of the model's thinking, or a call of a tool. */
public sealed interface TurnStep {
    /** `r1`, `r2` and on for a round of thinking; the call's id for a tool call. */
    public val id: String

    /** When it began. */
    public val at: Instant

    /** A round of the model's thinking on a written reply, which is read, never spoken. */
    public data class Reasoning(
        /** `r1`, `r2` and on, in the order the turn thought. */
        override val id: String,
        /** Whether the model is still thinking: until it writes, calls a tool or finishes. */
        val isStreaming: Boolean,
        /** The thinking so far. */
        val text: String,
        /**
         * When the model began on it: when it was asked, or took the reply up again after writing,
         * as a kept reply's step is timed. Not when its first words arrived, which a model that
         * sends its thinking in one burst sends only at the end.
         */
        override val at: Instant,
        /** How long the model thought, once it is done. */
        val duration: Duration? = null,
    ) : TurnStep

    /** A call of a tool: what was called and how it went, not what it was given or gave. */
    public data class ToolCall(
        /** The call's id, which [AgentSession.decide] answers it by while it waits in `approvals`. */
        override val id: String,
        /** The tool. */
        val name: String,
        val status: Status,
        /**
         * What whoever runs it last said about how it is going (`tool_updated`), or the empty
         * string.
         */
        val summary: String,
        override val at: Instant,
        /** How long it ran, once it is no longer running. */
        val duration: Duration? = null,
    ) : TurnStep {
        public enum class Status(public val wire: String) {
            Running("running"),
            Completed("completed"),
            Failed("failed"),
        }
    }
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
    /**
     * When the model last took each agent turn in flight up again, by turn id: when it was asked,
     * or went on after writing. A round of thinking is timed from it. Counted in equality, as
     * [uiWritten] is, so that a frame that moves only this is still kept.
     */
    private val resumedAt: Map<String, Instant>,
) {
    public constructor(
        turns: List<Turn> = emptyList(),
        state: State = State.Idle,
        failure: String? = null,
    ) : this(turns, state, failure, emptyMap(), emptyMap())

    /** This conversation with the given parts changed, and the UI being written kept. */
    public fun copy(
        turns: List<Turn> = this.turns,
        state: State = this.state,
        failure: String? = this.failure,
    ): Conversation = Conversation(turns, state, failure, uiWritten, resumedAt)

    public operator fun component1(): List<Turn> = turns

    public operator fun component2(): State = state

    public operator fun component3(): String? = failure

    override fun equals(other: Any?): Boolean =
        this === other ||
            other is Conversation &&
            turns == other.turns &&
            state == other.state &&
            failure == other.failure &&
            uiWritten == other.uiWritten &&
            resumedAt == other.resumedAt

    override fun hashCode(): Int = listOf(turns, state, failure, uiWritten, resumedAt).hashCode()

    override fun toString(): String = "Conversation(turns=$turns, state=$state, failure=$failure)"

    private fun writing(uiWritten: Map<String, UiWritten>): Conversation =
        Conversation(turns, state, failure, uiWritten, resumedAt)

    /** This conversation with agent turn [turnId] taken up again [at], or done with for null. */
    private fun resumed(turnId: String, at: Instant?): Conversation =
        Conversation(turns, state, failure, uiWritten, if (at == null) resumedAt - turnId else resumedAt + (turnId to at))

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
                    .resumed(event.turnId, at)

            // Writing ends the round of thinking under way, and the next one is timed from here.
            AgentEvent.Kind.ResponseDelta ->
                (if (event.text.isEmpty()) this else amend(event.turnId) { it.settled(at) }.resumed(event.turnId, at))
                    .write(event.turnId, event.text, at) { it + event.text }
                    .copy(state = State.Responding)

            // A round is timed from when the model was asked or went on, not from its first words.
            AgentEvent.Kind.ReasoningDelta ->
                if (event.text.isEmpty()) {
                    this
                } else {
                    val began = resumedAt[event.turnId] ?: at
                    amend(event.turnId, startingAt = at) { it.think(event.text, began) }
                }

            AgentEvent.Kind.ToolStarted -> {
                val call = TurnStep.ToolCall(
                    id = event.string("tool_call_id"),
                    name = event.string("tool"),
                    status = TurnStep.ToolCall.Status.Running,
                    summary = "",
                    at = at,
                )
                amend(event.turnId, startingAt = at) { it.settled(at).called(call) }
            }

            // Words to say as written are a line of the reply, not how the call is going.
            AgentEvent.Kind.ToolUpdated ->
                if (event["verbatim"].isTrue || event.text.isEmpty()) {
                    this
                } else {
                    amend(event.turnId) { turn -> turn.changedCall(event.string("id")) { it.copy(summary = event.text) } }
                }

            AgentEvent.Kind.ToolRan -> {
                val status = if (event.errorText.isEmpty()) TurnStep.ToolCall.Status.Completed else TurnStep.ToolCall.Status.Failed
                amend(event.turnId) { turn ->
                    turn.changedCall(event.string("tool_call_id")) { it.copy(status = status, duration = at - it.at) }
                }
            }

            AgentEvent.Kind.UiDelta -> {
                val delta = event.string("delta")
                val format = event.string("format")
                // Writing UI ends the round of thinking under way, as writing text does.
                val went = if (delta.isEmpty()) this else amend(event.turnId) { it.settled(at) }.resumed(event.turnId, at)
                if (delta.isEmpty() || format.isEmpty()) {
                    went
                } else {
                    val text = uiWritten[event.turnId]?.text.orEmpty() + delta
                    val written = UiWritten(event, format, text)
                    val kept = went.writing(uiWritten + (event.turnId to written))
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
            // A reply whose tools or work are still to come is carried on by a turn of its own.
            AgentEvent.Kind.Responded -> {
                val done = !event["pending_work"].isTrue
                settleUi(event.turnId, at)
                    .let { if (event.text.isEmpty()) it else it.write(event.turnId, event.text, at) { event.text } }
                    .resumed(event.turnId, null)
                    .amend(event.turnId) { turn -> turn.settled(at).let { if (done) it.copy(isDone = true) else it } }
                    .copy(state = State.Idle)
            }

            AgentEvent.Kind.Delegated -> copy(state = State.Working(working + event.string("skill")))

            AgentEvent.Kind.TaskSettled, AgentEvent.Kind.TaskCancelled -> {
                val left = working - event.string("skill")
                copy(state = if (left.isEmpty()) State.Responding else State.Working(left))
            }

            AgentEvent.Kind.Interrupted -> ended(event.turnId, at).copy(state = State.Idle)

            // The model writing the reply failing is the end of that reply, which may never be
            // reported as responded, so it is done as an interrupted one is. Its error may not say
            // which turn it ended, which is then the agent's latest one still going. Anything else
            // that fails, a tool, the voice or the transcriber, leaves the reply going, and the
            // agent doing what it was doing.
            AgentEvent.Kind.Error -> {
                val ends = endsTurn(event)
                (if (event.string("code") == PROMPT_INJECTION) refused() else this)
                    .let { conversation ->
                        val turnId = event.turnId.ifEmpty {
                            conversation.turns.lastOrNull { it.isAgent && !it.isDone }?.id.orEmpty()
                        }
                        if (ends && turnId.isNotEmpty()) conversation.ended(turnId, at) else conversation
                    }
                    .copy(failure = event.errorText, state = if (ends) State.Idle else state)
            }

            AgentEvent.Kind.Left -> copy(state = State.Ended)

            else -> this
        }

    /**
     * Agent turn [turnId] with nothing more of it to come: its round of thinking and its UI
     * settled, and the turn done.
     */
    private fun ended(turnId: String, at: Instant): Conversation =
        settleUi(turnId, at)
            .resumed(turnId, null)
            .amend(turnId) { it.settled(at).copy(isDone = true) }

    private val working: List<String>
        get() = (state as? State.Working)?.skills.orEmpty()

    /** Whether an error ends the agent's turn in flight: it is the reply model's, and there is one. */
    private fun endsTurn(event: AgentEvent): Boolean =
        (state == State.Responding || state is State.Working) && event.string("context") in replyModels

    private companion object {
        /** The `context` of an error from the model writing the reply, as text or as speech. */
        val replyModels = setOf("llm", "sts")

        /** The `code` of an error for a reply the prompt injection screen refused. */
        const val PROMPT_INJECTION = "prompt_injection"

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
     * The prompt injection screen refused the reply being written. What it streamed before the
     * verdict is not the answer, so its text and UI go, as they do from the kept conversation;
     * its thinking and tool calls stay.
     */
    private fun refused(): Conversation {
        val index = turns.indexOfLast { it.isAgent }
        val turn = turns.getOrNull(index)?.takeUnless { it.isDone } ?: return this
        val cleared = writing(uiWritten - turn.id)
        if (turn.text.isEmpty() && turn.ui == null) return cleared
        val updated = turns.toMutableList()
        updated[index] = turn.copy(text = "", ui = null)
        return cleared.copy(turns = updated)
    }

    /**
     * Applies [change] to agent turn [turnId]. A turn this never saw begin is started [startingAt]
     * when the event begins something, and left alone when it is null.
     */
    private fun amend(turnId: String, startingAt: Instant? = null, change: (Turn) -> Turn): Conversation {
        val index = turns.indexOfLast { it.id == turnId && it.isAgent }
        if (index < 0) {
            if (startingAt == null) return this
            return copy(turns = turns + change(Turn(turnId, Turn.Speaker.Agent, "", startingAt)))
        }
        val turn = turns[index]
        val changed = change(turn)
        if (changed == turn) return this
        val updated = turns.toMutableList()
        updated[index] = changed
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
     * one counts. A patch is merged onto the UI of the latest earlier turn, earlier than this
     * one even when its UI comes late, when that UI is OpenUI; when it is not, or with none in
     * this transcript, such as after a reload, it sets no UI, and Stream Chat's copy, which the router merged, is the one
     * to show.
     */
    private fun writeUi(turnId: String, written: UiWritten, keep: TurnUI.Keep, at: Instant): Conversation {
        val openUi = written.format == TurnUI.OPEN_UI
        var content = if (openUi) TurnUI.kept(written.text, keep) else written.text
        if (openUi && keep != TurnUI.Keep.All && content.isEmpty()) return this
        val index = turns.indexOfLast { it.id == turnId && it.isAgent }
        if (written.mode == "patch" && openUi) {
            // A turn we never saw begin starts after every one there is.
            val earlier = if (index < 0) turns else turns.subList(0, index)
            // The card before is the router's base whatever its format; only OpenUI takes a patch.
            val base = earlier.lastOrNull { it.isAgent && it.id != turnId && it.ui?.content?.isNotEmpty() == true }?.ui
            if (base?.format != TurnUI.OPEN_UI) return this
            content = TurnUI.merge(base.content, content)
        }
        val ui = TurnUI(written.format, written.version, keep == TurnUI.Keep.Finished, content)
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

/** Whether a field is JSON `true`, which is all a flag the router may leave out can be. */
private val JsonElement.isTrue: Boolean
    get() = (this as? JsonPrimitive)?.takeUnless { it.isString }?.booleanOrNull == true
