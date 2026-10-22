import { flag, number, text, type Frame } from "./socket.js";

/**
 * How fast a voice is taken to read, in characters per millisecond, until a `spoke` has
 * measured it: about 15 characters a second.
 */
export const DEFAULT_CHARS_PER_MS = 0.015;

/**
 * How long a reply known to be coming is held at nothing heard while its audio has not
 * started, from when it was first known to be coming. A router that sends no `speaking`, or
 * a reply that is never said, is then let go of: `heard` is undefined, and a page shows the
 * text as it is.
 */
export const SPEECH_GRACE_MS = 5_000;

/** How many turns are remembered. A page reveals the last few replies, not the whole call. */
const KEPT_TURNS = 20;

/** What `heard` says of a reply that is coming and not yet heard. */
const NOTHING_YET: HeardSpeech = Object.freeze({ chars: 0, done: false });

/**
 * How much of one turn has been heard.
 *
 * Characters are Unicode code points, as the session socket counts them, not UTF-16 units:
 * reveal `Array.from(text).slice(0, chars).join("")`, not `text.slice(0, chars)`.
 */
export interface HeardSpeech {
  /**
   * Code points of the turn heard so far: every utterance that has played out, and the
   * share of the one playing. Exact once the playing one's `spoke` has said how long it is,
   * and estimated from the voice's measured rate until then. A `spoke` that finds the
   * estimate ahead slows the rest down rather than jumping back, so it only drops when a
   * streamed reply's estimate overshot all the utterance turned out to hold, or when a cut
   * says less was heard than estimated.
   */
  readonly chars: number;
  /**
   * The turn's last known utterance has played out, or the turn was cut off. A voice that
   * takes a sentence at a time can still start another after a pause, so this means nothing
   * more is playing, not that nothing more will: a page shows a reply in full once this is
   * true and its message is no longer being written.
   */
  readonly done: boolean;
}

/** Where the agent's voice last was, for a page to know when to start animating. */
export interface SpeechSnapshot {
  /** The turn heard last. */
  readonly turnId: string;
  /** Its utterance heard last: the turn's id, or `<turn>#<n>` for a voice that takes sentences. */
  readonly synthesisId: string;
  /** When that utterance started, or is expected to start, playing, on the store's clock. */
  readonly startedAt: number;
  /** The turn was cut off, and stays at what was heard. */
  readonly interrupted: boolean;
  /** The voice's speaking rate, measured or `DEFAULT_CHARS_PER_MS` until it has been. */
  readonly charsPerMs: number;
}

/** How a `Speech` store is set up. */
export interface SpeechOptions {
  /** What times are read from: `performance.now()`, unless a test says otherwise. */
  clock?: () => number;
  /**
   * Whether the agent is on the call, so a reply written now will be said. The session
   * answers with `session.voice.started`. Without it, a reply is only held for its voice once
   * this store has heard the voice say something, or when it is a line said as written.
   */
  voice?: () => boolean;
}

/** One utterance of a turn: from its `speaking`, and its `spoke` once that arrives. */
interface Utterance {
  readonly synthesisId: string;
  readonly startedAt: number;
  /** A sentence of its own, whose `text` was all of it. A streamed one's is only its start. */
  readonly sentence: boolean;
  readonly textChars: number;
  /** From `spoke`: how long it is, and where the estimate had got to when that arrived. */
  measured?: {
    readonly chars: number;
    readonly durationMs: number;
    readonly at: number;
    readonly shown: number;
  };
}

interface Turn {
  readonly utterances: Utterance[];
  /** What was heard when it was cut off. */
  cut?: number | undefined;
  /** Until when it is held at nothing heard, while it has no utterance yet. */
  pendingUntil?: number | undefined;
  /** What lets go of it then, so listeners hear about it. */
  timer?: ReturnType<typeof setTimeout> | undefined;
  /**
   * It was let go of without being heard: its voice did not start in time, or it said
   * nothing. A page has shown it in full by then, so speech that comes after does not take
   * it back.
   */
  released?: boolean;
}

/** The clock a page animates by: requestAnimationFrame's timestamps are on it too. */
function monotonic(): number {
  return globalThis.performance?.now() ?? Date.now();
}

/**
 * How much of each of the agent's replies has been heard on the call, from the session
 * socket's `speaking`, `spoke` and `interrupted`.
 *
 * The Chat channel has a reply as fast as the model writes it, well ahead of the voice. Each
 * agent message there carries the `turn_id` these frames name, so a page can show as much of
 * it as has been said:
 *
 * ```ts
 * const heard = session.speech.heard(message, now); // or message.turn_id
 * const shown = heard ? Array.from(message.text).slice(0, heard.chars).join("") : message.text;
 * ```
 *
 * A reply's message usually reaches a page before its audio starts, and may reach it before
 * the socket says anything of the turn: Chat and the session socket are two connections. On
 * a call, a turn is held at `{ chars: 0, done: false }` until its `speaking` once it is known
 * to be coming: its text has begun on the socket (`said`, `response_delta`, or a `responded`
 * with words), or `heard` was asked about a message of the agent's that names the turn and
 * is still being written. It is let go of, and `heard` is undefined for it from then on, if
 * it turns out to say nothing (an empty `responded`, or a `turn` that published no audio), or
 * if no `speaking` has come `SPEECH_GRACE_MS` after it was first known to be coming. Speech
 * for a turn let go of does not take it back: the page has shown the text in full by then.
 *
 * The agent leaving the call (`left`), the voice being stopped, or the session ending
 * forgets that it was on the call, so a reply in writing afterwards is not held.
 *
 * The session feeds it from the frames it reads, beside `events()` rather than through it,
 * so reading events is unaffected. Times are on the store's clock, `performance.now()`, and
 * are when each frame arrived here rather than the router's `at`, which is another machine's
 * clock. Only the last 20 turns are kept.
 */
export class Speech {
  private readonly turns = new Map<string, Turn>();
  private readonly listeners = new Set<() => void>();
  private readonly clock: () => number;
  private readonly voice: () => boolean;
  private snapshot: SpeechSnapshot | undefined;
  private rate = DEFAULT_CHARS_PER_MS;
  private measured = false;
  /** Whether the voice has said anything on this call: a router that sends `speaking`. */
  private heardAny = false;
  /** Cleared when the agent leaves the call, and set again when it joins or is heard. */
  private onCall = true;
  /** A notification asked for from inside `heard`, waiting for it to return. */
  private later = false;
  /** The utterance heard last, which a later one plays after. */
  private last: { turn: Turn; utterance: Utterance } | undefined;

  constructor(options: SpeechOptions = {}) {
    this.clock = options.clock ?? monotonic;
    this.voice = options.voice ?? (() => false);
  }

  /**
   * Calls `listener` whenever what `heard` says of a turn changes other than with the clock:
   * an utterance starts, is measured or is cut off, a reply starts or stops being held for
   * its voice. It returns what stops it. Between those, `heard` moves with the clock alone,
   * so a page animating a reply reads it each frame rather than waiting to be told. `get`
   * describes the voice only, so it does not change with a reply being held.
   */
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** Where the voice last was, undefined until it has said anything. The same object until it changes. */
  get(): SpeechSnapshot | undefined {
    return this.snapshot;
  }

  /**
   * How much of a turn has been heard by `now`, or undefined for a turn the voice has not
   * reached, one let go of, a router that does not report it, or a message that names no
   * turn. That is what a page falls back on: a message with no speech to pace it by is shown
   * as it is.
   *
   * `turn` is the turn's id, or a Chat message, whose `turn_id` is read (top level in
   * stream-chat 9, under `custom` in 10), so `session.speech.heard(message, now)` needs no
   * custom message type declared. Given a message of the agent's still being written for a
   * turn the socket has said nothing of yet, on a call, it starts holding that turn (see
   * above) and tells listeners once it has returned. Cheap enough to call on every
   * animation frame.
   */
  heard(turn: unknown, now: number = this.clock()): HeardSpeech | undefined {
    const turnId = turnIdOf(turn);
    if (!turnId) {
      return undefined;
    }
    const held = this.turns.get(turnId) ?? this.awaited(turn, turnId, now);
    if (!held || held.released) {
      return undefined;
    }
    if (held.cut !== undefined) {
      return { chars: held.cut, done: true };
    }
    if (held.utterances.length === 0) {
      return held.pendingUntil !== undefined && now < held.pendingUntil ? NOTHING_YET : undefined;
    }
    return { chars: Math.floor(this.charsOf(held, now)), done: playedOut(held, now) };
  }

  /**
   * Takes one frame off the session socket, ignoring all but the few that say a reply is
   * coming, being heard, finished or cut off, and the agent joining or leaving the call. The
   * session calls this for every frame it reads; it is here for frames read some other way.
   */
  observe(frame: Frame): void {
    switch (frame.type) {
      case "said":
        // A line said as written is only ever said on a call: in writing it is `responded`.
        this.begun(text(frame, "turn_id"), true);
        return;
      case "response_delta":
        this.begun(text(frame, "turn_id"), false);
        return;
      case "responded":
        if (text(frame, "text")) {
          this.begun(text(frame, "turn_id"), false);
        } else {
          this.unsaid(text(frame, "turn_id"));
        }
        return;
      case "turn":
        // A turn is reported once all of it has been said, after its speaking, or with no
        // audio at all when there was nothing to say.
        if (number(frame, "audio_out_ms") <= 0 && !flag(frame, "interrupted")) {
          this.unsaid(text(frame, "turn_id"));
        }
        return;
      case "speaking":
        this.speaking(frame);
        return;
      case "spoke":
        this.spoke(frame);
        return;
      case "interrupted":
        this.interrupted(frame);
        return;
      case "joined":
        this.voiceChanged(true);
        return;
      case "left":
        this.voiceChanged(false);
        return;
    }
  }

  /**
   * Says the agent joined or left the call. Off, nothing is held for a voice any more: what
   * is waiting is let go of, and a reply in writing after is shown as it is. The session
   * calls this from `voice.start()` and `voice.stop()`, and the socket's `joined` and `left`
   * say the same.
   */
  voiceChanged(on: boolean): void {
    this.onCall = on;
    if (on) {
      return;
    }
    this.heardAny = false;
    let changed = false;
    for (const turn of this.turns.values()) {
      if (this.holding(turn)) {
        this.letGo(turn);
        changed = true;
      }
    }
    if (changed) {
      this.notify();
    }
  }

  /** Stops the timers holding replies for their voice. The session calls this when it ends. */
  close(): void {
    this.voiceChanged(false);
    for (const turn of this.turns.values()) {
      this.release(turn);
    }
  }

  /** Whether a reply written now is going to be said. */
  private spoken(): boolean {
    return this.onCall && (this.heardAny || this.voice());
  }

  /** Whether a turn is held at nothing heard: known to be coming, and not heard yet. */
  private holding(turn: Turn): boolean {
    return turn.utterances.length === 0 && turn.cut === undefined && !turn.released;
  }

  /** A turn's text began on the socket. */
  private begun(turnId: string, said: boolean): void {
    if (this.coming(turnId, said, this.clock())) {
      this.notify();
    }
  }

  /**
   * Holds a turn whose text has begun at nothing heard, if it is going to be said: on a
   * call, or for a line said as written. A turn already known is left as it is, so the deltas
   * that keep coming while it is said, or after it was let go of, change nothing.
   */
  private coming(turnId: string, said: boolean, now: number): Turn | undefined {
    if (!turnId || this.turns.has(turnId) || !(said || this.spoken())) {
      return undefined;
    }
    const turn: Turn = { utterances: [], pendingUntil: now + SPEECH_GRACE_MS };
    turn.timer = setTimeout(() => {
      turn.timer = undefined;
      if (this.holding(turn)) {
        this.letGo(turn);
        this.notify();
      }
    }, SPEECH_GRACE_MS);
    // A page or a process should not stay up for this.
    (turn.timer as { unref?: () => void }).unref?.();
    this.add(turnId, turn);
    return turn;
  }

  /**
   * A message `heard` was asked about for a turn the socket has said nothing of: Chat can be
   * faster than the socket. One of the agent's still being written, on a call, is coming.
   * Listeners are told once `heard` has returned, since a listener is what calls it.
   */
  private awaited(turn: unknown, turnId: string, now: number): Turn | undefined {
    if (typeof turn !== "object" || turn === null || !this.spoken() || !agentWriting(turn)) {
      return undefined;
    }
    const held = this.coming(turnId, false, now);
    if (held && !this.later) {
      this.later = true;
      queueMicrotask(() => {
        this.later = false;
        this.notify();
      });
    }
    return held;
  }

  /** Lets go of a turn held for its voice that turned out to say nothing. */
  private unsaid(turnId: string): void {
    const turn = this.turns.get(turnId);
    if (!turn || !this.holding(turn)) {
      return;
    }
    this.letGo(turn);
    this.notify();
  }

  /** Lets go of a turn without it having been heard, for good. */
  private letGo(turn: Turn): void {
    this.release(turn);
    turn.released = true;
  }

  /** Stops holding a turn: it is heard, cut off, let go of, or forgotten. */
  private release(turn: Turn): void {
    turn.pendingUntil = undefined;
    if (turn.timer !== undefined) {
      clearTimeout(turn.timer);
      turn.timer = undefined;
    }
  }

  private add(turnId: string, turn: Turn): void {
    this.turns.set(turnId, turn);
    for (const [oldest, forgotten] of this.turns) {
      if (this.turns.size <= KEPT_TURNS) {
        break;
      }
      this.release(forgotten);
      this.turns.delete(oldest);
    }
  }

  private speaking(frame: Frame): void {
    const turnId = text(frame, "turn_id");
    const synthesisId = text(frame, "synthesis_id") || turnId;
    if (!turnId) {
      return;
    }
    const now = this.clock();
    let turn = this.turns.get(turnId);
    if (turn?.cut !== undefined || turn?.utterances.some((u) => u.synthesisId === synthesisId)) {
      return;
    }
    if (!turn) {
      turn = { utterances: [] };
      this.add(turnId, turn);
    } else if (this.holding(turn) && turn.pendingUntil !== undefined && now >= turn.pendingUntil) {
      // Its grace ran out before the timer said so: it was let go of all the same.
      this.letGo(turn);
    }
    this.release(turn);
    this.heardAny = true;
    this.onCall = true;

    // An utterance reported before the track reached it says how long the speech queued
    // ahead of it will take. One track plays one thing at a time, so it also starts no
    // sooner than the one before it ends, if that is known.
    let startedAt = now + Math.max(0, number(frame, "delay_ms"));
    const before = this.last;
    if (before?.turn.cut === undefined && before?.utterance.measured) {
      startedAt = Math.max(startedAt, before.utterance.startedAt + before.utterance.measured.durationMs);
    }
    const utterance: Utterance = {
      synthesisId,
      startedAt,
      sentence: synthesisId !== turnId,
      textChars: codePoints(text(frame, "text")),
    };
    // A turn let go of keeps its utterances, for later ones to queue behind, and stays let go.
    turn.utterances.push(utterance);
    this.last = { turn, utterance };
    this.changed({ turnId, synthesisId, startedAt, interrupted: false, charsPerMs: this.rate });
  }

  private spoke(frame: Frame): void {
    const turnId = text(frame, "turn_id");
    const synthesisId = text(frame, "synthesis_id") || turnId;
    const utterance = this.turns.get(turnId)?.utterances.find((u) => u.synthesisId === synthesisId);
    const chars = number(frame, "characters");
    const durationMs = number(frame, "audio_duration_ms");
    if (!utterance || utterance.measured || chars <= 0 || durationMs <= 0) {
      return;
    }

    // The estimate so far is where the rest starts from, so a reveal that ran ahead slows
    // down rather than jumping back, and both end with the audio.
    const at = Math.max(this.clock(), utterance.startedAt);
    const shown = Math.min(chars, this.estimate(utterance, at));
    utterance.measured = { chars, durationMs, at, shown };
    const rate = chars / durationMs;
    this.rate = this.measured ? (this.rate + rate) / 2 : rate;
    this.measured = true;
    if (this.snapshot) {
      this.changed({ ...this.snapshot, charsPerMs: this.rate });
    }
  }

  /**
   * Freezes a turn at what was heard of it. The router says how many milliseconds of the
   * utterance it cut had been heard, which is what the voice really got through; without
   * that, the estimate stands.
   */
  private interrupted(frame: Frame): void {
    const turnId = text(frame, "turn_id");
    const turn = this.turns.get(turnId);
    const now = this.clock();
    // A turn let go of stays let go of: the page has shown it in full.
    if (!turn || turn.cut !== undefined || turn.released) {
      return;
    }
    if (this.holding(turn) && (turn.pendingUntil === undefined || now >= turn.pendingUntil)) {
      this.letGo(turn);
      return;
    }
    const cutId = text(frame, "synthesis_id");
    const heardMs = frame["heard_ms"];
    const index = cutId ? turn.utterances.findIndex((u) => u.synthesisId === cutId) : -1;
    turn.cut = Math.floor(
      typeof heardMs === "number" && index >= 0
        ? this.charsBefore(turn, index) + this.heardOf(turn.utterances[index]!, heardMs)
        : this.charsOf(turn, now),
    );
    this.release(turn);
    if (this.snapshot?.turnId === turnId) {
      this.changed({ ...this.snapshot, interrupted: true });
    } else {
      this.notify();
    }
  }

  /** Characters of the utterances of a turn before the one at index, all heard. */
  private charsBefore(turn: Turn, index: number): number {
    let chars = 0;
    for (let at = 0; at < index; at++) {
      chars += this.whole(turn.utterances[at]!, turn.utterances[at + 1]!.startedAt);
    }
    return chars;
  }

  /** How many characters heardMs of an utterance is, at its own rate when it was measured. */
  private heardOf(utterance: Utterance, heardMs: number): number {
    const { measured } = utterance;
    const chars = measured
      ? measured.chars * Math.min(1, Math.max(0, heardMs) / measured.durationMs)
      : Math.max(0, heardMs) * this.rate;
    return utterance.sentence && utterance.textChars > 0 ? Math.min(chars, utterance.textChars) : chars;
  }

  /** An utterance that played out before the next started at endedAt. */
  private whole(utterance: Utterance, endedAt: number): number {
    return (
      utterance.measured?.chars ??
      (utterance.sentence && utterance.textChars > 0
        ? utterance.textChars
        : this.estimate(utterance, endedAt))
    );
  }

  /** Characters of a turn heard by `now`, utterance by utterance. */
  private charsOf(turn: Turn, now: number): number {
    let chars = 0;
    turn.utterances.forEach((utterance, index) => {
      if (now <= utterance.startedAt) {
        return;
      }
      const next = turn.utterances[index + 1];
      if (next && next.startedAt <= now) {
        // The next one started, so this one played out.
        chars += this.whole(utterance, next.startedAt);
        return;
      }
      chars += this.share(utterance, now);
    });
    return chars;
  }

  /** How much of the utterance playing has been heard by `now`. */
  private share(utterance: Utterance, now: number): number {
    const { measured } = utterance;
    if (!measured || now < measured.at) {
      return this.estimate(utterance, now);
    }
    const remaining = utterance.startedAt + measured.durationMs - measured.at;
    if (remaining <= 0) {
      return measured.chars;
    }
    const progress = Math.min(1, (now - measured.at) / remaining);
    return measured.shown + (measured.chars - measured.shown) * progress;
  }

  /** The share of an utterance not yet measured, at the voice's rate. */
  private estimate(utterance: Utterance, now: number): number {
    const estimated = Math.max(0, now - utterance.startedAt) * this.rate;
    // A sentence's text is all of it; a streamed reply's is only how it starts.
    return utterance.sentence && utterance.textChars > 0
      ? Math.min(estimated, utterance.textChars)
      : estimated;
  }

  private changed(next: SpeechSnapshot): void {
    this.snapshot = next;
    this.notify();
  }

  private notify(): void {
    for (const listener of [...this.listeners]) {
      try {
        listener();
      } catch (cause) {
        // The session's socket is read on this call, and one listener's mistake must not
        // stop it. Reported the way a DOM event listener's is, without the others missing out.
        queueMicrotask(() => {
          throw cause;
        });
      }
    }
  }
}

/** The turn a message names, or the turn id itself. */
function turnIdOf(turn: unknown): string {
  if (typeof turn === "string") {
    return turn;
  }
  if (typeof turn !== "object" || turn === null) {
    return "";
  }
  const message = turn as { turn_id?: unknown; custom?: { turn_id?: unknown } };
  const named = message.turn_id ?? message.custom?.turn_id;
  return typeof named === "string" ? named : "";
}

/**
 * Whether a Chat message is one of the agent's still being written: `generating`, and
 * `source: "agent"` or `ai_generated`, each top level in stream-chat 9 or under `custom` in 10.
 */
function agentWriting(message: object): boolean {
  const fields = message as Record<string, unknown> & { custom?: Record<string, unknown> };
  const field = (key: string): unknown => fields[key] ?? fields.custom?.[key];
  return field("generating") === true && (field("source") === "agent" || field("ai_generated") === true);
}

/** How many Unicode code points a text is, which is how the session socket counts characters. */
function codePoints(text: string): number {
  let count = 0;
  for (let index = 0; index < text.length; index++) {
    const unit = text.charCodeAt(index);
    // A high surrogate and the low one after it are one code point.
    if (unit >= 0xd800 && unit <= 0xdbff && index + 1 < text.length) {
      const next = text.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        index++;
      }
    }
    count++;
  }
  return count;
}

/** The turn's last utterance has been measured and its audio has run out. */
function playedOut(turn: Turn, now: number): boolean {
  const last = turn.utterances.at(-1);
  return last?.measured !== undefined && now >= last.startedAt + last.measured.durationMs;
}
