import { nested, text, type Frame } from "./socket.js";
import type { Participant, ToolApprovalRequest } from "./session.js";

/** Interactive UI streamed alongside an agent turn. */
export interface TurnUI {
  readonly format: string;
  readonly version?: string;
  readonly status: "streaming" | "complete";
  /**
   * The UI to show. While OpenUI streams, that is its finished statements only: the one no
   * newline has ended yet is held back, so a card grows a statement at a time rather than
   * reflowing with every delta (a gauge reading 8 before 86). Empty until one is finished.
   */
  readonly content: string;
}

/** One line of a conversation. */
export interface Turn {
  /** The router's turn id for an agent turn, and one of our own for a participant's. */
  readonly id: string;
  /** Who said it: a person, on the call or typing, or the agent. */
  readonly speaker: "participant" | "agent";
  /** The person, when the router said who. Never set on the agent's turns. */
  readonly participant?: Participant;
  /** What was said. The agent's turn in flight grows as its deltas arrive. */
  readonly text: string;
  /** Interactive UI generated for an agent turn, streaming in and then complete. */
  readonly ui?: TurnUI;
  /** When the turn began, in milliseconds since the epoch. */
  readonly at: number;
}

/** The conversation as the socket has left it. The same object until something in it changes. */
export interface ConversationRecord {
  /** The conversation so far, oldest first. */
  readonly turns: readonly Turn[];
  /**
   * What the agent last reported going wrong, such as speech recognition failing on a call,
   * or the empty string. Errors arrive as frames rather than being thrown, because nobody is
   * awaiting the socket. It lasts until the agent shows it works again: it next hears, starts
   * answering or speaks.
   */
  readonly failure: string;
  /**
   * Which part of the agent `failure` came from, as the router names it: `stt`, `llm`, `sts`,
   * `tts`, or the empty string when it named none. A call shows "can't hear you" for one and
   * "can't speak" for another.
   */
  readonly failureContext: string;
  /** When `failure` was reported, in milliseconds since the epoch, or zero while there is none. */
  readonly failureAt: number;
  /** Calls of functions that ask first, waiting for the person to answer, oldest first. */
  readonly approvals: readonly ToolApprovalRequest[];
  /** Whether the socket is still carrying the conversation. */
  readonly connected: boolean;
  /** Why the socket stopped, or undefined. A conversation that ended normally has none. */
  readonly socketError?: Error;
}

/**
 * The conversation, folded from the session's socket: `session.conversation`.
 *
 * Its turns are a transcript of what this device heard: what it typed, what was transcribed
 * on the call, and the agent's replies as they were written. Fed beside `events()`, like
 * `session.state`, so it takes nothing from the one reader that stream allows. `get` and
 * `subscribe` are bound, so a page can hand them over as they are:
 *
 * ```ts
 * const { turns, approvals, failure } = useSyncExternalStore(
 *   session.conversation.subscribe,
 *   session.conversation.get,
 * );
 * ```
 */
export class Conversation {
  private snapshot: ConversationRecord;
  private readonly listeners = new Set<() => void>();
  /** What each turn's UI block has written so far, unmerged, until the turn's UI is settled. */
  private readonly uiRawByTurn = new Map<string, PendingUI>();

  constructor(connected: boolean) {
    this.snapshot = Object.freeze({
      turns: Object.freeze([]),
      failure: "",
      failureContext: "",
      failureAt: 0,
      approvals: Object.freeze([]),
      connected,
    });
  }

  /** Where the conversation is now. */
  readonly get = (): ConversationRecord => this.snapshot;

  /** Calls `listener` whenever the snapshot changes. It returns what stops it. */
  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  /**
   * @internal What the caller typed, shown before the router has confirmed hearing it. Returns
   * what takes it back out, for a question that could not be asked.
   */
  said(said: string, now = Date.now()): () => void {
    const turn = participantTurn(said, undefined, now);
    this.turns([...this.snapshot.turns, turn]);
    return () => {
      const turns = this.snapshot.turns;
      if (turns.includes(turn)) {
        this.turns(turns.filter((each) => each !== turn));
      }
    };
  }

  /**
   * @internal Folds one frame off the session's socket into the transcript. A frame with no
   * bearing on it, and one this SDK has never heard of, both leave it alone.
   */
  observe(frame: Frame, now = Date.now()): void {
    const turns = this.snapshot.turns;
    if (
      this.snapshot.failure &&
      (frame.type === "heard" || frame.type === "responding" || frame.type === "ui_delta" || frame.type === "spoke")
    ) {
      // The agent works again.
      this.change({ failure: "", failureContext: "", failureAt: 0 });
    }
    switch (frame.type) {
      case "heard": {
        // A text session echoes nothing back, so what the caller typed is already here. A
        // call transcribes what was spoken, which is the first anyone hears of it.
        const heard = text(frame, "text");
        const last = turns[turns.length - 1];
        if (last?.speaker === "participant" && last.text === heard) {
          return;
        }
        this.turns([...turns, participantTurn(heard, participantOf(frame), now)]);
        return;
      }
      case "responding": {
        const turnId = text(frame, "turn_id");
        this.uiRawByTurn.delete(turnId);
        this.turns([...turns, Object.freeze({ id: turnId, speaker: "agent", text: "", at: now })]);
        return;
      }
      case "response_delta": {
        const delta = text(frame, "text");
        this.write(text(frame, "turn_id"), delta, now, (was) => was + delta);
        return;
      }
      case "ui_delta": {
        const delta = text(frame, "delta");
        if (!delta) {
          return;
        }
        const turnId = text(frame, "turn_id");
        const format = text(frame, "format");
        if (!format) {
          return;
        }
        const pending: PendingUI = {
          format,
          version: text(frame, "version"),
          mode: text(frame, "mode") || "replace",
          raw: (this.uiRawByTurn.get(turnId)?.raw ?? "") + delta,
        };
        this.uiRawByTurn.set(turnId, pending);
        if (format === "openui" && !delta.includes("\n")) {
          // Only the line still being written grew, so no statement finished: nothing shown changes.
          return;
        }
        this.showUI(turnId, pending, "finished", now);
        return;
      }
      case "ui_completed": {
        const turnId = text(frame, "turn_id");
        const format = text(frame, "format");
        if (!format) {
          return;
        }
        const pending: PendingUI = {
          format,
          version: text(frame, "version"),
          mode: text(frame, "mode") || "replace",
          raw: text(frame, "content") || (this.uiRawByTurn.get(turnId)?.raw ?? ""),
        };
        this.uiRawByTurn.delete(turnId);
        this.showUI(turnId, pending, "all", now);
        return;
      }
      case "responded": {
        // The final text is authoritative: the deltas are what was being written, this is
        // what was said. An empty one adds nothing, which is what a spoken-only turn is.
        const turnId = text(frame, "turn_id");
        this.settleUI(turnId, now);
        const said = text(frame, "text");
        if (said) {
          this.write(turnId, said, now, () => said);
        }
        return;
      }
      case "interrupted":
        this.settleUI(text(frame, "turn_id"), now);
        return;
      case "error":
        this.change({ failure: text(frame, "error"), failureContext: text(frame, "context"), failureAt: now });
        return;
      default:
        return;
    }
  }

  /** @internal The calls waiting for the person changed. */
  approvalsChanged(approvals: readonly ToolApprovalRequest[]): void {
    this.change({ approvals: Object.freeze([...approvals]) });
  }

  /** @internal The socket stopped, with the error that stopped it if it did not end normally. */
  disconnected(cause?: Error): void {
    this.uiRawByTurn.clear();
    this.change({
      connected: false,
      approvals: Object.freeze([]),
      ...(cause ? { socketError: cause } : {}),
    });
  }

  /**
   * Applies a change to the agent turn `turnId`, starting one if the router sent a delta for
   * a turn this never saw begin.
   */
  private write(turnId: string, given: string, now: number, change: (was: string) => string): void {
    const turns = this.snapshot.turns;
    const index = this.agentTurnIndex(turnId);
    if (index === -1) {
      this.turns([...turns, Object.freeze({ id: turnId, speaker: "agent", text: given, at: now })]);
      return;
    }
    const turn = turns[index]!;
    const nextText = change(turn.text);
    if (nextText === turn.text) {
      // An empty delta, or a final text the deltas already spelled: nothing for a page to redraw.
      return;
    }
    const next = turns.slice();
    next[index] = Object.freeze({ ...turn, text: nextText });
    this.turns(next);
  }

  /**
   * The turn ended with no `ui_completed`, as an interrupted reply does: the router dropped its
   * UI block with the turn. What was written shows, but for a statement it ended in the middle
   * of, and the UI is complete, since nothing more of it will come.
   */
  private settleUI(turnId: string, now: number): void {
    const pending = this.uiRawByTurn.get(turnId);
    this.uiRawByTurn.delete(turnId);
    if (pending) {
      this.showUI(turnId, pending, "closed", now);
    }
  }

  /**
   * Shows `pending` as the UI of the turn `turnId`: streaming while only its `finished`
   * statements count, complete otherwise. A UI with nothing to show yet sets none.
   */
  private showUI(turnId: string, pending: PendingUI, keep: OpenUIKeep, now: number): void {
    const content = this.resolveUIContent(turnId, pending, keep);
    if (content === undefined) {
      return;
    }
    this.writeUI(
      turnId,
      Object.freeze({
        format: pending.format,
        ...(pending.version ? { version: pending.version } : {}),
        status: keep === "finished" ? "streaming" : "complete",
        content,
      }),
      now,
    );
  }

  private writeUI(turnId: string, ui: TurnUI, now: number): void {
    const turns = this.snapshot.turns;
    const index = this.agentTurnIndex(turnId);
    if (index === -1) {
      this.turns([...turns, Object.freeze({ id: turnId, speaker: "agent", text: "", ui, at: now })]);
      return;
    }
    const turn = turns[index]!;
    if (
      turn.ui?.format === ui.format &&
      turn.ui.version === ui.version &&
      turn.ui.status === ui.status &&
      turn.ui.content === ui.content
    ) {
      return;
    }
    const next = turns.slice();
    next[index] = Object.freeze({ ...turn, ui });
    this.turns(next);
  }

  /**
   * A turn's UI as it shows: what was written, or a patch merged onto the UI before it. Of OpenUI,
   * only the statements `keep` says count. Undefined when there is nothing to show yet, and for a
   * patch with no UI before it here, such as after a reload: Stream Chat's copy, which the router
   * merged, is the one that shows.
   */
  private resolveUIContent(turnId: string, { format, mode, raw }: PendingUI, keep: OpenUIKeep): string | undefined {
    if (format !== "openui") {
      return raw;
    }
    const written = keep === "all" ? raw : splitOpenUIStatements(raw, keep).join("\n");
    if (keep !== "all" && !written) {
      return undefined;
    }
    if (mode !== "patch") {
      return written;
    }
    const turns = this.snapshot.turns;
    for (let index = turns.length - 1; index >= 0; index--) {
      const earlier = turns[index];
      if (earlier?.speaker === "agent" && earlier.id !== turnId && earlier.ui?.format === "openui" && earlier.ui.content) {
        return mergeOpenUIStatements(earlier.ui.content, written);
      }
    }
    return undefined;
  }

  /**
   * Where the agent turn `turnId` is, or -1. The turn being written is almost always the last
   * one, and a reply streams a delta per few words, so that one is looked at first rather than
   * the transcript searched each time.
   */
  private agentTurnIndex(turnId: string): number {
    const turns = this.snapshot.turns;
    const last = turns.length - 1;
    const matches = (turn: Turn | undefined) => turn?.speaker === "agent" && turn.id === turnId;
    if (matches(turns[last])) {
      return last;
    }
    for (let index = last - 1; index >= 0; index--) {
      if (matches(turns[index])) {
        return index;
      }
    }
    return -1;
  }

  private turns(turns: Turn[]): void {
    this.change({ turns: Object.freeze(turns) });
  }

  private change(patch: Partial<ConversationRecord>): void {
    this.snapshot = Object.freeze({ ...this.snapshot, ...patch });
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

const OPENUI_IDENT = /^\$?[A-Za-z_][A-Za-z0-9_]*$/;

/**
 * Merges an OpenUI statement patch onto an earlier OpenUI program: matching identifiers
 * update in place so `root` stays at the top, component identifiers assigned `null` are
 * removed, and new identifiers append at the end.
 */
export function mergeOpenUIStatements(base: string, patch: string): string {
  if (!base.trim()) {
    return patch;
  }
  if (!patch.trim()) {
    return base;
  }
  const order: string[] = [];
  const byIdent = new Map<string, string>();

  for (const raw of splitOpenUIStatements(base)) {
    const parsed = parseOpenUIAssignment(raw);
    if (!parsed) {
      continue;
    }
    if (!byIdent.has(parsed.ident)) {
      order.push(parsed.ident);
    }
    byIdent.set(parsed.ident, raw);
  }

  for (const raw of splitOpenUIStatements(patch)) {
    const parsed = parseOpenUIAssignment(raw);
    if (!parsed) {
      continue;
    }
    if (parsed.rhs === "null" && !parsed.ident.startsWith("$")) {
      byIdent.delete(parsed.ident);
      const idx = order.indexOf(parsed.ident);
      if (idx >= 0) {
        order.splice(idx, 1);
      }
      continue;
    }
    if (!byIdent.has(parsed.ident)) {
      order.push(parsed.ident);
    }
    byIdent.set(parsed.ident, `${parsed.ident} = ${parsed.rhs}`);
  }

  return order.flatMap((ident) => {
    const stmt = byIdent.get(ident);
    return stmt !== undefined ? [stmt] : [];
  }).join("\n");
}

/** What a turn's UI was written to. */
interface PendingUI {
  readonly format: string;
  readonly version: string;
  readonly mode: string;
  readonly raw: string;
}

/**
 * Which statements of an OpenUI program count: `all` of them; those `closed`, leaving out one a
 * bracket or a string left open; or those `finished`, also leaving out the line no newline has
 * ended, which is still being written.
 */
type OpenUIKeep = "all" | "closed" | "finished";

/**
 * A program's statements: each line, or the lines a bracket or a string left open joined to the
 * one that closes it. Blank and comment lines are left out, and so is what `keep` leaves out.
 */
function splitOpenUIStatements(program: string, keep: OpenUIKeep = "all"): string[] {
  const stmts: string[] = [];
  let current = "";
  let quote = "";
  let escaped = false;
  let depth = 0;

  const lines = program.split("\n");
  if (keep === "finished") {
    // What follows the last newline is still being written.
    lines.pop();
  }
  for (const line of lines) {
    if (!quote && depth === 0) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("//") || trimmed.startsWith("#")) {
        continue;
      }
    }
    current = current ? `${current}\n${line}` : line;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i]!;
      if (escaped) {
        escaped = false;
        continue;
      }
      if (quote) {
        if (ch === "\\") {
          escaped = true;
        } else if (ch === quote) {
          quote = "";
        }
        continue;
      }
      if (ch === "/" && line[i + 1] === "/") {
        break;
      }
      if (ch === '"' || ch === "'") {
        quote = ch;
      } else if (ch === "(" || ch === "[" || ch === "{") {
        depth++;
      } else if ((ch === ")" || ch === "]" || ch === "}") && depth > 0) {
        depth--;
      }
    }
    if (!quote && depth === 0) {
      const finished = current.trim();
      if (finished) {
        stmts.push(finished);
      }
      current = "";
    }
  }
  const tail = current.trim();
  if (tail && keep === "all") {
    stmts.push(tail);
  }
  return stmts;
}

function parseOpenUIAssignment(stmt: string): { ident: string; rhs: string } | undefined {
  for (let i = 0; i < stmt.length; i++) {
    if (stmt[i] !== "=") {
      continue;
    }
    const prev = i > 0 ? stmt[i - 1] : "";
    if (prev === "!" || prev === "<" || prev === ">" || prev === "=") {
      continue;
    }
    if (stmt[i + 1] === "=") {
      i++;
      continue;
    }
    const ident = stmt.slice(0, i).trim();
    if (!OPENUI_IDENT.test(ident)) {
      return undefined;
    }
    return { ident, rhs: stmt.slice(i + 1).trim() };
  }
  return undefined;
}

function participantTurn(said: string, participant: Participant | undefined, now: number): Turn {
  return Object.freeze({
    id: crypto.randomUUID(),
    speaker: "participant",
    ...(participant ? { participant } : {}),
    text: said,
    at: now,
  });
}

function participantOf(frame: Frame): Participant | undefined {
  const who = nested(frame, "participant");
  return who
    ? Object.freeze({ id: text(who, "id"), userId: text(who, "user_id"), name: text(who, "name") })
    : undefined;
}
