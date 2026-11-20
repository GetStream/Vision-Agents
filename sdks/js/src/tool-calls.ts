import type { Schemas } from "./client.js";
import { number, text, type Frame } from "./socket.js";

/** One tool call as the session last reported it, whichever process runs it. */
export type ToolCallState = Schemas["ToolCallState"];

/** `in_progress`, `waiting` for a person's approval, `success`, `failed` or `cancelled`. */
export type ToolCallStatus = Schemas["ToolCallStatus"];

const frameTypes: ReadonlySet<string> = new Set(["tool_call.started", "tool_call.progress", "tool_call.finished"]);

/**
 * The session's tool calls by `tool_call_id`, in the order they started, each as the session
 * last reported it: `session.toolCalls`. Finished calls stay, with how they ended.
 *
 * Fed beside `events()`, like `session.state`. `get` and `subscribe` are bound, and the map is
 * the same object until a call changes:
 *
 * ```ts
 * const calls = useSyncExternalStore(session.toolCalls.subscribe, session.toolCalls.get);
 * ```
 */
export class ToolCalls {
  private snapshot: ReadonlyMap<string, ToolCallState> = new Map();
  private readonly listeners = new Set<() => void>();

  /** Every call so far. */
  readonly get = (): ReadonlyMap<string, ToolCallState> => this.snapshot;

  /** Calls `listener` whenever a call changes. It returns what stops it. */
  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  /** @internal Folds one frame off the session's socket in. Frames about no call's state leave it alone. */
  observe(frame: Frame): void {
    const state = toolCallStateOf(frame);
    if (state === undefined) {
      return;
    }
    const next = new Map(this.snapshot);
    next.set(state.tool_call_id, state);
    this.snapshot = next;
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

/** The state a `tool_call.*` frame carries, or undefined for any other frame. */
export function toolCallStateOf(frame: Frame): ToolCallState | undefined {
  const type = text(frame, "type");
  const id = text(frame, "tool_call_id");
  if (!frameTypes.has(type) || id === "") {
    return undefined;
  }
  const optional = (key: "progress" | "approved_at" | "rejected_at" | "error") =>
    text(frame, key) === "" ? {} : { [key]: text(frame, key) };
  return Object.freeze({
    type: type as ToolCallState["type"],
    tool_call_id: id,
    tool: text(frame, "tool"),
    turn_id: text(frame, "turn_id"),
    status: text(frame, "status") as ToolCallStatus,
    started_at: text(frame, "started_at"),
    duration_ms: number(frame, "duration_ms"),
    ...optional("progress"),
    ...optional("approved_at"),
    ...optional("rejected_at"),
    ...optional("error"),
  });
}
