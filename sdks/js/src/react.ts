/**
 * React hooks over a session's stores, opted into by importing this entry:
 *
 * ```ts
 * import { useToolCall, useToolCalls } from "@stream-io/stream-agents/react";
 * ```
 *
 * Its own entry point so `react` is resolved only for an app that asked for it.
 */

import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";

import type { Session, SessionEvent } from "./session.js";
import type { ToolCallState } from "./tool-calls.js";

/** Every tool call of the session by `tool_call_id`, in the order they started, re-rendering as one changes. */
export function useToolCalls(session: Session): ReadonlyMap<string, ToolCallState> {
  return useSyncExternalStore(session.toolCalls.subscribe, session.toolCalls.get, session.toolCalls.get);
}

/** One tool call as it was last reported, or undefined before the session said anything of it. */
export function useToolCall(session: Session, id: string): ToolCallState | undefined {
  const read = useCallback(() => session.toolCalls.get().get(id), [session, id]);
  return useSyncExternalStore(session.toolCalls.subscribe, read, read);
}

/**
 * Calls `handler` with every event of `kind` while the component is mounted. The latest
 * `handler` is the one called, so it need not be memoised.
 */
export function useSessionEvent(session: Session, kind: string, handler: (event: SessionEvent) => void): void {
  const latest = useRef(handler);
  latest.current = handler;
  useEffect(() => session.on(kind, (event) => latest.current(event)), [session, kind]);
}
