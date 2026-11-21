/**
 * React hooks over a session's stores, opted into by importing this entry:
 *
 * ```ts
 * import { useToolCall, useToolCalls, useTurnToolCalls } from "@stream-io/stream-agents/react";
 * ```
 *
 * Its own entry point so `react` is resolved only for an app that asked for it.
 */

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

import type { Session, SessionEvent } from "./session.js";
import type { ToolCallState } from "./tool-calls.js";
import { Tools, type Tool } from "./tools.js";

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
 * The tool calls one agent turn made, in the order they started, re-rendering as one of them
 * changes: the ones to show under that turn's message.
 */
export function useTurnToolCalls(session: Session, turnId: string): readonly ToolCallState[] {
  const read = useCallback(() => session.toolCalls.forTurn(turnId), [session, turnId]);
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

/**
 * Offers `tool` to the session's agent while the component is mounted, with `addTools`, and
 * withdraws it with `removeTool` on unmount or when `session` or `tool.name` changes. What the
 * model is told of it is read once per session and name; the latest `run` is the one called,
 * so a re-render with new state does not offer it again. Answers the last add or removal that
 * failed, such as the 409 for a name the session already has, or undefined.
 */
export function useTool<In>(session: Session | undefined, tool: Tool<In>): Error | undefined {
  const latest = useRef(tool);
  latest.current = tool;
  const queue = useRef(Promise.resolve());
  const [failure, setFailure] = useState<Error>();
  useEffect(
    () => (session ? offerTool(session, latest, queue, setFailure) : undefined),
    [session, tool.name],
  );
  return failure;
}

/**
 * @internal What `useTool` does when it mounts, returning what it does when it unmounts. Each
 * change waits in `queue` for the one before, so a tool is withdrawn only once it was added:
 * withdrawn after a refused add, it would be the session's own of the same name that went.
 */
export function offerTool<In>(
  session: Session,
  latest: { readonly current: Tool<In> },
  queue: { current: Promise<void> },
  failed: (cause: Error) => void,
): () => void {
  const { name } = latest.current;
  const report = (cause: unknown): void => failed(cause instanceof Error ? cause : new Error(String(cause)));
  const added = queue.current
    .then(() =>
      session.addTools(
        new Tools().register<In>({ ...latest.current, run: (input, context) => latest.current.run(input, context) }),
      ),
    )
    .then(
      () => true,
      (cause: unknown) => {
        report(cause);
        return false;
      },
    );
  queue.current = added.then(() => undefined);
  return () => {
    queue.current = added.then((offered) => (offered ? session.removeTool(name) : undefined)).catch(report);
  };
}
