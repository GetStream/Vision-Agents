import { type Frame, nested, text } from "./socket.js";
import { ToolCall, type ToolLine } from "./tools.js";

/** The calls that asked to be held, waiting on the router's `tool_held` or `tool_cancel`. */
export class Holds {
  private readonly waiting = new Map<string, (held: boolean) => void>();
  private readonly holding = new Set<string>();

  /** Sends the hold and waits for the router to say whether the call was. */
  async ask(id: string, send: () => void): Promise<boolean> {
    const answer = new Promise<boolean>((resolve) => this.waiting.set(id, resolve));
    try {
      send();
      const held = await answer;
      if (held) {
        this.holding.add(id);
      }
      return held;
    } finally {
      this.waiting.delete(id);
    }
  }

  /** Hands the router's answer to the call waiting on it, if one is. */
  answer(id: string, held: boolean): void {
    this.waiting.get(id)?.(held);
  }

  /** Whether the router keeps this call going when the caller talks over the agent. */
  held(id: string): boolean {
    return this.holding.has(id);
  }

  /** Drops what is known about a call once it has finished. */
  forget(id: string): void {
    this.holding.delete(id);
    this.waiting.get(id)?.(false);
  }
}

/** A call's way back to its session, over the socket that delivered it. */
export function socketLine(send: (frame: Record<string, unknown>) => void, holds: Holds): ToolLine {
  return {
    update: (id, said, verbatim) => send({ type: "tool_update", tool_call_id: id, text: said, verbatim }),
    hold: (id) => holds.ask(id, () => send({ type: "tool_hold", tool_call_id: id })),
  };
}

/**
 * The call a `tool_call` frame asks for, as the function is handed it. `sessionId` stands in
 * for a frame that does not name its session, as a router's from before it said does not.
 */
export function toolCallContextOf(
  frame: Frame,
  sessionId: string,
  line: ToolLine,
  lasting: AbortSignal,
): ToolCall {
  const user = nested(frame, "user");
  const conversation = nested(frame, "conversation");
  return new ToolCall(
    {
      id: text(frame, "id"),
      sessionId: text(frame, "session_id") || sessionId,
      turnId: text(frame, "turn_id"),
      user: {
        id: user ? text(user, "id") : "",
        name: user ? text(user, "name") : "",
        custom: (user && objectOf(user["custom"])) ?? {},
      },
      conversation: {
        cid: conversation ? text(conversation, "cid") : "",
        custom: (conversation && objectOf(conversation["custom"])) ?? {},
      },
      approved: frame["approved"] === true,
    },
    line,
    lasting,
  );
}

function objectOf(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}
