/**
 * A reply's thinking, put back together from its live updates.
 *
 * The router stores only the opening of each round of reasoning, as its step's `preview`.
 * The whole of it reaches people watching the reply live: each update carries the next window
 * of the round being streamed, in the message's `reasoning` field. Read every update into one
 * of these, from the client's `message.updated` events rather than the channel's message
 * list, which can fold several updates into one, and show a step by its reply and its id:
 * every reply numbers its steps from `r1`.
 *
 * ```ts
 * client.on("message.updated", (event) => event.message && reasoning.read(event.message));
 * reasoning.get(message.id, step.id);
 * ```
 */
export class LiveReasoning {
  private readonly replies = new Map<string, Map<string, { text: string; end: number }>>();

  /**
   * The thinking of reasoning step `stepId` of reply `messageId` so far, or undefined before
   * any of it arrived.
   */
  get(messageId: string, stepId: string): string | undefined {
    return this.replies.get(messageId)?.get(stepId)?.text;
  }

  /** Reads the window a live update carries, if it carries one. Takes a Stream Chat message. */
  read(message: { id: string; reasoning?: unknown }): void {
    const window = message.reasoning;
    if (typeof window !== "object" || window === null) {
      return;
    }
    const { id, offset, text } = window as Record<string, unknown>;
    if (typeof id === "string" && typeof offset === "number" && typeof text === "string") {
      this.add(text, offset, message.id, id);
    }
  }

  /**
   * Adds what `text`, which starts `offset` code points into the thinking of step `id` of
   * reply `messageId`, has that is not held yet. A window that repeats recent thinking, for
   * somebody who joined midway, overlaps what is held; one that starts past it means updates
   * were missed, and is kept after a paragraph break rather than run on.
   */
  private add(text: string, offset: number, messageId: string, id: string): void {
    let steps = this.replies.get(messageId);
    if (!steps) {
      steps = new Map();
      this.replies.set(messageId, steps);
    }
    const held = steps.get(id) ?? { text: "", end: 0 };
    const points = Array.from(text);
    const end = offset + points.length;
    if (end <= held.end) {
      return;
    }
    const gap = offset > held.end && held.text ? "\n\n" : "";
    const fresh = points.slice(Math.max(0, held.end - offset)).join("");
    steps.set(id, { text: held.text + gap + fresh, end });
  }
}
