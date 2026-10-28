import { text, type Frame } from "./socket.js";

/**
 * What the agent is doing.
 *
 * - `idle`: waiting to be spoken to.
 * - `listening`: somebody is talking and being transcribed.
 * - `responding`: the model is answering.
 * - `working`: skills are thinking, named in `skills` so a page can say what about.
 * - `ended`: the conversation is over.
 */
export type AgentState = "idle" | "listening" | "responding" | "working" | "ended";

/** Where the conversation is. The same object until it changes. */
export interface ConversationSnapshot {
  readonly kind: AgentState;
  /** The skills thinking, while `kind` is `working`. Empty otherwise. */
  readonly skills: readonly string[];
}

const snapshots: Record<Exclude<AgentState, "working">, ConversationSnapshot> = {
  idle: Object.freeze({ kind: "idle", skills: Object.freeze([]) }),
  listening: Object.freeze({ kind: "listening", skills: Object.freeze([]) }),
  responding: Object.freeze({ kind: "responding", skills: Object.freeze([]) }),
  ended: Object.freeze({ kind: "ended", skills: Object.freeze([]) }),
};

/**
 * What the agent is doing, folded from the session's socket: `session.state`.
 *
 * Not the session's lifecycle, which is `session.created.state` (`live`, …): this is the
 * moment to moment of a conversation, for a listening or thinking indicator.
 *
 * Fed beside `events()` rather than from it, so it takes nothing from the one reader that
 * stream allows. `get` and `subscribe` are bound, so a page can hand them over as they are:
 *
 * ```ts
 * const state = useSyncExternalStore(session.state.subscribe, session.state.get);
 * ```
 */
export class ConversationState {
  private snapshot: ConversationSnapshot = snapshots.idle;
  private readonly listeners = new Set<() => void>();

  /** Where the conversation is now. */
  readonly get = (): ConversationSnapshot => this.snapshot;

  /** Calls `listener` whenever the state changes. It returns what stops it. */
  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  /** @internal Folds one frame off the session's socket in. Frames with no bearing leave it alone. */
  observe(frame: Frame): void {
    if (this.snapshot.kind === "ended") {
      return;
    }
    switch (frame.type) {
      case "hearing":
        this.change(snapshots.listening);
        break;
      case "responding":
      case "response_delta":
        this.change(snapshots.responding);
        break;
      case "responded":
      case "interrupted":
        this.change(snapshots.idle);
        break;
      case "delegated":
        this.work([...this.working(), text(frame, "skill")]);
        break;
      case "task_settled":
      case "task_cancelled": {
        const skill = text(frame, "skill");
        this.work(this.working().filter((each) => each !== skill));
        break;
      }
      case "left":
        this.end();
        break;
      default:
        break;
    }
  }

  /** @internal The conversation is over: the agent left, it was closed, or its socket dropped. */
  end(): void {
    this.change(snapshots.ended);
  }

  private working(): readonly string[] {
    return this.snapshot.kind === "working" ? this.snapshot.skills : [];
  }

  /** Skills still thinking, or the model answering once none are. */
  private work(skills: string[]): void {
    this.change(
      skills.length === 0
        ? snapshots.responding
        : Object.freeze({ kind: "working", skills: Object.freeze(skills) }),
    );
  }

  private change(next: ConversationSnapshot): void {
    if (next === this.snapshot) {
      return;
    }
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
