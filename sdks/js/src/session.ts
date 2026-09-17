import type { Client, Schemas } from "./client.js";
import { Socket, type Frame, flag, nested, text } from "./socket.js";
import type { Tools } from "./tools.js";

/**
 * How many events are held for a caller that is not reading them yet.
 *
 * Past this the oldest is dropped rather than the socket stalled: a conversation that
 * nobody is watching should still be held, and a tool call is answered by the reader below
 * rather than by whoever is iterating, so dropping an event never drops a turn.
 */
const BUFFERED_EVENTS = 256;

/** Somebody on the call, as the backend reports them. */
export interface Participant {
  readonly id: string;
  readonly userId: string;
  readonly name: string;
}

/**
 * One thing the conversation did.
 *
 * `kind` is the backend's own name for it: joined, hearing, heard, decision, responding,
 * response_delta, responded, blocked, spoke, turn, delegated, task_settled, task_cancelled,
 * tool_started, tool_ran, transferred, pressed, looked_up, backchannel, interrupted,
 * overlap_decided, conversation_compacted, conversation_updated, error and left. The fields
 * below are filled from whichever of those carry them, and `frame` is the whole thing for
 * anything they do not cover.
 */
export interface SessionEvent {
  readonly kind: string;
  readonly text: string;
  readonly participant?: Participant;
  readonly interrupted: boolean;
  readonly pendingWork: boolean;
  readonly error: string;
  readonly frame: Frame;
}

export interface SessionOptions {
  /** The caller's own functions, which the model is offered and this process runs. */
  tools?: Tools;
  /** Also report what the caller is part way through saying, as `hearing` events. */
  interim?: boolean;
  /** Report the router's own routing decisions. On by default at the backend. */
  decisions?: boolean;
}

/**
 * One conversation, held in the acceleration backend.
 *
 * Nothing here does inference or touches media. The backend joins the call, hears the
 * caller, answers and speaks, and what arrives here are the events saying so. What stays
 * here is function calling, because the functions are here.
 */
export class Session {
  /** What the router said when it created this. */
  readonly created: Schemas["Session"];

  private readonly socket: Socket;
  private readonly client: Client;
  private readonly tools: Tools | undefined;
  private readonly buffered: SessionEvent[] = [];
  private readonly waiting: ((next: IteratorResult<SessionEvent>) => void)[] = [];
  private readonly running = new Map<string, AbortController>();
  private readonly finished: Promise<void>;
  private ended = false;

  private constructor(
    client: Client,
    created: Schemas["Session"],
    socket: Socket,
    tools: Tools | undefined,
  ) {
    this.client = client;
    this.created = created;
    this.socket = socket;
    this.tools = tools;
    this.finished = this.watch();
  }

  /**
   * Creates the session and starts watching it.
   *
   * It resolves once the backend is in the call, so a session that has opened is one that
   * is already listening. A request with no `call_id` and `text: true` holds the
   * conversation in writing instead.
   */
  static async open(
    client: Client,
    request: Schemas["CreateSessionRequest"],
    options: SessionOptions = {},
  ): Promise<Session> {
    const declared = options.tools?.declared() ?? [];
    const created = await client.post("/v1/agents/sessions", {
      body: declared.length > 0 ? { ...request, tools: declared } : request,
    });

    const query: Record<string, string> = {};
    if (options.interim) {
      query["interim"] = "true";
    }
    if (options.decisions === false) {
      query["decisions"] = "false";
    }

    const url = await client.backend.socketURL(
      `/v1/agents/sessions/${encodeURIComponent(created.id)}/events`,
      query,
    );

    try {
      const socket = await Socket.open(client.backend, url);
      return new Session(client, created, socket, options.tools);
    } catch (cause) {
      // The session is live in the backend even though nothing here can watch it, so it is
      // closed rather than left holding a call nobody is listening to.
      await client
        .delete("/v1/agents/sessions/{id}", { path: { id: created.id } })
        .catch(() => undefined);
      throw cause;
    }
  }

  /** The backend's id for the session. */
  get id(): string {
    return this.created.id;
  }

  /** The conversation replies are written into, for a session that persists one. */
  get conversationId(): string {
    return this.created.conversation_id ?? "";
  }

  /** Whether the conversation is still being held. */
  get live(): boolean {
    return !this.ended;
  }

  /**
   * Yields what the backend did until the conversation ends.
   *
   * There is one stream: two loops over it would take half the events each.
   */
  async *events(): AsyncGenerator<SessionEvent> {
    while (true) {
      const next = this.buffered.shift();
      if (next !== undefined) {
        yield next;
        continue;
      }
      if (this.ended) {
        return;
      }
      const waited = await new Promise<IteratorResult<SessionEvent>>((resolve) => {
        this.waiting.push(resolve);
      });
      if (waited.done) {
        return;
      }
      yield waited.value;
    }
  }

  /** Speaks text without going through the model, for when you know what should be said. */
  say(said: string, options: { interrupt?: boolean } = {}): void {
    if (options.interrupt) {
      this.socket.send({ type: "interrupt" });
    }
    this.socket.send({ type: "say", text: said });
  }

  /** Answers text through the model, as though it had been said on the call. */
  respond(said: string, options: { interrupt?: boolean } = {}): void {
    if (options.interrupt) {
      this.socket.send({ type: "interrupt" });
    }
    this.socket.send({ type: "respond", text: said });
  }

  /** Abandons the reply being spoken. */
  interrupt(): void {
    this.socket.send({ type: "interrupt" });
  }

  /** Changes what the agent is told to be, from the next turn. */
  setInstructions(instructions: string): void {
    this.socket.send({ type: "instructions", instructions });
  }

  /** Resolves when the conversation ends. */
  async wait(): Promise<void> {
    await this.finished;
  }

  /** Ends the conversation. Safe to call after it has already ended. */
  async close(): Promise<void> {
    if (this.socket.open) {
      this.socket.send({ type: "close" });
    } else if (!this.ended) {
      await this.client
        .delete("/v1/agents/sessions/{id}", { path: { id: this.id } })
        .catch(() => undefined);
    }
    this.socket.close();
    await this.finished;
  }

  /**
   * Reads the socket until the conversation ends, answering the model's tool calls as they
   * arrive and handing everything else to whoever is reading events.
   *
   * It runs whether or not anybody is reading, because a tool call the model is waiting on
   * cannot depend on the caller having started a loop.
   */
  private async watch(): Promise<void> {
    try {
      for await (const message of this.socket.messages()) {
        if (message instanceof Uint8Array) {
          continue;
        }

        if (message.type === "tool_call") {
          void this.runTool(message);
          continue;
        }
        if (message.type === "tool_cancel") {
          this.running.get(text(message, "id"))?.abort();
          continue;
        }
        this.deliver(eventOf(message));
      }
    } finally {
      this.ended = true;
      for (const cancel of this.running.values()) {
        cancel.abort();
      }
      for (const resolve of this.waiting.splice(0)) {
        resolve({ done: true, value: undefined });
      }
    }
  }

  /**
   * Runs one of the caller's functions and answers the model with what it said.
   *
   * A failure is reported rather than dropped: the model is mid-sentence waiting for this,
   * and it can only say something useful about a tool that did not work if it is told that
   * it did not work.
   */
  private async runTool(frame: Frame): Promise<void> {
    const id = text(frame, "id");
    const name = text(frame, "name");
    const result: Record<string, unknown> = { type: "tool_result", tool_call_id: id };

    const cancel = new AbortController();
    this.running.set(id, cancel);
    try {
      if (!this.tools) {
        throw new Error(`${name} was asked for, and no tools are registered`);
      }
      result["output"] = await this.tools.call(name, text(frame, "arguments"), cancel.signal);
    } catch (cause) {
      result["error"] = cause instanceof Error ? cause.message : String(cause);
    } finally {
      this.running.delete(id);
    }

    if (this.socket.open) {
      this.socket.send(result);
    }
  }

  private deliver(event: SessionEvent): void {
    const resolve = this.waiting.shift();
    if (resolve) {
      resolve({ done: false, value: event });
      return;
    }
    this.buffered.push(event);
    if (this.buffered.length > BUFFERED_EVENTS) {
      this.buffered.shift();
    }
  }
}

/** Fills in the fields the frames that carry them have in common. */
export function eventOf(frame: Frame): SessionEvent {
  const who = nested(frame, "participant");
  return {
    kind: frame.type ?? "",
    text: text(frame, "text"),
    interrupted: flag(frame, "interrupted"),
    pendingWork: flag(frame, "pending_work"),
    error: text(frame, "error"),
    frame,
    ...(who
      ? {
          participant: {
            id: text(who, "id"),
            userId: text(who, "user_id"),
            name: text(who, "name"),
          },
        }
      : {}),
  };
}
