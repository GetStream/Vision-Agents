import type { Client, Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";
import { Responses } from "./responses.js";
import { Socket, type Frame, flag, nested, text } from "./socket.js";
import type { ChatClient, VideoClient } from "./stream.js";
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
  /**
   * Whether to watch the conversation. On by default.
   *
   * A watched conversation is the lower-latency arrangement and the only one that reports
   * what the agent is doing word by word. A page on a hosted deployment can watch one too:
   * the socket carries the credential in its query, where the proxy reads it. Off, the
   * conversation is opened and read over HTTP instead — `responses.create()` asks,
   * `responses.items` reads the turn back — and `say`, `interrupt`,
   * `setInstructions` and `events` have nothing to send to.
   */
  watch?: boolean;
}

/** What to change about a conversation while continuing it. */
export interface ForkOptions
  extends Omit<Schemas["ForkSessionRequest"], "model_overwrites">,
    SessionOptions {
  /** What to change about the models, over whatever the parent was running. */
  modelOverwrites?: Schemas["ModelOverwrites"];
}

/**
 * Stream Chat, as this session's transcript.
 *
 * Typed loosely on purpose. `stream-chat` is an optional peer dependency, so its types are
 * not here to import, and declaring them would mean keeping a copy in step with a package
 * this one does not depend on. A caller who has installed it gets the real types by naming
 * them: `(await session.chat()).channel as Channel`.
 */
export interface SessionChat {
  /** The connected `StreamChat` client. */
  client: ChatClient;
  /** The channel replies are written into. */
  channel: unknown;
}

/** Stream video, as the call this session is on. */
export interface SessionVideo {
  /** The connected `StreamVideoClient`. */
  client: VideoClient;
  /** The call the agent is in, ready to be joined. */
  call: unknown;
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
  /** This conversation's turns, and what each of them was made of. */
  readonly responses: Responses;

  /**
   * Undefined for a conversation that is read and written to over HTTP rather than watched
   * (`watch: false`). Turns go in through `responses.create` and come back out of
   * `responses.items`.
   */
  private readonly socket: Socket | undefined;
  /** The Stream Chat channel and the video call, each opened on first use. */
  private chatPeer: Promise<SessionChat> | undefined;
  private videoPeer: Promise<SessionVideo> | undefined;
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
    socket: Socket | undefined,
    tools: Tools | undefined,
  ) {
    this.client = client;
    this.created = created;
    this.socket = socket;
    this.tools = tools;
    this.responses = new Responses(client, created.id, Boolean(created.conversation_id));
    this.finished = socket ? this.watch() : Promise.resolve();
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
    return Session.watching(client, created, options);
  }

  /**
   * Starts watching a session the router has already created.
   *
   * Separate from `open` because a fork is created by a different request and is otherwise
   * the same thing afterwards: one socket, one tool runner, one way of being closed.
   */
  static async watching(
    client: Client,
    created: Schemas["Session"],
    options: SessionOptions = {},
  ): Promise<Session> {
    // A conversation nothing is watching. The turns are the whole of what a caller here
    // does with it, and asking for a socket first would mean a page could not have one:
    // the proxy refuses the only socket a browser is able to open.
    if (options.watch === false) {
      return new Session(client, created, undefined, options.tools);
    }

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
      // stopped rather than left holding a call nobody is listening to.
      await client
        .post("/v1/agents/sessions/{id}/stop", { path: { id: created.id } })
        .catch(() => undefined);
      throw cause;
    }
  }

  /** The backend's id for the session. */
  get id(): string {
    return this.created.id;
  }

  /** The conversation replies are written into, for a text session that is not incognito. */
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
      this.held().send({ type: "interrupt" });
    }
    this.held().send({ type: "say", text: said });
  }

  /** Abandons the reply being spoken, or with `commandId` the one answering that command. */
  interrupt(options: { commandId?: string } = {}): void {
    this.held().send({
      type: "interrupt",
      ...(options.commandId ? { command_id: options.commandId } : {}),
    });
  }

  /** Changes what the agent is told to be, from the next turn. */
  setInstructions(instructions: string): void {
    this.held().send({ type: "instructions", instructions });
  }

  /**
   * Changes this session alone: its title, description, custom labels, instructions, models
   * or voice.
   *
   * Server side only. The agent config it started from is untouched and a field left out is
   * left as it is. A session that ended can still be renamed and relabelled; the rest needs
   * it running and takes over from its next turn, so a reply being spoken finishes on what it
   * started with. A target that does not route is refused and the conversation carries on.
   */
  update(changes: Schemas["UpdateSessionRequest"]): Promise<Schemas["Session"]> {
    return this.client.patch("/v1/agents/sessions/{id}", {
      path: { id: this.id },
      body: changes,
    });
  }

  /**
   * The socket, or what to do instead of it.
   *
   * Everything reached through here is a message to a conversation being watched, and a
   * conversation opened with `watch: false` is not being watched by anything. Saying so, and
   * saying what does work, beats a property read on undefined: this is the difference
   * between a page and a server, and it is the first thing a caller writing a page hits.
   */
  private held(): Socket {
    if (!this.socket) {
      throw new ConfigurationError(
        "this conversation is not being watched, so there is nothing to send to: open it " +
          "without watch: false, or use responses.create() and read the turn back through " +
          "responses.items",
      );
    }
    return this.socket;
  }

  /**
   * Continues this conversation as a new one.
   *
   * What a fork is for is asking the same question differently: from here on with a harder
   * model, or of a different agent, or down a branch you want to keep separately from the one
   * you already have. The parent is untouched and keeps its own transcript.
   *
   * The history comes across by default. The fork writes into a channel of its own, so the
   * two do not end up interleaved in one transcript with no way to tell which turn belonged
   * to which. An incognito parent cannot be forked, because there is nothing to fork from.
   */
  async fork(options: ForkOptions = {}): Promise<Session> {
    const { tools, interim, decisions, watch, modelOverwrites, ...rest } = options;
    const forked = await this.client.post("/v1/agents/sessions/{id}/fork", {
      path: { id: this.id },
      body: {
        ...rest,
        ...(modelOverwrites ? { model_overwrites: modelOverwrites } : {}),
      },
    });
    // The fork inherits the parent's tools unless it was given its own: the functions are
    // here in this process, and a conversation continued without them would offer the model
    // tools it cannot run.
    const inherited = tools ?? this.tools;
    return Session.watching(this.client, forked, {
      ...(inherited ? { tools: inherited } : {}),
      ...(interim === undefined ? {} : { interim }),
      ...(decisions === undefined ? {} : { decisions }),
      // A fork of a conversation nothing is watching is not watched either, unless the
      // caller says otherwise: whatever stopped the parent holding a socket still holds.
      watch: watch ?? Boolean(this.socket),
    });
  }

  /**
   * The Stream Chat channel this conversation is written into.
   *
   * `stream-chat` is an optional peer dependency, opted into by importing
   * `@stream-io/vision-agents/chat` once, so a caller who never touches chat installs nothing
   * and ships nothing. A caller who forgot the import is told which one.
   *
   * It opens on the `StreamChat` handed over with `client.use`, or else on one connected as
   * the user `setUser` named, with the Stream key the client was given, which every session
   * of this client shares.
   */
  chat(): Promise<SessionChat> {
    this.chatPeer ??= this.openChat();
    return this.chatPeer;
  }

  /**
   * The Stream video call the agent is on.
   *
   * The same arrangement as chat: `@stream-io/video-client` is an optional peer dependency,
   * opted into by importing `@stream-io/vision-agents/video` once, and the client handed over
   * with `client.use` is used when there is one. A session held in writing has no call, and
   * asking for one says so.
   */
  video(): Promise<SessionVideo> {
    this.videoPeer ??= this.openVideo();
    return this.videoPeer;
  }

  /** Resolves when the conversation ends. */
  async wait(): Promise<void> {
    await this.finished;
  }

  /**
   * Deletes what this conversation remembered. Ending it keeps its memories, so the next
   * conversation knows what this one established; this is how to take them back. Server
   * side only.
   */
  deleteMemories(): Promise<void> {
    return this.client.delete("/v1/agents/sessions/{id}/memories", { path: { id: this.id } });
  }

  /**
   * Deletes this conversation: it is stopped, and its turns and what it remembered are
   * deleted with it. See `sessions.delete`.
   */
  delete(): Promise<void> {
    return this.client.delete("/v1/agents/sessions/{id}", { path: { id: this.id } });
  }

  /**
   * Stops the conversation. Safe to call after it has already ended. Everything it recorded
   * and remembered is kept; `delete` is what takes those away.
   */
  async close(): Promise<void> {
    if (this.socket?.open) {
      this.socket.send({ type: "close" });
    } else if (!this.ended) {
      await this.client
        .post("/v1/agents/sessions/{id}/stop", { path: { id: this.id } })
        .catch(() => undefined);
    }
    this.socket?.close();
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
      for await (const message of this.held().messages()) {
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
    // A durable command's result is only accepted back with the command and turn it names.
    for (const key of ["command_id", "turn_id"]) {
      if (text(frame, key)) {
        result[key] = text(frame, key);
      }
    }

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

    if (this.socket?.open) {
      this.socket.send(result);
    }
  }

  private async openChat(): Promise<SessionChat> {
    const channel = this.created.conversation_id ?? "";
    if (!channel) {
      throw new ConfigurationError(
        "this session keeps no transcript, so there is no channel to read: only a text " +
          "session has one, and an incognito session never does",
      );
    }
    // The wire writes the channel as type:id, which is what the backend calls a
    // conversation. Splitting it here keeps that spelling out of the caller's way.
    const [type, ...rest] = channel.split(":");

    const connected = await this.client.peers.chat();
    return { client: connected, channel: connected.channel(type ?? "agent", rest.join(":")) };
  }

  private async openVideo(): Promise<SessionVideo> {
    const callId = this.created.call_id ?? "";
    if (!callId) {
      throw new ConfigurationError(
        "this conversation is held in writing, so there is no call to join",
      );
    }

    const connected = await this.client.peers.video();
    return {
      client: connected,
      call: connected.call(this.created.call_type ?? "agent", callId),
    };
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
