import type { Client, Schemas } from "./client.js";
import { Conversation } from "./conversation.js";
import { ConfigurationError, RouterError } from "./errors.js";
import { CameraFrames } from "./frames.js";
import { Responses } from "./responses.js";
import { Socket, type Frame, flag, nested, text } from "./socket.js";
import { Speech } from "./speech.js";
import { ConversationState } from "./state.js";
import type { ChatClient, VideoClient } from "./stream.js";
import { Holds, socketLine, toolCallContextOf } from "./tool_line.js";
import type { ToolApproval, Tools } from "./tools.js";

/**
 * How many events are held for one reader of `events()` that has not taken them yet, and for
 * the first reader while there is none.
 *
 * Past this a reader is ended rather than the socket stalled or a hole left in what it reads,
 * as Swift's are; while nobody reads, the oldest is dropped, since a conversation nobody is
 * watching should still be held. A tool call is answered by the session's own loop rather than
 * by whoever is iterating, so neither ever drops a turn.
 */
const BUFFERED_EVENTS = 256;

/** One caller's `events()`: what arrived for it, and how to tell it more has. */
interface Reader {
  queue: SessionEvent[];
  wake: (() => void) | undefined;
  /** Fallen too far behind: it ends once it has read what it holds. */
  behind: boolean;
}

/** A call of one of the caller's functions, waiting for the person to allow it. */
export interface ToolApprovalRequest {
  /** The call's id, which `decide` answers it by. */
  readonly id: string;
  /** The function the model wants run. */
  readonly name: string;
  /** The arguments as the model wrote them, a JSON object encoded as a string. */
  readonly arguments: string;
  /** The question the function asks. */
  readonly approval: ToolApproval;
  /**
   * The model's own words for why it wants the call, from the argument the approval names,
   * or the empty string.
   */
  readonly reason: string;
}

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
 * response_delta, responded, blocked, speaking, spoke, turn, delegated, task_settled,
 * task_cancelled, tool_started, tool_ran, transferred, pressed, looked_up, backchannel,
 * interrupted, overlap_decided, conversation_compacted, conversation_updated, error and left,
 * and tool_call for a call waiting in `session.approvals`. The fields
 * below are filled from whichever of those carry them, and `frame` is the whole thing for
 * anything they do not cover.
 *
 * `speaking` is an utterance starting to be heard, and `text` is what the voice had been sent
 * for it by then; its `frame` has `turn_id`, `synthesis_id` and `at`. The utterance's `spoke`
 * follows with the same `synthesis_id`, its `characters` and `audio_duration_ms`, which is
 * how a page reveals a reply as it is said rather than as it is written. `session.speech`
 * does that sum, by turn, without a loop of its own.
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
   * `responses.items` reads the turn back — and `say`, `interrupt` and `events` have
   * nothing to send to.
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
  /** Starts and stops the agent talking on the session's call. */
  readonly voice: Voice;
  /**
   * How much of each of the agent's replies has been heard on the call, by turn: what a
   * page reveals a reply's Chat message by, as it is said rather than as it is written.
   * Fed from the socket beside `events()`, so it needs no loop of its own and takes nothing
   * from one. Empty for a conversation in writing, one not watched, or a router that does
   * not send `speaking`. While the agent is on the call, a reply known to be coming is held
   * at nothing heard until its voice starts, for up to `SPEECH_GRACE_MS`; `voice.stop()`,
   * the agent leaving the call and the session ending stop that.
   */
  readonly speech = new Speech({ voice: () => this.voice.started });
  /**
   * What the agent is doing: listening, responding, working or idle, and ended once the
   * conversation is over. Fed from the socket beside `events()`, like `speech`, so a page can
   * show it without a loop of its own. Not the lifecycle, which is `created.state`.
   */
  readonly state = new ConversationState();
  /**
   * The conversation as this device heard it: its turns, what the agent reported going wrong,
   * the calls waiting in `approvals`, and whether the socket is still carrying it. Fed from the
   * socket beside `events()`, like `state`.
   */
  readonly conversation: Conversation;
  /**
   * @internal What this device's camera showed lately, set by a `VoiceSession` while its
   * camera is on. The router's `get_video_frames` is answered from it.
   */
  cameraFrames: CameraFrames | undefined;

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
  /** What arrived while nobody was reading, for the first reader to start with. */
  private readonly backlog: SessionEvent[] = [];
  private readonly readers = new Set<Reader>();
  private readonly running = new Map<string, AbortController>();
  private readonly holds = new Holds();
  /** Aborts when the socket ends, which stops even a call that asked to be held. */
  private readonly ending = new AbortController();
  private readonly asking = new Map<string, { request: ToolApprovalRequest; frame: Frame }>();
  private readonly finished: Promise<void>;
  private ended = false;
  private stopped = false;

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
    this.conversation = new Conversation(socket !== undefined);
    this.responses = new Responses(client, created.id, (asked) => this.conversation.said(asked));
    this.voice = new Voice(client, created.id, Boolean(created.call_id), (on) =>
      this.speech.voiceChanged(on),
    );
    this.finished = socket ? this.watch() : Promise.resolve();
  }

  /**
   * Creates the session and starts watching it.
   *
   * With `start_voice` it resolves once the backend is in the session's call, so a session
   * that has opened is one that is already listening. Without it the conversation is held
   * in writing.
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
    return Session.startWatching(client, created, options);
  }

  /** @internal Watches a newly created session, stopping it if watching fails. */
  static async startWatching(
    client: Client,
    created: Schemas["Session"],
    options: SessionOptions = {},
  ): Promise<Session> {
    try {
      return await Session.watching(client, created, options);
    } catch (cause) {
      // The session is live in the backend even though nothing here can watch it, so it is
      // stopped rather than left holding a call nobody is listening to.
      await client.post("/v1/agents/sessions/{id}/stop", { path: { id: created.id } })
        .catch(() => undefined);
      throw cause;
    }
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

    const socket = await Socket.dial(
      client.backend,
      `/v1/agents/sessions/${encodeURIComponent(created.id)}/events`,
      query,
    );
    return new Session(client, created, socket, options.tools);
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
   * Why the socket stopped carrying the conversation, or undefined: while it is open, and once
   * it has ended normally. `conversation` says the same, for a page to follow.
   */
  get failure(): Error | undefined {
    return this.conversation.get().socketError;
  }

  /**
   * Calls of functions that ask first, waiting for the person to answer with `decide`,
   * oldest first. Each also arrives through `events` as a `tool_call`.
   */
  get approvals(): readonly ToolApprovalRequest[] {
    return this.conversation.get().approvals;
  }

  /**
   * Answers a call waiting in `approvals`.
   *
   * Allowed, the function runs and its result goes back as it would have. Declined, it never
   * runs: the model is told the person declined, and the call's step shows `summary`. A call
   * that is not waiting is left alone, so answering twice does nothing. If the answer cannot
   * be sent, the call goes on waiting.
   */
  decide(id: string, answer: { allowed: boolean; summary?: string }): void {
    const waiting = this.asking.get(id);
    if (!waiting) {
      return;
    }
    const ids: Record<string, string> = {};
    for (const key of ["request_id", "turn_id"]) {
      if (text(waiting.frame, key)) {
        ids[key] = text(waiting.frame, key);
      }
    }
    this.held().send({
      type: "tool_approval",
      tool_call_id: id,
      allowed: answer.allowed,
      ...(answer.summary ? { summary: answer.summary } : {}),
      ...ids,
    });
    this.asking.delete(id);
    this.approvalsChanged();
    if (answer.allowed) {
      void this.runTool(waiting.frame);
    } else {
      this.held().send({
        type: "tool_result",
        tool_call_id: id,
        ...ids,
        error: "The person declined this call.",
      });
    }
  }

  /**
   * Declines the sign-in a call of one of the agent's connectors waits for, whose step in the
   * conversation is `awaiting_authorization`: the person's Not now. The router answers the call
   * itself, so it runs nothing and the agent is told the person did not connect. A call that
   * is not waiting is left alone.
   */
  declineLogin(toolCallId: string): void {
    this.held().send({ type: "tool_approval", tool_call_id: toolCallId, allowed: false });
  }

  /**
   * Yields what the backend did until the conversation ends.
   *
   * Each loop over it is told everything from when it started reading, so a caption view and a
   * latency panel can each have one. The first loop also gets what arrived while nobody was
   * reading. A loop that falls 256 events behind is ended rather than left with a hole.
   */
  async *events(): AsyncGenerator<SessionEvent> {
    const reader: Reader = {
      queue: this.readers.size === 0 ? this.backlog.splice(0) : [],
      wake: undefined,
      behind: false,
    };
    this.readers.add(reader);
    try {
      while (true) {
        const next = reader.queue.shift();
        if (next !== undefined) {
          yield next;
          continue;
        }
        if (this.ended || reader.behind) {
          return;
        }
        await new Promise<void>((resolve) => {
          reader.wake = resolve;
        });
        reader.wake = undefined;
      }
    } finally {
      this.readers.delete(reader);
    }
  }

  /** Speaks text without going through the model, for when you know what should be said. */
  say(said: string, options: { interrupt?: boolean } = {}): void {
    if (options.interrupt) {
      this.held().send({ type: "interrupt" });
    }
    this.held().send({ type: "say", text: said });
  }

  /** Abandons the reply being spoken, or with `requestId` the one answering that request. */
  interrupt(options: { requestId?: string } = {}): void {
    this.held().send({
      type: "interrupt",
      ...(options.requestId ? { request_id: options.requestId } : {}),
    });
  }

  /**
   * Changes this session alone: its title, description, custom labels, whether it is pinned,
   * its models or its voice.
   *
   * A person's device may change the title, description and custom labels and pin or unpin
   * the session; the models and the voice are server side only. The agent config it started
   * from is untouched and a field left out is left as it is. A session that ended can still be
   * renamed, relabelled and pinned; the rest needs it running and takes over from its next
   * turn, so a reply being spoken finishes on what it started with. A target that does not
   * route is refused and the conversation carries on.
   */
  update(changes: Schemas["UpdateSessionRequest"]): Promise<Schemas["Session"]> {
    return this.client.patch("/v1/agents/sessions/{id}", {
      path: { id: this.id },
      body: changes,
    });
  }

  /**
   * Pins this conversation, which a query sorted by `pinned` lists first, and answers it with
   * its `pinned_at`. It moves nothing else. See `sessions.pin`.
   */
  pin(): Promise<Schemas["Session"]> {
    return this.update({ pinned: true });
  }

  /** Unpins this conversation, and answers it without a `pinned_at`. */
  unpin(): Promise<Schemas["Session"]> {
    return this.update({ pinned: false });
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
    return Session.startWatching(this.client, forked, {
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
   * `@stream-io/stream-agents/chat` once, so a caller who never touches chat installs nothing
   * and ships nothing. A caller who forgot the import is told which one.
   *
   * It opens on the `StreamChat` handed over with `client.use`, or else on one connected as
   * the user `setUser` named, with the Stream key the client was given, which every session
   * of this client shares.
   */
  chat(): Promise<SessionChat> {
    if (!this.chatPeer) {
      const opening = this.openChat();
      this.chatPeer = opening;
      // A failure is not kept: the next call tries again, since what failed may have been the
      // network, or a `setUser` that had not happened yet.
      opening.catch(() => {
        if (this.chatPeer === opening) {
          this.chatPeer = undefined;
        }
      });
    }
    return this.chatPeer;
  }

  /**
   * The Stream video call the agent is on, `agent:<session id>`.
   *
   * The same arrangement as chat: `@stream-io/video-client` is an optional peer dependency,
   * opted into by importing `@stream-io/stream-agents/video` once, and the client handed over
   * with `client.use` is used when there is one. A session held in writing has no call, and
   * asking for one says so.
   */
  video(): Promise<SessionVideo> {
    if (!this.videoPeer) {
      const opening = this.openVideo();
      this.videoPeer = opening;
      // Not kept when it fails, for the same reason as chat's, and so a failed join can be tried again.
      opening.catch(() => {
        if (this.videoPeer === opening) {
          this.videoPeer = undefined;
        }
      });
    }
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
   * @internal Stops watching the conversation without ending it, for a watcher of its own such
   * as an attached `VoiceSession`'s: the socket closes, and the session goes on for whoever
   * else holds it.
   */
  async stopWatching(): Promise<void> {
    this.socket?.close();
    await this.finished.catch(() => undefined);
  }

  /**
   * Stops the conversation. Safe to call after it has already ended. Everything it recorded
   * and remembered is kept; `delete` is what takes those away.
   */
  async close(): Promise<void> {
    if (this.stopped) {
      return;
    }
    if (this.socket?.open) {
      this.socket.send({ type: "close" });
    } else {
      try {
        await this.client.post("/v1/agents/sessions/{id}/stop", { path: { id: this.id } });
      } catch (cause) {
        if (!(cause instanceof RouterError) || cause.status !== 404 || cause.code !== "session_not_found") {
          throw cause;
        }
      }
    }
    this.stopped = true;
    this.socket?.close();
    await this.finished;
    if (!this.socket) {
      this.end();
    }
  }

  /**
   * Reads the socket until the conversation ends, answering the model's tool calls as they
   * arrive and handing everything else to whoever is reading events.
   *
   * It runs whether or not anybody is reading, because a tool call the model is waiting on
   * cannot depend on the caller having started a loop.
   */
  private async watch(): Promise<void> {
    let stoppedBy: Error | undefined;
    try {
      for await (const message of this.held().messages()) {
        if (message instanceof Uint8Array) {
          continue;
        }
        // Read here rather than off events(), so a store needs no reader of its own.
        this.speech.observe(message);
        this.state.observe(message);
        this.conversation.observe(message);

        if (message.type === "tool_call" && text(message, "name") === CameraFrames.toolName) {
          // Answered by the session rather than offered as a tool, and only by a device whose
          // camera is on and that the router asked: a video worker may be the one answering it.
          if (this.cameraFrames && message["run"] !== false) {
            this.show(this.cameraFrames, message);
          }
          continue;
        }
        if (message.type === "tool_call") {
          // Every watcher is sent every call, and the router says which are to run it. One
          // from before it said holds the same rule: answering a tool held elsewhere would
          // beat whoever runs it, with an error.
          if (message["run"] === false || !this.tools?.has(text(message, "name"))) {
            continue;
          }
          // A caller who already said yes on a voice call is not asked again.
          const approval = message.approved === true ? undefined : this.tools.approval(text(message, "name"));
          if (approval) {
            this.hold(message, approval);
            this.deliver(eventOf(message));
          } else {
            void this.runTool(message);
          }
          continue;
        }
        if (message.type === "tool_cancel") {
          const id = text(message, "id");
          this.holds.answer(id, false);
          if (!this.holds.held(id)) {
            this.running.get(id)?.abort();
          }
          if (this.asking.delete(id)) {
            this.approvalsChanged();
          }
          continue;
        }
        if (message.type === "tool_held") {
          this.holds.answer(text(message, "id"), true);
        }
        this.deliver(eventOf(message));
      }
    } catch (cause) {
      stoppedBy = cause instanceof Error ? cause : new Error(String(cause));
      throw cause;
    } finally {
      this.end(stoppedBy ?? this.socket?.closedBy);
    }
  }

  private end(cause?: Error): void {
    this.ended = true;
    this.speech.close();
    this.state.end();
    this.asking.clear();
    this.conversation.disconnected(cause);
    this.ending.abort();
    for (const cancel of this.running.values()) {
      cancel.abort();
    }
    for (const reader of this.readers) {
      reader.wake?.();
    }
  }

  private hold(frame: Frame, approval: ToolApproval): void {
    let reason = "";
    if (approval.reasonArgument) {
      try {
        const args: unknown = JSON.parse(text(frame, "arguments") || "{}");
        const said = (args as Record<string, unknown> | null)?.[approval.reasonArgument];
        reason = typeof said === "string" ? said : "";
      } catch {
        // Arguments the model garbled carry no reason; the call itself still waits.
      }
    }
    const id = text(frame, "id");
    this.asking.set(id, {
      request: { id, name: text(frame, "name"), arguments: text(frame, "arguments"), approval, reason },
      frame,
    });
    this.approvalsChanged();
  }

  /** Tells `conversation` which calls are waiting now. */
  private approvalsChanged(): void {
    this.conversation.approvalsChanged([...this.asking.values()].map(({ request }) => request));
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
    // A durable request's result is only accepted back with the request and turn it names.
    for (const key of ["request_id", "turn_id"]) {
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
      const line = socketLine((sent) => this.socket?.open && this.socket.send(sent), this.holds);
      const toolCallContext = toolCallContextOf(frame, this.id, line, this.ending.signal);
      result["output"] = await this.tools.call(name, text(frame, "arguments"), cancel.signal, toolCallContext);
    } catch (cause) {
      result["error"] = cause instanceof Error ? cause.message : String(cause);
    } finally {
      this.running.delete(id);
      this.holds.forget(id);
    }

    if (this.socket?.open) {
      this.socket.send(result);
    }
  }

  /**
   * Answers the router's request for frames with the ones the camera showed at the moment it
   * names: what the frames are, then the frames, oldest first. Nothing recent enough is said
   * as the answer rather than as a failure, as the router's own video workers say it, so the
   * agent can ask the person to show it again.
   */
  private show(frames: CameraFrames, frame: Frame): void {
    const result: Record<string, unknown> = { type: "tool_result", tool_call_id: text(frame, "id") };
    for (const key of ["request_id", "turn_id"]) {
      if (text(frame, key)) {
        result[key] = text(frame, key);
      }
    }

    const asked = argumentsOf(frame);
    const at = typeof asked["at_ms"] === "number" ? asked["at_ms"] : Date.now();
    const limit = typeof asked["limit"] === "number" ? asked["limit"] : 1;
    try {
      const selected = frames.select(at, limit);
      const described = JSON.stringify({
        clock: "device_unix_ms",
        frames: selected.map((each) => ({
          captured_at_ms: Math.floor(each.capturedAt),
          frame_id: `camera-${each.id}`,
        })),
        source: "camera",
      });
      // Pictures make the output an array of parts, which is how the router takes them.
      result["output"] = [
        { type: "text", text: described },
        ...selected.map((each) => ({
          type: "image_url",
          image_url: { url: `data:image/jpeg;base64,${each.jpeg}` },
        })),
      ];
    } catch (cause) {
      result["output"] = cause instanceof Error ? cause.message : String(cause);
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
    if (!this.voice.started) {
      throw new ConfigurationError(
        "this conversation is held in writing, so there is no call to join",
      );
    }

    const connected = await this.client.peers.video();
    return { client: connected, call: connected.call("agent", this.id) };
  }

  private deliver(event: SessionEvent): void {
    if (this.readers.size === 0) {
      this.backlog.push(event);
      if (this.backlog.length > BUFFERED_EVENTS) {
        this.backlog.shift();
      }
      return;
    }
    for (const reader of this.readers) {
      if (reader.queue.length >= BUFFERED_EVENTS) {
        reader.behind = true;
      } else if (!reader.behind) {
        reader.queue.push(event);
      }
      reader.wake?.();
    }
  }
}

/**
 * The agent talking on a session's call, `agent:<session id>`.
 *
 * Typed and spoken turns are one conversation, with one history and one channel, whether
 * voice is on or not.
 */
export class Voice {
  private readonly client: Client;
  private readonly sessionId: string;
  private on: boolean;
  private readonly changed: ((on: boolean) => void) | undefined;

  /** `changed` hears the agent being put on the call or taken off it from here. */
  constructor(client: Client, sessionId: string, on: boolean, changed?: (on: boolean) => void) {
    this.client = client;
    this.sessionId = sessionId;
    this.on = on;
    this.changed = changed;
  }

  /** Whether the agent is on the call, as far as this process knows. */
  get started(): boolean {
    return this.on;
  }

  /** Has the agent join the call and carry the conversation on there. Already on does nothing. */
  async start(): Promise<Schemas["Session"]> {
    const started = await this.client.post("/v1/agents/sessions/{id}/voice", {
      path: { id: this.sessionId },
    });
    this.on = true;
    this.changed?.(true);
    return started;
  }

  /** Takes the agent off the call and carries the conversation on in writing. Already off does nothing. */
  async stop(): Promise<Schemas["Session"]> {
    const stopped = await this.client.delete("/v1/agents/sessions/{id}/voice", {
      path: { id: this.sessionId },
    });
    this.on = false;
    this.changed?.(false);
    return stopped;
  }
}

/** A tool call's arguments, which arrive as a JSON object encoded as a string. Garbled ones are none. */
function argumentsOf(frame: Frame): Frame {
  const given = frame["arguments"];
  if (typeof given === "object" && given !== null && !Array.isArray(given)) {
    return given as Frame;
  }
  try {
    const parsed: unknown = JSON.parse(typeof given === "string" ? given : "{}");
    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Frame)
      : {};
  } catch {
    return {};
  }
}

/**
 * A connector tool call the provider refused because the person's own connection lacks access
 * it asked for, and the step-up consent the router began for it. The old grant keeps working
 * until the step-up succeeds, and the same call works afterwards in the same session.
 *
 * A page finishes it as it does any consent: it opens `launchUrl` in a popup from a page of the
 * session's `return_url` origin and posts it `handoffToken`.
 */
export interface ConnectorScopeRequired {
  /** The binding's alias. */
  readonly name: string;
  readonly connectorId: string;
  readonly connectionId: string;
  /** What the provider asked for, empty for a claims challenge. */
  readonly scopes: readonly string[];
  readonly authorizationId: string;
  readonly launchUrl: string;
  readonly handoffToken: string;
  /** When the consent stops being offered, if the router said. */
  readonly expiresAt?: Date;
}

/** The step-up a `connector_scope_required` event asks for, or undefined for any other event. */
export function connectorScopeRequiredOf(event: SessionEvent): ConnectorScopeRequired | undefined {
  if (event.kind !== "connector_scope_required") {
    return undefined;
  }
  const { frame } = event;
  const scopes = Array.isArray(frame["scopes"])
    ? frame["scopes"].filter((scope): scope is string => typeof scope === "string")
    : [];
  const expires = Date.parse(text(frame, "expires_at"));
  return {
    name: text(frame, "name"),
    connectorId: text(frame, "connector_id"),
    connectionId: text(frame, "connection_id"),
    scopes,
    authorizationId: text(frame, "authorization_id"),
    launchUrl: text(frame, "launch_url"),
    handoffToken: text(frame, "handoff_token"),
    ...(Number.isFinite(expires) ? { expiresAt: new Date(expires) } : {}),
  };
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
