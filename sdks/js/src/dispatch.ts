import type { Agent } from "./agent.js";
import type { BackendOptions } from "./backend.js";
import { Client } from "./client.js";
import { ConfigurationError, HostingRefusedError } from "./errors.js";
import type { AgentHandle } from "./handle.js";
import { callOf, messageOf, type InboundCall, type InboundMessage } from "./inbound.js";
import { createResponse, type AgentResponse } from "./responses.js";
import type { Session } from "./session.js";
import { Socket, number as numberOf, text, type Frame } from "./socket.js";
import type { Tools } from "./tools.js";

/** How many calls a worker takes at once when it does not say. */
const DEFAULT_CAPACITY = 4;

/** How often the worker tells the router how it is doing. */
const DEFAULT_REPORT_EVERY_MS = 15_000;

/** How long a round trip is waited for before the last figure is left standing. */
const PING_TIMEOUT_MS = 5_000;

export interface DispatchOptions {
  /** The router to wait on. */
  client?: Client | BackendOptions;
  /**
   * How many calls and messages to hold at once.
   *
   * The router passes over a worker that is full rather than queueing behind it, so this is
   * a promise about what this process can actually answer.
   */
  capacity?: number;
  /** How often to report load. */
  reportEveryMs?: number;
}

/** What to do with an arriving call. */
export type CallHandler = (call: InboundCall) => void | Promise<void>;

/**
 * What to do with a message written to an agent that is not running, or to a running session
 * whose agent leaves text to dispatch.
 */
export type MessageHandler = (message: InboundMessage) => void | Promise<void>;

export interface HostOptions {
  /**
   * How long the router waits for one tool call to be answered before telling the model it
   * failed. Not how long the worker runs. Left out, the router's default of two minutes.
   */
  toolTimeoutMs?: number;
}

/** One set of functions this worker runs for every session under an agent id. */
interface Hosting {
  agentId: string;
  tools: Tools;
  toolTimeoutMs: number;
}

/**
 * Waits for inbound calls and messages, and runs a handler for each one.
 *
 * Neither arrives here first: a caller reached a Stream call over SIP, or somebody wrote in
 * a channel, and the router found out by webhook. The agent, though, runs in this process.
 * So this connects out and waits, and the router pushes work down the connection as it
 * arrives — nothing here has to be publicly reachable.
 *
 * Several workers can wait at once, in which case the work is shared between them.
 *
 * Server side only. A worker is offered other people's callers, so anything that can open
 * this socket can answer for the whole app, and the router refuses a caller that
 * authenticated as an end user's device.
 *
 * ```ts
 * const dispatch = new Dispatch({ client: { customerId: "local" } });
 *
 * dispatch.onCall(async (call) => {
 *   const agent = new Agent({ name: "John", instructions: "Be brief." });
 *   const session = await agent.answer(call);
 *   await session.wait();
 * });
 *
 * await dispatch.run();
 * ```
 */
export class Dispatch {
  readonly client: Client;
  readonly capacity: number;

  /** What the router calls this connection, for matching a log line here against one there. */
  workerId = "";

  private readonly reportEveryMs: number;
  private readonly running = new Set<Promise<void>>();
  /**
   * The calls and messages alone, which is what the router counts against this worker's
   * capacity. A hosted tool call is not one of them: the router tracks those by the answer
   * it is waiting for.
   */
  private handling = 0;
  /**
   * Which session is answering which channel.
   *
   * A channel is one conversation, so the session that answered the last message on it is
   * the one that knows what has been said and should answer the next.
   */
  private readonly answering = new Map<string, Session>();
  private onCallHandler: CallHandler | undefined;
  private onMessageHandler: MessageHandler | undefined;
  private readonly hosted: Hosting[] = [];
  private socket: Socket | undefined;
  private pong: ((at: number) => void) | undefined;
  /** The last round trip measured, from this side, because this is the side audio crosses. */
  private latencyMs = 0;

  constructor(options: DispatchOptions = {}) {
    this.capacity = options.capacity ?? DEFAULT_CAPACITY;
    if (this.capacity < 1) {
      throw new ConfigurationError("a worker that can hold no calls cannot answer any");
    }
    this.reportEveryMs = options.reportEveryMs ?? DEFAULT_REPORT_EVERY_MS;
    this.client =
      options.client instanceof Client ? options.client : new Client(options.client ?? {});
  }

  /** How much work is being handled right now. */
  get active(): number {
    return this.running.size;
  }

  /**
   * Registers what to do with an arriving call.
   *
   * The handler runs on its own, so one long call does not stop the next from being
   * answered. What it throws is reported to the router with the call's `done`.
   */
  onCall(handler: CallHandler): this {
    this.onCallHandler = handler;
    return this;
  }

  /**
   * Registers what to do with a message written to an agent that is not running, or to a
   * running session whose agent leaves text to dispatch. That one has a `sessionId`, and is
   * answered with `answer`.
   */
  onMessage(handler: MessageHandler): this {
    this.onMessageHandler = handler;
    return this;
  }

  /**
   * Runs an agent's tools for every session opened under it, whoever opened it.
   *
   * A session's own functions run in the process that opened it, which is no use to a
   * conversation opened from a browser that wants to read a source tree. Hosting is the
   * other direction: the router offers `agent.tools` to each session naming the agent and
   * sends every call to a worker hosting them. Call before `run`.
   *
   * ```ts
   * const agent = client.agent("my-agent");
   * agent.tools.register({ name: "lookup", description: "Look up an order", run });
   * await new Dispatch().host(agent).run();
   * ```
   */
  host(agent: AgentHandle, options: HostOptions = {}): this {
    this.hosted.push({
      agentId: agent.name,
      tools: agent.tools,
      toolTimeoutMs: options.toolTimeoutMs ?? 0,
    });
    return this;
  }

  /**
   * The session answering on this message's channel, started if none is.
   *
   * A channel is one conversation. The second message on it goes to the session that
   * answered the first, which is still open and knows what has been said; only a channel
   * nothing is answering calls `create`. Sessions are kept until this worker stops waiting,
   * so a conversation is not restarted between messages.
   */
  async sessionFor(
    message: InboundMessage,
    create: () => Agent | Promise<Agent>,
  ): Promise<Session> {
    if (message.sessionId) {
      throw new ConfigurationError(
        "a session is already holding this conversation; answer it there with answer",
      );
    }
    const open = this.answering.get(message.channelId);
    if (open?.live) {
      return open;
    }

    const agent = await create();
    const session = await agent.reply(message);
    this.answering.set(message.channelId, session);
    return session;
  }

  /**
   * Has the model answer a message written to a running session whose agent leaves text to
   * dispatch, which is what the person who wrote it is waiting on.
   *
   * The response is created with this worker's own credential, acting for whoever wrote the
   * message, so it reaches a conversation that belongs to them and goes to the model rather
   * than back to a worker. It carries the message's request id, so the answer lands on it.
   */
  async answer(message: InboundMessage): Promise<AgentResponse> {
    if (!message.sessionId) {
      throw new ConfigurationError("no session is holding this message; open one with sessionFor");
    }
    const acting = new Client(this.client.backend.actingFor(message.userId));
    return createResponse(acting, message.sessionId, message.text, {}, message.requestId);
  }

  /**
   * Waits for calls and messages until the router closes the connection or `signal` aborts.
   *
   * Work still being handled is waited for on the way out, because dropping a call would
   * hang up on whoever is talking. Throws `HostingRefusedError` if the router refuses the
   * tools this worker hosts: a worker nobody will call should say so rather than sit
   * connected looking healthy.
   */
  async run(signal?: AbortSignal): Promise<void> {
    if (!this.onCallHandler && !this.onMessageHandler && this.hosted.length === 0) {
      throw new ConfigurationError(
        "register a handler with onCall or onMessage, or host tools, before running",
      );
    }

    const url = await this.client.backend.socketURL("/v1/dispatch", this.waiting());
    const socket = await Socket.open(this.client.backend, url);
    this.socket = socket;

    const close = () => socket.close();
    signal?.addEventListener("abort", close, { once: true });
    // A signal that aborted while the socket was still being opened would otherwise be
    // missed, because a listener added to an aborted signal is never called.
    if (signal?.aborted) {
      close();
    }
    const reporting = setInterval(() => void this.report(), this.reportEveryMs);
    const stopping = signal ?? new AbortController().signal;

    try {
      for await (const message of socket.messages()) {
        if (message instanceof Uint8Array) {
          continue;
        }

        switch (message.type) {
          case "call":
            this.pickUp(text(message, "work_id"), callOf(message));
            break;
          case "message":
            this.write(text(message, "work_id"), messageOf(message));
            break;
          case "ready":
            this.workerId = typeof message["worker_id"] === "string" ? message["worker_id"] : "";
            this.declare();
            break;
          case "tool_call":
            this.runHosted(message, stopping);
            break;
          case "hosting":
            // The router took the offer. Nothing changes here: the calls simply start.
            break;
          case "hosting_refused":
            throw new HostingRefusedError(text(message, "agent_id"), text(message, "reason"));
          case "pong":
            this.pong?.(numberOf(message, "at"));
            break;
          default:
            break;
        }
      }
    } finally {
      clearInterval(reporting);
      signal?.removeEventListener("abort", close);
      await this.drain();
      this.answering.clear();
      socket.close();
      this.socket = undefined;
    }
  }

  /** Stops waiting. Work already being handled is still waited for by `run`. */
  stop(): void {
    this.socket?.close();
  }

  /**
   * What this worker says about itself on the way in: how much it can hold, how much it is
   * already holding, and which kinds of work it answers.
   *
   * On the handshake rather than in a frame because the router may hand this worker
   * something before it has read anything. `handles` is said even when it is nothing: a
   * worker that only hosts tools answers neither, and one handed a call it has no handler for
   * leaves a caller listening to a phone.
   */
  private waiting(): Record<string, string> {
    const kinds = [
      ...(this.onCallHandler ? ["call"] : []),
      ...(this.onMessageHandler ? ["message"] : []),
    ];
    return {
      capacity: String(this.capacity),
      active: String(this.handling),
      handles: kinds.join(","),
    };
  }

  /**
   * Starts handling one call.
   *
   * Not awaited, because reading the socket is also what delivers the next call: answering
   * one caller in line would leave the next listening to a ringing phone.
   */
  private pickUp(workId: string, call: InboundCall): void {
    const handler = this.onCallHandler;
    if (!handler) {
      this.done(workId, "this worker answers no calls");
      return;
    }
    this.handle(workId, () => handler(call));
  }

  /** Starts handling one message, on its own for the same reason a call is. */
  private write(workId: string, message: InboundMessage): void {
    const handler = this.onMessageHandler;
    if (!handler) {
      this.done(workId, "this worker answers no messages");
      return;
    }
    this.handle(workId, () => handler(message));
  }

  /** Runs one call or message, and tells the router when it is over however it went. */
  private handle(workId: string, run: () => void | Promise<void>): void {
    this.handling += 1;
    this.track(
      (async () => {
        try {
          await run();
        } catch (cause) {
          this.done(workId, cause instanceof Error ? cause.message : String(cause));
          return;
        } finally {
          this.handling -= 1;
        }
        this.done(workId);
      })(),
    );
  }

  /**
   * Tells the router one piece of work is over, which is what gives this worker its room for
   * the next back.
   *
   * What went wrong goes with it, because the router is where somebody is looking when a
   * caller says nobody picked up. It is said even for work this worker had no handler for,
   * because the room it took is held until something says it is free.
   */
  private done(workId: string, error?: string): void {
    this.tell({ type: "done", work_id: workId, ...(error === undefined ? {} : { error }) });
  }

  /** Tells the router what this worker runs, every time it says it is listening. */
  private declare(): void {
    for (const offer of this.hosted) {
      this.tell({
        type: "host_tools",
        agent_id: offer.agentId,
        tools: offer.tools.declared(),
        timeout_ms: offer.toolTimeoutMs,
      });
    }
  }

  /**
   * Answers one hosted call, on its own for the same reason a call is: an investigation
   * takes a minute, and the socket it arrived on is also what delivers the next.
   */
  private runHosted(frame: Frame, signal: AbortSignal): void {
    const id = text(frame, "id");
    const name = text(frame, "name");
    const offer = this.hosted.find((one) =>
      one.tools.declared().some((tool) => tool.name === name),
    );
    if (!offer) {
      this.tell({ type: "tool_result", id, error: `this worker does not run ${name}` });
      return;
    }

    this.track(
      (async () => {
        const result: Record<string, unknown> = { type: "tool_result", id };
        try {
          result["output"] = await offer.tools.call(name, text(frame, "arguments"), signal);
        } catch (cause) {
          result["error"] = cause instanceof Error ? cause.message : String(cause);
        }
        this.tell(result);
      })(),
    );
  }

  private track(work: Promise<void>): void {
    // Settled rather than awaited, so a handler that throws does not become an unhandled
    // rejection; whether it threw is already decided by whoever created the promise.
    const held = work.catch(() => undefined);
    this.running.add(held);
    void held.finally(() => this.running.delete(held));
  }

  /**
   * Tells the router how this process is doing, on a timer.
   *
   * The router does not use any of it to choose a worker yet. It is sent so that a policy
   * which does has numbers to read, and so an operator can see which worker is under load
   * without logging into it. Only what this runtime can honestly measure is sent: host CPU
   * and memory are not portable, and an invented figure would be read as a real one.
   */
  private async report(): Promise<void> {
    await this.measure();
    this.tell({
      type: "load",
      active_agents: this.active,
      latency_ms: this.latencyMs,
    });
  }

  /** Times a round trip to the router, taken from this side because audio crosses it. */
  private measure(): Promise<void> {
    const sent = performance.now() / 1000;
    return new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        this.pong = undefined;
        resolve();
      }, PING_TIMEOUT_MS);

      this.pong = (at) => {
        clearTimeout(timer);
        this.pong = undefined;
        this.latencyMs = (performance.now() / 1000 - at) * 1000;
        resolve();
      };
      this.tell({ type: "ping", at: sent });
    });
  }

  /**
   * Sends one frame, if the socket is still there.
   *
   * A closed socket is not an error here: every one of these is something the router would
   * like to know rather than something a call depends on.
   */
  private tell(frame: Record<string, unknown>): void {
    if (this.socket?.open) {
      this.socket.send(frame);
    }
  }

  private async drain(): Promise<void> {
    await Promise.all([...this.running]);
  }
}
