import type { Agent } from "./agent.js";
import type { BackendOptions } from "./backend.js";
import { Client } from "./client.js";
import { ConfigurationError, HostingRefusedError } from "./errors.js";
import { callOf, messageOf, type InboundCall, type InboundMessage } from "./inbound.js";
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
   * How many calls to hold at once.
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

/** What to do with a message written to an agent that is not running. */
export type MessageHandler = (message: InboundMessage) => void | Promise<void>;

export interface HostOptions {
  /** How long the router gives one call. Left out, the router's default. */
  timeoutMs?: number;
}

/** One set of functions this worker runs for every session under an agent id. */
interface Hosting {
  agentId: string;
  tools: Tools;
  timeoutMs: number;
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
   * answered. What it throws is reported to the router as a call nobody took.
   */
  onCall(handler: CallHandler): this {
    this.onCallHandler = handler;
    return this;
  }

  /** Registers what to do with a message written to an agent that is not running. */
  onMessage(handler: MessageHandler): this {
    this.onMessageHandler = handler;
    return this;
  }

  /**
   * Runs these functions for every session opened under an agent id, whoever opened it.
   *
   * A session's own functions run in the process that opened it, which is no use to a
   * conversation opened from a browser that wants to read a source tree. Hosting is the
   * other direction: the router offers these functions to each session naming the agent and
   * sends every call to a worker hosting them. Call before `run`.
   */
  host(agentId: string, tools: Tools, options: HostOptions = {}): this {
    this.hosted.push({ agentId, tools, timeoutMs: options.timeoutMs ?? 0 });
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

    const url = await this.client.backend.socketURL("/v1/dispatch", {
      capacity: String(this.capacity),
    });
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
            this.answer(callOf(message));
            break;
          case "message":
            this.write(messageOf(message));
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
   * Starts handling one call.
   *
   * Not awaited, because reading the socket is also what delivers the next call: answering
   * one caller in line would leave the next listening to a ringing phone.
   */
  private answer(call: InboundCall): void {
    const handler = this.onCallHandler;
    if (!handler) {
      return;
    }

    this.track(
      (async () => {
        try {
          await handler(call);
        } catch (cause) {
          // The router is told, so a call nobody answered shows up there rather than only
          // in this process's log.
          this.tell({
            type: "rejected",
            call_id: call.callId,
            reason: cause instanceof Error ? cause.message : String(cause),
          });
          return;
        }
        this.tell({ type: "accepted", call_id: call.callId });
      })(),
    );
  }

  /**
   * Starts handling one message, on its own for the same reason a call is.
   *
   * Nothing is reported back to the router. Accepting and rejecting are about a caller
   * waiting on a line, and there is no line here: a message nobody answered is a log line,
   * not a silence somebody is sitting in.
   */
  private write(message: InboundMessage): void {
    const handler = this.onMessageHandler;
    if (!handler) {
      return;
    }
    this.track(Promise.resolve(handler(message)).then(() => undefined));
  }

  /** Tells the router what this worker runs, every time it says it is listening. */
  private declare(): void {
    for (const offer of this.hosted) {
      this.tell({
        type: "host_tools",
        agent_id: offer.agentId,
        tools: offer.tools.declared(),
        timeout_ms: offer.timeoutMs,
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
