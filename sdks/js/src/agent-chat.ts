import { RouterError } from "./errors.js";
import { LiveReasoning } from "./reasoning.js";
import type { CreateResponseOptions, ImageSource, UIAction } from "./responses.js";
import type { Session } from "./session.js";

/** How many messages are read at a time, unless `pageSize` says otherwise. */
export const AGENT_CHAT_PAGE_SIZE = 30;

/**
 * A value to follow, in the shape of `StateStore` from `@stream-io/state-store`, which the AI
 * components' `useStateStore(store, selector)` reads. Its own here, since this package has no
 * runtime dependencies: `subscribe` calls the handler at once, and `subscribeWithSelector` calls
 * it again only when one of the values the selector picks changed.
 */
export class AgentChatStore<T extends Readonly<Record<string, unknown>>> {
  private value: T;
  private readonly handlers = new Set<(value: T, previous: T | undefined) => void>();

  constructor(value: T) {
    this.value = value;
  }

  getLatestValue(): T {
    return this.value;
  }

  next(value: T): void {
    if (Object.is(value, this.value)) {
      return;
    }
    const previous = this.value;
    this.value = value;
    for (const handler of [...this.handlers]) {
      handler(value, previous);
    }
  }

  readonly partialNext = (partial: Partial<T>): void => this.next({ ...this.value, ...partial });

  subscribe(handler: (value: T, previous: T | undefined) => void): () => void {
    handler(this.value, undefined);
    this.handlers.add(handler);
    return () => {
      this.handlers.delete(handler);
    };
  }

  readonly subscribeWithSelector = <O extends Readonly<Record<string, unknown>> | readonly unknown[]>(
    selector: (value: T) => O,
    handler: (selected: O, previous: O | undefined) => void,
  ): (() => void) => {
    let selected: O | undefined;
    return this.subscribe((value) => {
      const next = selector(value);
      if (selected !== undefined && sameSelection(selected, next)) {
        return;
      }
      const previous = selected;
      selected = next;
      handler(next, previous);
    });
  };
}

function sameSelection(a: object, b: object): boolean {
  const before = a as Record<string, unknown>;
  const after = b as Record<string, unknown>;
  return Object.keys(after).every((key) => Object.is(before[key], after[key]));
}

/**
 * Opens a conversation's session: carries on `sessionId`, or starts a new conversation without
 * one. Usually `sessions.resume(id, { tools })` or `sessions.create({ tools, … })`.
 */
export type OpenAgentSession = (sessionId: string | undefined) => Promise<Session>;

/**
 * What the agent is doing besides what its reply shows, from the session's socket:
 *
 * - `thinking`: the model is answering.
 * - `working`: skills are running, named in `skills`.
 * - `failed`: the agent couldn't write its reply. `context` is the part of the agent that failed,
 *   as the router names it (`llm`, or empty when it named none). It lasts until the agent answers
 *   again.
 */
export type AgentActivity =
  | { readonly kind: "thinking" }
  | { readonly kind: "working"; readonly skills: readonly string[] }
  | { readonly kind: "failed"; readonly context: string };

/** A Stream Chat message, as stream-chat 9 or 10 hands it over. */
export interface AgentChatMessage {
  readonly id: string;
}

export interface AgentChatState<M extends AgentChatMessage = AgentChatMessage> extends Record<string, unknown> {
  /** The channel's messages, oldest first. */
  readonly messages: readonly M[];
  /** Whether the channel holds messages older than those loaded. */
  readonly hasOlderMessages: boolean;
  /** Whether the first page of messages is in. A new conversation has none to wait for. */
  readonly isLoaded: boolean;
  /** Bumped each time more of a reply's live thinking arrives, which `reasoning` holds. */
  readonly reasoningVersion: number;
  /** The message whose live thinking arrived last, which is the one to show afresh. */
  readonly reasoningMessageId: string | undefined;
  /**
   * Why the channel couldn't be read again after the connection came back, or undefined. The
   * messages are as they were until it reads again.
   */
  readonly failure: Error | undefined;
  /** The session the conversation is held in, once something needed it. */
  readonly session: Session | undefined;
  /** What the agent is doing, or undefined. */
  readonly activity: AgentActivity | undefined;
  /**
   * The conversation's session id once something was asked in it. A new conversation joins a
   * list of conversations then, and not before.
   */
  readonly usedSessionId: string | undefined;
}

export interface AgentChatOptions {
  /** The conversation to carry on, or undefined for a new one. */
  sessionId?: string | undefined;
  /** Opens the session, the first time something needs it. */
  open: OpenAgentSession;
  /** How many messages are read at a time. */
  pageSize?: number;
}

/** What `send` takes besides the text. */
export interface AgentChatSendOptions {
  images?: ImageSource[];
  reasoningEffort?: string;
  /** What the person did in a reply's generated UI, which the text says in their words. */
  uiAction?: UIAction;
  /** The id the router answers the question once under, and stores the person's message under. */
  requestId?: string;
}

/**
 * A conversation with an agent held in writing: the messages of the channel the router keeps it
 * in, each reply's whole thinking, what the agent is doing, and asking.
 *
 * The router writes everything there. Asking goes through the session, which records the question
 * in the channel and answers it there: the reply's text, and its steps (each round of reasoning and
 * each tool call) as attachments.
 *
 * The session opens only once something needs it, such as a question, so a new conversation the
 * person only looked at is never created. A new conversation's session that was opened but never
 * asked anything is deleted when the chat closes, so it isn't left in the list.
 *
 * Make one per conversation, and close it when the conversation is left. In React, make it in an
 * effect and close it in that effect's cleanup, so StrictMode's second mount makes a fresh one:
 *
 * ```ts
 * useEffect(() => {
 *   const chat = new AgentChat({ sessionId, open });
 *   setChat(chat);
 *   return () => void chat.close();
 * }, [sessionId]);
 * ```
 *
 * It reads the channel with stream-chat 9 or 10, whichever `session.chat()` opens it on.
 */
export class AgentChat<M extends AgentChatMessage = AgentChatMessage> {
  /** Each reply's thinking in full, from its live updates. A stored step only keeps its opening. */
  readonly reasoning = new LiveReasoning();

  /** The conversation, to follow with `useStateStore` or `subscribe`. */
  readonly state: AgentChatStore<AgentChatState<M>>;

  private readonly sessionId: string | undefined;
  private readonly openSession: OpenAgentSession;
  private readonly pageSize: number;
  private reader: ChannelReader | undefined;
  private readonly stops: (() => void)[] = [];
  private stopFollowingSession: (() => void) | undefined;
  private opening: Promise<Session> | undefined;
  private asked = false;
  private closed = false;

  constructor({ sessionId, open, pageSize = AGENT_CHAT_PAGE_SIZE }: AgentChatOptions) {
    this.sessionId = sessionId;
    this.openSession = open;
    this.pageSize = pageSize;
    this.state = new AgentChatStore<AgentChatState<M>>({
      messages: [],
      hasOlderMessages: false,
      isLoaded: sessionId === undefined,
      reasoningVersion: 0,
      reasoningMessageId: undefined,
      failure: undefined,
      session: undefined,
      activity: undefined,
      usedSessionId: undefined,
    });
  }

  /** The conversation's session id: the one it was opened with, or the one asking created. */
  get id(): string | undefined {
    return this.state.getLatestValue().session?.id ?? this.sessionId;
  }

  /** Whether `close` was called. A closed chat is never used again: make a new one. */
  get isClosed(): boolean {
    return this.closed;
  }

  /**
   * Opens the session ahead of the first question, such as while the person writes it: opening
   * takes seconds. A new conversation's session opened this way and never asked anything is
   * deleted on `close`.
   */
  async prepare(): Promise<void> {
    await this.session();
  }

  /**
   * The conversation's session, opened the first time it is needed. A new conversation is created
   * then, and its channel followed. A session whose socket dropped is carried on in a new one.
   * Rejects when it can't open, to be asked again.
   */
  session(): Promise<Session> {
    if (this.closed) {
      return Promise.reject(new Error("The conversation was closed"));
    }
    if (this.dropped()) {
      this.opening = undefined;
    }
    this.opening ??= this.open().catch((error: unknown) => {
      this.opening = undefined;
      throw error;
    });
    return this.opening;
  }

  /**
   * Follows a channel the conversation is kept in before its session opens, such as one from a
   * list of conversations, so its messages show while the session comes back. The session's own
   * channel is followed when it opens, if none is.
   */
  async follow(channel: unknown): Promise<void> {
    if (this.reader || this.closed) {
      return;
    }
    const reader = readerFor(channel, this.pageSize, {
      messages: (messages, hasOlderMessages) =>
        this.state.partialNext({
          messages: messages as readonly M[],
          hasOlderMessages,
          failure: undefined,
          // Messages this client read before show at once, while the latest page loads.
          ...(messages.length > 0 ? { isLoaded: true } : {}),
        }),
      updated: (message) => {
        this.reasoning.read(message);
        this.state.partialNext({
          reasoningVersion: this.state.getLatestValue().reasoningVersion + 1,
          reasoningMessageId: message.id,
        });
      },
      failed: (error) => this.state.partialNext({ failure: error }),
    });
    this.reader = reader;
    try {
      await reader.start();
    } catch (error) {
      reader.stop();
      this.reader = undefined;
      throw error;
    }
    if (!this.closed) {
      this.state.partialNext({ isLoaded: true });
    }
  }

  /**
   * Asks the agent. Resolves once the router has the question: the question and its reply arrive
   * in `messages`. Rejects when it couldn't be asked. A new conversation whose first question
   * never reached the router isn't left in the list.
   */
  async send(text: string, options: AgentChatSendOptions = {}): Promise<void> {
    // A session that can't open was never asked anything, and stays unasked.
    const session = await this.session();
    this.asked = true;
    if (this.state.getLatestValue().usedSessionId !== session.id) {
      this.state.partialNext({ usedSessionId: session.id });
    }
    try {
      await session.responses.create(text, responseOptions(options));
    } catch (error) {
      // A question that may have reached the router keeps its session: the router may be
      // answering it, and asking again under the same request id is answered once, there.
      if (!mayHaveReachedRouter(error)) {
        await this.discardUnusedSession();
      }
      throw error;
    }
  }

  /** Stops the reply in flight. What was written so far stays. */
  async stop(): Promise<void> {
    // A new conversation that was never opened has nothing to stop.
    if (this.sessionId === undefined && !this.opening) {
      return;
    }
    const session = await this.session();
    session.interrupt();
  }

  /** Loads the messages before the oldest one loaded. Rejects when they couldn't load. */
  async loadOlderMessages(): Promise<void> {
    if (!this.state.getLatestValue().hasOlderMessages) {
      return;
    }
    await this.reader?.loadOlder(this.state.getLatestValue().messages);
  }

  /**
   * Stops following the channel and lets the session go: deleted when it was a new conversation's
   * that nothing was asked in, closed otherwise. Safe to call twice.
   */
  async close(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.reader?.stop();
    this.stopFollowingSession?.();
    for (const stop of this.stops.splice(0)) {
      stop();
    }
    const opened = await this.opening?.catch(() => undefined);
    if (!opened) {
      return;
    }
    const unused = this.sessionId === undefined && !this.asked;
    await (unused ? opened.delete() : opened.close()).catch(() => undefined);
  }

  /**
   * Deletes the session a new conversation was given when the question that created it failed
   * before anything was written, so the router doesn't list an empty conversation. The next
   * question creates another.
   */
  private async discardUnusedSession(): Promise<void> {
    const { messages, session } = this.state.getLatestValue();
    if (this.sessionId !== undefined || !session || messages.length > 0) {
      return;
    }
    this.asked = false;
    this.opening = undefined;
    this.stopFollowingSession?.();
    this.stopFollowingSession = undefined;
    // The next question creates another conversation, in a channel of its own.
    this.reader?.stop();
    this.reader = undefined;
    this.state.partialNext({
      session: undefined,
      usedSessionId: undefined,
      activity: undefined,
      messages: [],
      hasOlderMessages: false,
    });
    await session.delete().catch(() => undefined);
  }

  /** Whether the open session's socket has stopped carrying the conversation. */
  private dropped(): boolean {
    const { session } = this.state.getLatestValue();
    return session !== undefined && !session.live;
  }

  private async open(): Promise<Session> {
    const opened = this.id;
    const session = await this.openSession(opened);
    if (this.closed) {
      // The person moved on while it opened. A conversation created for them was never used.
      await (opened === undefined ? session.delete() : session.close()).catch(() => undefined);
      throw new Error("The conversation was closed");
    }
    if (!this.reader) {
      try {
        const { channel } = await session.chat();
        await this.follow(channel);
      } catch (error) {
        // A session whose conversation can't be read is no use; let it go, to open again.
        await (opened === undefined ? session.delete() : session.close()).catch(() => undefined);
        throw error;
      }
    }
    if (this.closed) {
      // Closed while its channel opened: nothing more is asked of it.
      await (opened === undefined ? session.delete() : session.close()).catch(() => undefined);
      throw new Error("The conversation was closed");
    }
    this.followSession(session);
    this.state.partialNext({ session });
    return session;
  }

  /** Follows what the agent is doing, from the session's socket. */
  private followSession(session: Session): void {
    this.stopFollowingSession?.();
    // A failure reported before this session was followed belongs to an earlier one.
    const since = session.conversation.get().failureAt;
    const update = () => {
      const activity = activityOf(session, since);
      if (!sameActivity(activity, this.state.getLatestValue().activity)) {
        this.state.partialNext({ activity });
      }
    };
    const stopState = session.state.subscribe(update);
    const stopConversation = session.conversation.subscribe(update);
    this.stopFollowingSession = () => {
      stopState();
      stopConversation();
    };
    update();
  }
}

/** The failures of a written conversation's reply. A call's speech failures are not these. */
const REPLY_FAILURES = new Set(["llm", ""]);

function activityOf(session: Session, since: number): AgentActivity | undefined {
  const { failure, failureAt, failureContext } = session.conversation.get();
  if (failure && failureAt > since && REPLY_FAILURES.has(failureContext)) {
    return { kind: "failed", context: failureContext };
  }
  const state = session.state.get();
  if (state.kind === "responding") {
    return { kind: "thinking" };
  }
  if (state.kind === "working") {
    return { kind: "working", skills: state.skills };
  }
  return undefined;
}

function sameActivity(a: AgentActivity | undefined, b: AgentActivity | undefined): boolean {
  if (a === undefined || b === undefined || a.kind !== b.kind) {
    return a === b;
  }
  if (a.kind === "working" && b.kind === "working") {
    return a.skills.length === b.skills.length && a.skills.every((skill, index) => skill === b.skills[index]);
  }
  if (a.kind === "failed" && b.kind === "failed") {
    return a.context === b.context;
  }
  return true;
}

function responseOptions({ images, reasoningEffort, uiAction, requestId }: AgentChatSendOptions): CreateResponseOptions {
  return {
    ...(images && images.length > 0 ? { images } : {}),
    ...(reasoningEffort ? { reasoningEffort } : {}),
    ...(uiAction ? { uiAction } : {}),
    ...(requestId ? { requestId } : {}),
  };
}

// Reading a channel, in stream-chat 9 or 10.

interface ReaderEvents {
  /** The channel's messages, oldest first, each time they change. */
  messages(messages: readonly AgentChatMessage[], hasOlderMessages: boolean): void;
  /** A live update of a message, which may carry more of a reply's thinking. */
  updated(message: AgentChatMessage): void;
  /** Reading the channel again failed, outside any call that could reject with it. */
  failed(error: Error): void;
}

interface ChannelReader {
  start(): Promise<void>;
  loadOlder(loaded: readonly AgentChatMessage[]): Promise<void>;
  stop(): void;
}

interface Subscription {
  unsubscribe(): void;
}

interface ChatEvent {
  readonly type: string;
  readonly message?: AgentChatMessage;
  readonly online?: boolean;
}

/** What this module reads of a stream-chat 10 channel. */
interface ChannelV10 {
  readonly messagePaginator: {
    pageSize: number;
    readonly hasMoreTail: boolean;
    readonly state: {
      subscribeWithSelector<O extends Readonly<Record<string, unknown>>>(
        selector: (value: { items?: readonly AgentChatMessage[] | undefined; hasMoreTail: boolean }) => O,
        handler: (value: O) => void,
      ): () => void;
    };
    toTail(): Promise<unknown>;
  };
  on(type: string, handler: (event: ChatEvent) => void): Subscription;
  activate(): () => void;
  watch(options: { messages: { limit: number } }): Promise<unknown>;
}

/** What this module reads of a stream-chat 9 channel. */
interface ChannelV9 {
  readonly state: { readonly messages: readonly AgentChatMessage[] };
  on(handler: (event: ChatEvent) => void): Subscription;
  query(options: { messages: { id_lt: string; limit: number } }): Promise<{ messages: readonly unknown[] }>;
  watch(options: { messages: { limit: number } }): Promise<{ messages?: readonly unknown[] }>;
  getClient(): { on(type: string, handler: (event: ChatEvent) => void): Subscription };
}

function readerFor(channel: unknown, pageSize: number, events: ReaderEvents): ChannelReader {
  if (typeof channel !== "object" || channel === null) {
    throw new TypeError("Expected a Stream Chat channel");
  }
  return "messagePaginator" in channel
    ? readerV10(channel as ChannelV10, pageSize, events)
    : readerV9(channel as ChannelV9, pageSize, events);
}

/** stream-chat 10: the messages are the channel's paginator's, which keeps them current. */
function readerV10(channel: ChannelV10, pageSize: number, events: ReaderEvents): ChannelReader {
  const stops: (() => void)[] = [];
  return {
    async start() {
      const paginator = channel.messagePaginator;
      paginator.pageSize = pageSize;
      stops.push(
        paginator.state.subscribeWithSelector(
          ({ items, hasMoreTail }) => ({ items, hasMoreTail }),
          ({ items, hasMoreTail }) => events.messages([...(items ?? [])], hasMoreTail),
        ),
      );
      // Every update rather than the list kept for us: each live update carries its own window of
      // a reply's thinking, and the list folds them into one.
      const updates = channel.on("message.updated", (event) => {
        if (event.message) {
          events.updated(event.message);
        }
      });
      stops.push(() => updates.unsubscribe());
      // On screen, so it is loaded again after a reconnect: a channel that is only watched is not.
      stops.push(channel.activate());
      await channel.watch({ messages: { limit: pageSize } });
    },
    async loadOlder() {
      if (channel.messagePaginator.hasMoreTail) {
        await channel.messagePaginator.toTail();
      }
    },
    stop() {
      for (const stop of stops.splice(0)) {
        stop();
      }
    },
  };
}

/** stream-chat 9: the messages are `channel.state.messages`, read again on each change. */
function readerV9(channel: ChannelV9, pageSize: number, events: ReaderEvents): ChannelReader {
  const stops: (() => void)[] = [];
  let hasOlderMessages = false;
  let stopped = false;
  const publish = () => events.messages([...channel.state.messages], hasOlderMessages);
  const watchLatest = async () => {
    const watched = await channel.watch({ messages: { limit: pageSize } });
    if (stopped) {
      return;
    }
    hasOlderMessages = (watched.messages?.length ?? 0) >= pageSize;
    publish();
  };
  return {
    async start() {
      const subscription = channel.on((event) => {
        if (event.type === "message.updated" && event.message) {
          events.updated(event.message);
        }
        if (event.type === "message.new" || event.type === "message.updated" || event.type === "message.deleted") {
          publish();
        }
      });
      stops.push(() => subscription.unsubscribe());
      const reconnected = channel.getClient().on("connection.changed", (event) => {
        if (event.online) {
          watchLatest().catch((error: unknown) => {
            if (!stopped) {
              events.failed(error instanceof Error ? error : new Error(String(error)));
            }
          });
        }
      });
      stops.push(() => reconnected.unsubscribe());
      // A channel this client read before shows what it had at once, while the latest page loads.
      if (channel.state.messages.length > 0) {
        publish();
      }
      await watchLatest();
    },
    async loadOlder(loaded) {
      const oldest = loaded[0]?.id;
      if (!oldest) {
        return;
      }
      const page = await channel.query({ messages: { id_lt: oldest, limit: pageSize } });
      if (stopped) {
        return;
      }
      hasOlderMessages = page.messages.length >= pageSize;
      publish();
    },
    stop() {
      stopped = true;
      for (const stop of stops.splice(0)) {
        stop();
      }
    },
  };
}

/**
 * Whether a failed question may have reached the router: it never got an answer (status 0), or
 * the router failed while handling it (5xx). A refusal (4xx) wrote nothing, and nor did a
 * failure before anything was sent.
 */
function mayHaveReachedRouter(error: unknown): boolean {
  return error instanceof RouterError && (error.status === 0 || error.status >= 500);
}
