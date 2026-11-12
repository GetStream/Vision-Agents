import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  AgentChat,
  AgentChatStore,
  agentChatTransport,
  type AgentTransportActivity,
  type AgentTransportFeed,
  type AgentTransportRequest,
} from "../src/chat.js";
import { RouterError, type CreateResponseOptions, type Session } from "../src/index.js";

type Message = { id: string; text?: string; reasoning?: unknown };

/** A `get`/`subscribe` store, as `session.state` and `session.conversation` are. */
function store<T extends object>(initial: T) {
  let value = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => value,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => void listeners.delete(listener);
    },
    set(patch: Partial<T>) {
      value = { ...value, ...patch };
      for (const listener of [...listeners]) listener();
    },
  };
}

class FakeSession {
  live = true;
  deleted = 0;
  closed = 0;
  interrupts = 0;
  failCreate: Error | undefined;
  failChat: Error | undefined;
  readonly asked: { text: string; options: CreateResponseOptions }[] = [];
  readonly state = store<{ kind: string; skills: readonly string[] }>({ kind: "idle", skills: [] });
  readonly conversation = store({ failure: "", failureContext: "", failureAt: 0 });
  readonly responses = {
    create: async (text: string, options: CreateResponseOptions = {}) => {
      if (this.failCreate) throw this.failCreate;
      this.asked.push({ text, options });
      return {};
    },
  };

  constructor(
    readonly id: string,
    private readonly channel: unknown,
  ) {}

  async chat() {
    if (this.failChat) throw this.failChat;
    return { client: {}, channel: this.channel };
  }
  interrupt() {
    this.interrupts += 1;
  }
  async close() {
    this.closed += 1;
  }
  async delete() {
    this.deleted += 1;
  }
}

/** A stream-chat 9 channel: `state.messages`, one `on` for every event, `query` for older pages. */
class ChannelV9 {
  readonly state = { messages: [] as Message[] };
  /** What the next `watch` loads. */
  latest: Message[] = [];
  /** What `query` answers with, or the error it fails with. */
  older: Message[] | Error = [];
  readonly queried: { id_lt: string; limit: number }[] = [];
  private readonly handlers = new Set<(event: { type: string; message?: Message }) => void>();

  on(handler: (event: { type: string; message?: Message }) => void) {
    this.handlers.add(handler);
    return { unsubscribe: () => void this.handlers.delete(handler) };
  }
  emit(event: { type: string; message?: Message }) {
    if (event.type === "message.new" && event.message) this.state.messages = [...this.state.messages, event.message];
    if (event.type === "message.updated" && event.message) {
      const updated = event.message;
      this.state.messages = this.state.messages.map((message) => (message.id === updated.id ? updated : message));
    }
    for (const handler of [...this.handlers]) handler(event);
  }
  get listeners() {
    return this.handlers.size;
  }
  /** What the next `watch` fails with, if anything. */
  watchError: Error | undefined;
  private readonly clientHandlers = new Set<(event: { type: string; online?: boolean }) => void>();

  async watch(options: { messages: { limit: number } }) {
    if (this.watchError) throw this.watchError;
    this.state.messages = [...this.latest];
    return { messages: this.latest.slice(-options.messages.limit) };
  }
  async query(options: { messages: { id_lt: string; limit: number } }) {
    this.queried.push(options.messages);
    if (this.older instanceof Error) throw this.older;
    this.state.messages = [...this.older, ...this.state.messages];
    return { messages: this.older };
  }
  getClient() {
    return {
      on: (_type: string, handler: (event: { type: string; online?: boolean }) => void) => {
        this.clientHandlers.add(handler);
        return { unsubscribe: () => void this.clientHandlers.delete(handler) };
      },
    };
  }
  /** The connection coming back. */
  reconnect() {
    for (const handler of [...this.clientHandlers]) handler({ type: "connection.changed", online: true });
  }
}

/** A stream-chat 10 channel: the messages are its paginator's. */
class ChannelV10 {
  activated = 0;
  deactivated = 0;
  tails = 0;
  latest: Message[] = [];
  private readonly handlers = new Map<string, Set<(event: { type: string; message?: Message }) => void>>();
  readonly messagePaginator = {
    pageSize: 0,
    state: new AgentChatStore<{ items: Message[] | undefined; hasMoreTail: boolean }>({ items: undefined, hasMoreTail: false }),
    get hasMoreTail() {
      return this.state.getLatestValue().hasMoreTail;
    },
    toTail: async () => {
      this.tails += 1;
      const { items = [] } = this.messagePaginator.state.getLatestValue();
      this.messagePaginator.state.next({ items: [{ id: "older" }, ...items], hasMoreTail: false });
    },
  };

  on(type: string, handler: (event: { type: string; message?: Message }) => void) {
    const set = this.handlers.get(type) ?? new Set();
    set.add(handler);
    this.handlers.set(type, set);
    return { unsubscribe: () => void set.delete(handler) };
  }
  emit(event: { type: string; message?: Message }) {
    for (const handler of [...(this.handlers.get(event.type) ?? [])]) handler(event);
  }
  activate() {
    this.activated += 1;
    return () => void (this.deactivated += 1);
  }
  async watch() {
    this.messagePaginator.state.next({ items: [...this.latest], hasMoreTail: true });
    return this;
  }
}

/** Opens sessions on `channel`, numbering new ones, and records what it was asked to open. */
function opener(channel: unknown) {
  const opened: (string | undefined)[] = [];
  const sessions: FakeSession[] = [];
  const open = async (sessionId: string | undefined) => {
    opened.push(sessionId);
    const session = new FakeSession(sessionId ?? `new-${sessions.length + 1}`, channel);
    sessions.push(session);
    return session as unknown as Session;
  };
  return { open, opened, sessions };
}

describe("AgentChatStore", () => {
  it("tells a selector's handler at once, then only when what it picks changed", () => {
    const store = new AgentChatStore({ count: 0, label: "a" });
    const seen: number[] = [];
    store.subscribeWithSelector(({ count }) => ({ count }), ({ count }) => seen.push(count));

    store.partialNext({ label: "b" });
    store.partialNext({ count: 1 });
    store.partialNext({ count: 1 });

    assert.deepEqual(seen, [0, 1]);
  });
});

describe("AgentChat", () => {
  it("opens no session for a new conversation until it is asked something", async () => {
    const channel = new ChannelV9();
    const { open, opened, sessions } = opener(channel);
    const chat = new AgentChat<Message>({ open });

    assert.equal(chat.state.getLatestValue().isLoaded, true);
    assert.deepEqual(opened, []);

    channel.latest = [{ id: "q1", text: "Hi" }];
    await chat.send("Hi", { requestId: "q1", reasoningEffort: "high" });

    assert.deepEqual(opened, [undefined]);
    assert.deepEqual(sessions[0]?.asked, [{ text: "Hi", options: { reasoningEffort: "high", requestId: "q1" } }]);
    const state = chat.state.getLatestValue();
    assert.equal(state.usedSessionId, "new-1");
    assert.deepEqual(
      state.messages.map((message) => message.id),
      ["q1"],
    );
  });

  it("carries on a listed conversation, loading its latest page", async () => {
    const channel = new ChannelV9();
    channel.latest = [{ id: "a" }, { id: "b" }];
    const { open, opened } = opener(channel);
    const chat = new AgentChat<Message>({ sessionId: "s1", open, pageSize: 2 });

    assert.equal(chat.state.getLatestValue().isLoaded, false);
    await chat.prepare();

    assert.deepEqual(opened, ["s1"]);
    const state = chat.state.getLatestValue();
    assert.equal(state.isLoaded, true);
    assert.equal(state.hasOlderMessages, true);
    assert.equal(state.usedSessionId, undefined);
  });

  it("opens one session for everything that asks for it at once", async () => {
    const { open, opened } = opener(new ChannelV9());
    const chat = new AgentChat<Message>({ open });

    await Promise.all([chat.prepare(), chat.send("one"), chat.send("two")]);

    assert.equal(opened.length, 1);
  });

  it("deletes a new conversation's session that was opened and never asked anything", async () => {
    const { open, sessions } = opener(new ChannelV9());
    const chat = new AgentChat<Message>({ open });

    await chat.prepare();
    await chat.close();
    await chat.close();

    assert.equal(sessions[0]?.deleted, 1);
    assert.equal(sessions[0]?.closed, 0);
  });

  it("closes, rather than deletes, a session that was asked something or carried on", async () => {
    const asked = opener(new ChannelV9());
    const chat = new AgentChat<Message>({ open: asked.open });
    await chat.send("Hi");
    await chat.close();
    assert.deepEqual([asked.sessions[0]?.closed, asked.sessions[0]?.deleted], [1, 0]);

    const listed = opener(new ChannelV9());
    const carried = new AgentChat<Message>({ sessionId: "s1", open: listed.open });
    await carried.prepare();
    await carried.close();
    assert.deepEqual([listed.sessions[0]?.closed, listed.sessions[0]?.deleted], [1, 0]);
  });

  it("closing a chat that never opened anything does nothing, and it stays closed", async () => {
    const { open, opened } = opener(new ChannelV9());
    const chat = new AgentChat<Message>({ open });

    await chat.close();

    assert.equal(chat.isClosed, true);
    await assert.rejects(chat.send("Hi"), /closed/);
    assert.deepEqual(opened, []);
  });

  it("deletes a new conversation's session when the router refuses its first question, and reads the next one's channel", async () => {
    // Each new conversation is kept in a channel of its own.
    const channels = [new ChannelV9(), new ChannelV9()];
    const sessions: FakeSession[] = [];
    const chat = new AgentChat<Message>({
      open: async () => {
        const channel = channels[sessions.length];
        const session = new FakeSession(`new-${sessions.length + 1}`, channel);
        if (sessions.length === 0) session.failCreate = new RouterError(400, "createResponse", "refused");
        sessions.push(session);
        return session as unknown as Session;
      },
    });

    await assert.rejects(chat.send("Hi"), /refused/);
    assert.equal(sessions[0]?.deleted, 1);
    assert.equal(chat.state.getLatestValue().usedSessionId, undefined);
    assert.equal(chat.state.getLatestValue().session, undefined);

    channels[1]!.latest = [{ id: "q2", text: "Hi again" }];
    await chat.send("Hi again");
    assert.equal(chat.state.getLatestValue().usedSessionId, "new-2");
    assert.deepEqual(
      chat.state.getLatestValue().messages.map((message) => message.id),
      ["q2"],
    );
    channels[1]!.emit({ type: "message.new", message: { id: "r2" } });
    assert.deepEqual(
      chat.state.getLatestValue().messages.map((message) => message.id),
      ["q2", "r2"],
    );
  });

  it("keeps the session of a question that may have reached the router, and asks again there", async () => {
    const { open, opened, sessions } = opener(new ChannelV9());
    const chat = new AgentChat<Message>({ open });
    await chat.prepare();
    sessions[0]!.failCreate = new RouterError(0, "createResponse", "network");

    await assert.rejects(chat.send("Hi", { requestId: "q1" }), /network/);
    assert.equal(sessions[0]?.deleted, 0);
    assert.equal(chat.state.getLatestValue().usedSessionId, "new-1");

    sessions[0]!.failCreate = undefined;
    await chat.send("Hi", { requestId: "q1" });
    assert.deepEqual(opened, [undefined]);
    assert.equal(sessions[0]?.asked.at(-1)?.options.requestId, "q1");
  });

  it("deletes a later unused session when an earlier question never opened one", async () => {
    let attempts = 0;
    const { open, sessions } = opener(new ChannelV9());
    const chat = new AgentChat<Message>({
      open: async (id) => {
        attempts += 1;
        if (attempts === 1) throw new Error("offline");
        return open(id);
      },
    });

    await assert.rejects(chat.send("Hi"), /offline/);
    await chat.prepare();
    await chat.close();

    assert.deepEqual([sessions[0]?.deleted, sessions[0]?.closed], [1, 0]);
  });

  it("asks nothing of a session whose channel opened after the chat was closed", async () => {
    const channel = new ChannelV9();
    let release: () => void = () => undefined;
    const sessions: FakeSession[] = [];
    const chat = new AgentChat<Message>({
      open: async () => {
        const session = new FakeSession("slow", channel);
        session.chat = async () => {
          await new Promise<void>((resolve) => (release = resolve));
          return { client: {}, channel };
        };
        sessions.push(session);
        return session as unknown as Session;
      },
    });

    const sending = chat.send("Hi");
    await new Promise((resolve) => setImmediate(resolve));
    const closing = chat.close();
    release();

    await assert.rejects(sending, /closed/);
    await closing;
    assert.equal(sessions[0]?.asked.length, 0);
    assert.equal(sessions[0]?.deleted, 1);
    assert.equal(chat.state.getLatestValue().session, undefined);
  });

  it("lets go of a session that opened after the chat was closed", async () => {
    const channel = new ChannelV9();
    let release: () => void = () => undefined;
    const sessions: FakeSession[] = [];
    const chat = new AgentChat<Message>({
      open: async () => {
        await new Promise<void>((resolve) => (release = resolve));
        const session = new FakeSession("late", channel);
        sessions.push(session);
        return session as unknown as Session;
      },
    });

    const preparing = chat.prepare();
    const closing = chat.close();
    release();

    await assert.rejects(preparing, /closed/);
    await closing;
    assert.equal(sessions[0]?.deleted, 1);
  });

  it("lets go of a new session whose channel can't be read", async () => {
    const { sessions, open } = opener(new ChannelV9());
    const chat = new AgentChat<Message>({
      open: async (id) => {
        const session = await open(id);
        (session as unknown as FakeSession).failChat = new Error("no chat");
        return session;
      },
    });

    await assert.rejects(chat.prepare(), /no chat/);
    assert.equal(sessions[0]?.deleted, 1);
  });

  it("counts a conversation used without asking, such as talked to, and keeps it on close", async () => {
    const { open, sessions } = opener(new ChannelV9());
    const chat = new AgentChat<Message>({ open });

    const session = await chat.use();
    await chat.close();

    assert.equal(session.id, "new-1");
    assert.equal(chat.state.getLatestValue().usedSessionId, "new-1");
    assert.deepEqual([sessions[0]?.closed, sessions[0]?.deleted], [1, 0]);
  });

  it("stops a reply through the session, and has nothing to stop before one opened", async () => {
    const { open, opened, sessions } = opener(new ChannelV9());
    const chat = new AgentChat<Message>({ open });

    await chat.stop();
    assert.deepEqual(opened, []);

    await chat.send("Hi");
    await chat.stop();
    assert.equal(sessions[0]?.interrupts, 1);
  });

  it("opens the session again once its socket dropped", async () => {
    const { open, opened, sessions } = opener(new ChannelV9());
    const chat = new AgentChat<Message>({ sessionId: "s1", open });

    await chat.prepare();
    if (sessions[0]) sessions[0].live = false;
    await chat.send("Still there?");

    assert.deepEqual(opened, ["s1", "s1"]);
    assert.equal(sessions[1]?.asked.length, 1);
  });

  describe("activity", () => {
    async function opened() {
      const { open, sessions } = opener(new ChannelV9());
      const chat = new AgentChat<Message>({ sessionId: "s1", open });
      await chat.prepare();
      const session = sessions[0];
      assert.ok(session);
      return { chat, session };
    }

    it("says the agent is thinking while it responds, and which skills it runs", async () => {
      const { chat, session } = await opened();

      session.state.set({ kind: "responding" });
      assert.deepEqual(chat.state.getLatestValue().activity, { kind: "thinking" });

      session.state.set({ kind: "working", skills: ["trip_planning"] });
      assert.deepEqual(chat.state.getLatestValue().activity, { kind: "working", skills: ["trip_planning"] });

      session.state.set({ kind: "idle", skills: [] });
      assert.equal(chat.state.getLatestValue().activity, undefined);
    });

    it("says the reply failed when the model did, until the agent answers again", async () => {
      const { chat, session } = await opened();

      session.conversation.set({ failure: "402 Payment Required", failureContext: "llm", failureAt: 10 });
      assert.deepEqual(chat.state.getLatestValue().activity, { kind: "failed", context: "llm" });

      session.conversation.set({ failure: "", failureContext: "", failureAt: 0 });
      session.state.set({ kind: "responding" });
      assert.deepEqual(chat.state.getLatestValue().activity, { kind: "thinking" });
    });

    it("leaves out a call's speech failures and those from before it followed the session", async () => {
      const { open, sessions } = opener(new ChannelV9());
      const chat = new AgentChat<Message>({
        sessionId: "s1",
        open: async (id) => {
          const session = await open(id);
          (session as unknown as FakeSession).conversation.set({ failure: "old", failureContext: "llm", failureAt: 5 });
          return session;
        },
      });
      await chat.prepare();
      assert.equal(chat.state.getLatestValue().activity, undefined);

      sessions[0]?.conversation.set({ failure: "can't hear", failureContext: "stt", failureAt: 20 });
      assert.equal(chat.state.getLatestValue().activity, undefined);
    });
  });

  describe("on stream-chat 9", () => {
    it("says when the channel can't be read again after the connection came back, until it can", async () => {
      const channel = new ChannelV9();
      const { open } = opener(channel);
      const chat = new AgentChat<Message>({ sessionId: "s1", open });
      await chat.prepare();

      channel.watchError = new Error("token expired");
      channel.reconnect();
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(chat.state.getLatestValue().failure?.message, "token expired");

      channel.watchError = undefined;
      channel.reconnect();
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(chat.state.getLatestValue().failure, undefined);
    });

    it("loads older messages before the oldest one, and rejects when they can't load", async () => {
      const channel = new ChannelV9();
      channel.latest = [{ id: "c" }, { id: "d" }];
      const { open } = opener(channel);
      const chat = new AgentChat<Message>({ sessionId: "s1", open, pageSize: 2 });
      await chat.prepare();

      channel.older = [{ id: "a" }, { id: "b" }];
      await chat.loadOlderMessages();
      assert.deepEqual(channel.queried, [{ id_lt: "c", limit: 2 }]);
      assert.deepEqual(
        chat.state.getLatestValue().messages.map((message) => message.id),
        ["a", "b", "c", "d"],
      );

      channel.older = new Error("offline");
      await assert.rejects(chat.loadOlderMessages(), /offline/);
    });

    it("follows new messages and reads each live update's thinking", async () => {
      const channel = new ChannelV9();
      const { open } = opener(channel);
      const chat = new AgentChat<Message>({ sessionId: "s1", open });
      await chat.prepare();

      channel.emit({ type: "message.new", message: { id: "r1", text: "" } });
      channel.emit({
        type: "message.updated",
        message: { id: "r1", text: "", reasoning: { id: "s1", offset: 0, text: "Thinking it over" } },
      });

      const state = chat.state.getLatestValue();
      assert.deepEqual(
        state.messages.map((message) => message.id),
        ["r1"],
      );
      assert.equal(state.reasoningVersion, 1);
      assert.equal(chat.reasoning.get("r1", "s1"), "Thinking it over");

      await chat.close();
      assert.equal(channel.listeners, 0);
    });
  });

  describe("on stream-chat 10", () => {
    it("reads the paginator's messages, loads older pages through it, and lets the channel go", async () => {
      const channel = new ChannelV10();
      channel.latest = [{ id: "a" }];
      const { open } = opener(channel);
      const chat = new AgentChat<Message>({ sessionId: "s1", open, pageSize: 20 });
      await chat.prepare();

      assert.equal(channel.messagePaginator.pageSize, 20);
      assert.equal(channel.activated, 1);
      assert.equal(chat.state.getLatestValue().hasOlderMessages, true);

      await chat.loadOlderMessages();
      assert.equal(channel.tails, 1);
      assert.deepEqual(
        chat.state.getLatestValue().messages.map((message) => message.id),
        ["older", "a"],
      );

      channel.emit({ type: "message.updated", message: { id: "a", reasoning: { id: "s1", offset: 0, text: "Hmm" } } });
      assert.equal(chat.reasoning.get("a", "s1"), "Hmm");

      await chat.close();
      assert.equal(channel.deactivated, 1);
    });
  });
});

describe("agentChatTransport", () => {
  class Feed implements AgentTransportFeed<string> {
    isConnected = true;
    messages: (readonly string[])[] = [];
    older: boolean[] = [];
    activities: (AgentTransportActivity | undefined)[] = [];
    updateMessages(messages: readonly string[]) {
      this.messages.push(messages);
    }
    updateHasOlderMessages(hasOlderMessages: boolean) {
      this.older.push(hasOlderMessages);
    }
    updateActivity(activity: AgentTransportActivity | undefined) {
      this.activities.push(activity);
    }
  }

  async function connected(options: Partial<Parameters<typeof agentChatTransport<Message, string>>[1]> = {}) {
    const channel = new ChannelV9();
    channel.latest = [{ id: "q1" }];
    const { open, sessions } = opener(channel);
    const chat = new AgentChat<Message>({ sessionId: "s1", open });
    await chat.prepare();
    const transport = agentChatTransport(chat, {
      adapt: (message, reasoningText) => `${message.id}:${reasoningText(message.id, "s1") ?? ""}`,
      ...options,
    });
    const feed = new Feed();
    const disconnect = transport.connect(feed);
    const session = sessions[0];
    assert.ok(session);
    return { chat, channel, session, transport, feed, disconnect };
  }

  const request = (patch: Partial<AgentTransportRequest> = {}): AgentTransportRequest => ({
    kind: "message",
    messageId: "m1",
    text: "Plan a weekend in Ohrid",
    attachments: [],
    ...patch,
  });

  it("feeds the conversation its messages, and again as a reply's thinking grows", async () => {
    const { channel, feed } = await connected();
    assert.deepEqual(feed.messages.at(-1), ["q1:"]);

    channel.emit({ type: "message.updated", message: { id: "q1", reasoning: { id: "s1", offset: 0, text: "Hm" } } });
    assert.deepEqual(feed.messages.at(-1), ["q1:Hm"]);
  });

  it("says what the agent is doing in the AI indicator's states", async () => {
    const { session, feed } = await connected();

    session.state.set({ kind: "responding" });
    session.state.set({ kind: "working", skills: ["trip_planning", "web_search"] });
    session.state.set({ kind: "idle", skills: [] });
    session.conversation.set({ failure: "raw provider error", failureContext: "llm", failureAt: 9 });

    assert.deepEqual(feed.activities.slice(1), [
      { state: "AI_STATE_THINKING" },
      { state: "AI_STATE_EXTERNAL_SOURCES", text: "Working on trip planning and web search" },
      undefined,
      // No text: the kit shows its own translated message, never the provider's.
      { state: "AI_STATE_ERROR" },
    ]);
  });

  it("words a failure the app's way when it asks to", async () => {
    const { session, feed } = await connected({ failedText: (context) => `Juno couldn't answer (${context})` });

    session.conversation.set({ failure: "raw", failureContext: "llm", failureAt: 9 });

    assert.deepEqual(feed.activities.at(-1), { state: "AI_STATE_ERROR", text: "Juno couldn't answer (llm)" });
  });

  it("asks under the person's message id, with the composer option and the UI action", async () => {
    const { session, transport } = await connected({
      reasoningEffort: (option) => (option?.id === "think_harder" ? "high" : undefined),
      images: async (attachments) => attachments.map((each) => ({ url: String(each) })),
    });
    const uiAction = { message_id: "r1", format: "a2ui", name: "pick", payload: {} };

    await transport.send(
      request({ attachments: ["data:image/png;base64,AA"], chatOption: { id: "think_harder" }, custom: { uiAction } }),
    );

    assert.deepEqual(session.asked.at(-1), {
      text: "Plan a weekend in Ohrid",
      options: { images: [{ url: "data:image/png;base64,AA" }], reasoningEffort: "high", uiAction, requestId: "m1" },
    });
  });

  it("makes afresh only the messages an update touched, and the reply whose thinking grew", async () => {
    const channel = new ChannelV9();
    channel.latest = [{ id: "q1" }, { id: "r1" }];
    const { open } = opener(channel);
    const chat = new AgentChat<Message>({ sessionId: "s1", open });
    await chat.prepare();
    const transport = agentChatTransport(chat, { adapt: (message) => ({ id: message.id }) });
    const fed: (readonly { id: string }[])[] = [];
    transport.connect({
      isConnected: true,
      updateMessages: (messages) => void fed.push(messages),
      updateHasOlderMessages: () => undefined,
      updateActivity: () => undefined,
    });
    const before = fed.at(-1);

    channel.emit({ type: "message.updated", message: { id: "r1", reasoning: { id: "s1", offset: 0, text: "Hm" } } });

    const after = fed.at(-1);
    assert.equal(after?.[0], before?.[0]);
    assert.notEqual(after?.[1], before?.[1]);
  });

  it("asks a question again with the photos and UI action it went out with", async () => {
    const { session, transport } = await connected({
      images: async (attachments) => attachments.map((each) => ({ url: String(each) })),
    });
    const uiAction = { message_id: "r1", format: "a2ui", name: "pick", payload: {} };

    await transport.send(request({ attachments: ["data:image/png;base64,AA"], custom: { uiAction } }));
    await transport.send(request({ kind: "regenerate", messageId: "m2" }));

    assert.deepEqual(session.asked.at(-1)?.options, { images: [{ url: "data:image/png;base64,AA" }], uiAction });
  });

  it("asks a regenerate again as a new question, and refuses an edit", async () => {
    const { session, transport } = await connected();

    await transport.send(request({ kind: "regenerate", messageId: "m2" }));
    assert.deepEqual(session.asked.at(-1)?.options, {});

    await assert.rejects(transport.send(request({ kind: "edit" })), /edit/);
  });

  it("refuses photos it has no way to read", async () => {
    const { transport } = await connected();

    await assert.rejects(transport.send(request({ attachments: ["photo"] })), /images/);
  });

  it("stops telling a conversation that was disconnected", async () => {
    const { session, feed, disconnect } = await connected();
    const before = feed.activities.length;

    disconnect();
    session.state.set({ kind: "responding" });

    assert.equal(feed.activities.length, before);
  });
});
