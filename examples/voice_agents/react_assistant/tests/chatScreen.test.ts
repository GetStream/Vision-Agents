import { StateStore } from "@stream-io/state-store";
import type { Session } from "@stream-io/stream-agents";
import type { VoiceSession } from "@stream-io/stream-agents/video";
import { describe, expect, it } from "vitest";

import { ChatScreen, THINK_HARDER_ID, type ChatScreenDeps } from "../src/chatScreen";

type Message = { id: string; text: string; custom: Record<string, unknown> };

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

/** A stream-chat 10 channel: its messages are its paginator's. */
class FakeChannel {
  readonly messagePaginator = {
    pageSize: 0,
    state: new StateStore<{ items: Message[] | undefined; hasMoreTail: boolean }>({ items: undefined, hasMoreTail: false }),
    get hasMoreTail() {
      return this.state.getLatestValue().hasMoreTail;
    },
    toTail: async () => undefined,
  };

  constructor(private readonly latest: Message[] = []) {}

  on() {
    return { unsubscribe: () => undefined };
  }
  activate() {
    return () => undefined;
  }
  async watch() {
    this.messagePaginator.state.next({ items: [...this.latest], hasMoreTail: false });
    return this;
  }
}

/** A session on the router, recording what it was asked and how it was let go. */
class FakeSession {
  live = true;
  closed = 0;
  deleted = 0;
  voiceStarted = false;
  readonly asked: { text: string; options: Record<string, unknown> }[] = [];
  readonly state = store<{ kind: string; skills: readonly string[] }>({ kind: "idle", skills: [] });
  readonly conversation = store({ failure: "", failureContext: "", failureAt: 0, turns: [] });
  readonly responses = {
    create: async (text: string, options: Record<string, unknown> = {}) => {
      this.asked.push({ text, options });
      return {};
    },
  };
  readonly voice = {
    start: async () => void (this.voiceStarted = true),
    stop: async () => void (this.voiceStarted = false),
  };

  constructor(
    readonly id: string,
    private readonly channel: FakeChannel,
  ) {}

  async chat() {
    return { client: {}, channel: this.channel };
  }
  interrupt() {}
  decide() {}
  declineLogin() {}
  async close() {
    this.closed += 1;
  }
  async delete() {
    this.deleted += 1;
  }
}

/**
 * A router whose sessions open when the test says (`open()`), each on `channel`, beside Stream,
 * which hands back the conversation's channel at once.
 */
function router(channel = new FakeChannel()) {
  const sessions: FakeSession[] = [];
  const asked: (string | undefined)[] = [];
  let release: () => void = () => undefined;
  const opened = new Promise<void>((resolve) => (release = resolve));
  const deps: ChatScreenDeps = {
    open: async (sessionId) => {
      asked.push(sessionId);
      await opened;
      const session = new FakeSession(sessionId ?? `new-${sessions.length + 1}`, channel);
      sessions.push(session);
      return session as unknown as Session;
    },
    channel: async () => channel,
    approver: () => ({ userId: "juno-owner", clientId: "test", decide: async () => undefined }),
    voiceOf: () =>
      ({ join: async () => undefined, leave: async () => undefined }) as unknown as VoiceSession,
  };
  return { deps, sessions, asked, open: () => release() };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("ChatScreen", () => {
  it("opens no session for a new conversation until the person writes", async () => {
    const { deps, asked, open } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);
    open();
    await settle();
    expect(asked).toEqual([]);

    screen.draftChanged("Remind me");
    await settle();
    expect(asked).toEqual([undefined]);
  });

  it("deletes a new conversation's session the person left without sending", async () => {
    const { deps, sessions, open } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);
    screen.draftChanged("Remind me");
    open();
    await settle();

    screen.leave();
    await settle();

    expect(sessions[0]?.deleted).toBe(1);
    expect(sessions[0]?.closed).toBe(0);
    expect(screen.chat.isClosed).toBe(true);
  });

  it("shows a question at once and asks it under the id it shows with", async () => {
    const { deps, sessions, open } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);

    screen.conversation.send({ text: "What's on my calendar today?" });
    const pending = screen.conversation.state.getLatestValue().pendingMessage;
    expect(pending?.text).toBe("What's on my calendar today?");

    open();
    await settle();
    // The router stores the person's message under the request id, so Stream's copy replaces it.
    expect(sessions[0]?.asked).toEqual([{ text: "What's on my calendar today?", options: { requestId: pending?.id } }]);
  });

  it("asks with more time to reason when Think harder is picked", async () => {
    const { deps, sessions, open } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);
    open();

    screen.conversation.send({ text: "Plan a weekend in Lisbon", chatOption: { id: THINK_HARDER_ID, title: "Think harder" } });
    await settle();

    expect(sessions[0]?.asked[0]?.options.reasoningEffort).toBe("high");
  });

  it("sends what the person did in a reply's UI as the question's UI action", async () => {
    const { deps, sessions, open } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);
    open();
    const uiAction = { message_id: "r1", format: "a2ui", name: "pick", payload: { day: 2 } };

    screen.conversation.send({ text: "Day 2", custom: { uiAction } });
    await settle();

    expect(sessions[0]?.asked[0]?.options.uiAction).toEqual(uiAction);
  });

  it("offers neither editing nor regenerating, which the router can't do", () => {
    const { deps } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);

    expect(screen.conversation.actions).toMatchObject({ edit: false, regenerate: false, rate: true });
  });

  it("names a new conversation to the list once it is asked something, and keeps it on leaving", async () => {
    const { deps, sessions, open } = router();
    const started: string[] = [];
    const screen = new ChatScreen(undefined, deps, (id) => started.push(id));
    open();

    screen.conversation.send({ text: "Plan a weekend in Lisbon" });
    await settle();
    screen.leave();
    await settle();

    expect(started).toEqual(["new-1"]);
    expect(sessions[0]?.closed).toBe(1);
    expect(sessions[0]?.deleted).toBe(0);
  });

  it("counts a new conversation talked to as used: it joins the list and is kept", async () => {
    const { deps, sessions, open } = router();
    const started: string[] = [];
    const screen = new ChatScreen(undefined, deps, (id) => started.push(id));
    open();

    await screen.talk();
    expect(sessions[0]?.voiceStarted).toBe(true);
    expect(screen.get().isVoiceShown).toBe(true);

    screen.leave();
    await settle();
    expect(started).toEqual(["new-1"]);
    expect(sessions[0]?.deleted).toBe(0);
    expect(sessions[0]?.voiceStarted).toBe(false);
  });

  it("shows a listed conversation's messages before its session is open", async () => {
    const earlier: Message[] = [
      { id: "q1", text: "Plan a weekend in Lisbon", custom: {} },
      { id: "r1", text: "Here is a plan", custom: { ai_generated: true } },
    ];
    const { deps, sessions } = router(new FakeChannel(earlier));
    const screen = new ChatScreen("session-1", deps, () => undefined, "agent:session-1");
    expect(screen.chat.state.getLatestValue().isLoaded).toBe(false);

    await settle();
    expect(sessions).toEqual([]);
    expect(screen.chat.state.getLatestValue().isLoaded).toBe(true);
    expect(screen.conversation.state.getLatestValue().messages.map((message) => message.id)).toEqual(["q1", "r1"]);
  });

  it("carries a listed conversation on without naming it to the list again", async () => {
    const { deps, asked, open } = router();
    const started: string[] = [];
    const screen = new ChatScreen("session-1", deps, (id) => started.push(id));
    open();

    screen.conversation.send({ text: "Make it three days" });
    await settle();

    expect(asked).toEqual(["session-1"]);
    expect(started).toEqual([]);
  });

  it("says why a conversation could not be opened", async () => {
    const { deps } = router();
    const failing: ChatScreenDeps = { ...deps, open: async () => Promise.reject(new Error("the router is down")) };
    const screen = new ChatScreen("session-1", failing, () => undefined);

    await settle();
    expect(screen.get().failure).toContain("the router is down");
  });

  it("asks nothing more of the router once the screen is left", async () => {
    const { deps, asked, open } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);
    open();
    screen.leave();

    screen.conversation.send({ text: "Anyone there?" });
    await settle();

    expect(screen.chat.isClosed).toBe(true);
    expect(asked).toEqual([]);
  });
});
