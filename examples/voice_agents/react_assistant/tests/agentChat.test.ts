import type { Session } from "@stream-io/stream-agents";
import type { LocalMessage } from "stream-chat";
import { describe, expect, it } from "vitest";

import { AgentChat } from "../src/agentChat";
import { Store } from "../src/store";
import { troubleText } from "../src/voicePresence";

type PaginatorState = { items: LocalMessage[] | undefined; hasMoreTail: boolean };

/**
 * The paginator's state store, as far as the chat uses it: `subscribeWithSelector`, which tells a
 * follower at once and on every change. It says how many follow it, so a test sees what was let go.
 */
class PaginatorStore {
  private value: PaginatorState;
  private readonly followers = new Set<(value: PaginatorState) => void>();

  constructor(initial: PaginatorState) {
    this.value = initial;
  }

  get following(): number {
    return this.followers.size;
  }

  getLatestValue(): PaginatorState {
    return this.value;
  }

  partialNext(patch: Partial<PaginatorState>): void {
    this.value = { ...this.value, ...patch };
    for (const follower of [...this.followers]) follower(this.value);
  }

  subscribeWithSelector<O>(select: (value: PaginatorState) => O, handler: (selected: O) => void): () => void {
    const follower = (value: PaginatorState) => handler(select(value));
    this.followers.add(follower);
    follower(this.value);
    return () => {
      this.followers.delete(follower);
    };
  }
}

/** stream-chat 10's message paginator, as far as the chat reads it: a page, and older ones. */
class FakePaginator {
  pageSize = 0;
  loadsOfOlder = 0;
  older: LocalMessage[] = [];
  readonly state = new PaginatorStore({ items: undefined, hasMoreTail: true });

  get hasMoreTail(): boolean {
    return this.state.getLatestValue().hasMoreTail;
  }

  /** The page `watch()` places. */
  place(items: LocalMessage[], hasMoreTail: boolean): void {
    this.state.partialNext({ items, hasMoreTail });
  }

  async toTail(): Promise<void> {
    this.loadsOfOlder++;
    const { items } = this.state.getLatestValue();
    this.state.partialNext({ items: [...this.older, ...(items ?? [])], hasMoreTail: false });
    this.older = [];
  }
}

/** A conversation's channel: its live updates, its paginator, and being on screen. */
class FakeChannel {
  readonly messagePaginator = new FakePaginator();
  readonly listeners = new Set<(event: { message: { id: string } }) => void>();
  active = 0;
  page: LocalMessage[] = [];
  watchFails: Error | undefined;

  on(_type: "message.updated", listener: (event: { message: { id: string } }) => void) {
    this.listeners.add(listener);
    return { unsubscribe: () => this.listeners.delete(listener) };
  }

  activate(): () => void {
    this.active++;
    let released = false;
    return () => {
      if (!released) {
        released = true;
        this.active--;
      }
    };
  }

  async watch(): Promise<this> {
    if (this.watchFails) throw this.watchFails;
    this.messagePaginator.place(this.page, true);
    return this;
  }
}

/** The Stream Agents session the chat is opened on: what the agent is doing, failures, and closing. */
function session(channel: FakeChannel) {
  const fake = {
    closed: false,
    state: new Store<{ kind: string }>({ kind: "idle" }),
    conversation: new Store({ failure: "", failureContext: "", failureAt: 0 }),
    chat: async () => ({ client: {}, channel }),
    responses: { create: async () => undefined },
    interrupt: () => undefined,
    close: async () => {
      fake.closed = true;
    },
  };
  return fake;
}

function open(fake: ReturnType<typeof session>): Promise<AgentChat> {
  return AgentChat.open(fake as unknown as Session);
}

/** A message as stream-chat 10 holds it: fields it does not define are under `custom`. */
function message(text: string, custom: Record<string, unknown> = {}): LocalMessage {
  return { id: crypto.randomUUID(), text, attachments: [], custom } as unknown as LocalMessage;
}

describe("AgentChat", () => {
  it("shows the channel's messages, and whether older ones are left, from its paginator", async () => {
    const channel = new FakeChannel();
    channel.page = [message("Hi"), message("Hello!", { ai_generated: true })];

    const chat = await open(session(channel));

    expect(chat.get().messages.map((each) => each.text)).toEqual(["Hi", "Hello!"]);
    expect(chat.get().hasOlderMessages).toBe(true);
    expect(channel.messagePaginator.pageSize).toBe(30);
  });

  it("loads older messages through the paginator, and asks no more once there are none", async () => {
    const channel = new FakeChannel();
    channel.page = [message("Newer")];
    channel.messagePaginator.older = [message("Older")];
    const chat = await open(session(channel));

    await chat.loadOlderMessages();
    await chat.loadOlderMessages();

    expect(chat.get().messages.map((each) => each.text)).toEqual(["Older", "Newer"]);
    expect(chat.get().hasOlderMessages).toBe(false);
    expect(channel.messagePaginator.loadsOfOlder).toBe(1);
  });

  it("is answering while the agent's reply says it is still being written", async () => {
    const channel = new FakeChannel();
    const reply = message("Half a", { ai_generated: true, generating: true });
    channel.page = [message("Question"), reply];
    const chat = await open(session(channel));

    expect(chat.isAnswering()).toBe(true);

    channel.messagePaginator.place([channel.page[0]!, { ...reply, custom: { ai_generated: true, generating: false } }], true);

    expect(chat.isAnswering()).toBe(false);
  });

  it("tells the person when the agent could not answer what they asked", async () => {
    const channel = new FakeChannel();
    const fake = session(channel);
    const chat = await open(fake);
    await chat.ask("How are you?");

    fake.conversation.update({ failure: "402 Payment Required", failureContext: "llm", failureAt: Date.now() });

    expect(chat.get().failure).toBe(troubleText("llm"));
    expect(chat.get().askedAfter).toBeUndefined();
  });

  it("leaves nothing followed or open when the channel cannot be watched", async () => {
    const channel = new FakeChannel();
    channel.watchFails = new Error("No connection id is available");
    const fake = session(channel);

    await expect(open(fake)).rejects.toThrow("No connection id is available");

    expect(channel.listeners.size).toBe(0);
    expect(channel.messagePaginator.state.following).toBe(0);
    expect(channel.active).toBe(0);
    expect(fake.closed).toBe(true);
  });

  it("lets go of the channel when it closes", async () => {
    const channel = new FakeChannel();
    const fake = session(channel);
    const chat = await open(fake);
    expect(channel.active).toBe(1);

    await chat.close();

    expect(channel.listeners.size).toBe(0);
    expect(channel.messagePaginator.state.following).toBe(0);
    expect(channel.active).toBe(0);
    expect(fake.closed).toBe(true);
  });
});
