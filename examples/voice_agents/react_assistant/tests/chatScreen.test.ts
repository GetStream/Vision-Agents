import type { LocalMessage } from "stream-chat";
import { describe, expect, it } from "vitest";

import type { AgentChat } from "../src/agentChat";
import { ChatScreen, type ChatScreenDeps } from "../src/chatScreen";
import { Store } from "../src/store";

/** A conversation's chat as the screen uses it: its messages, what was asked, and closing. */
class FakeChat extends Store<{ messages: LocalMessage[]; failure: string | undefined }> {
  readonly asked: string[] = [];
  closed = false;
  answering = false;
  readonly session = { id: "session-1" };

  constructor() {
    super({ messages: [], failure: undefined });
  }

  async ask(text: string): Promise<void> {
    this.asked.push(text);
  }

  isAnswering(): boolean {
    return this.answering;
  }

  async close(): Promise<void> {
    this.closed = true;
  }

  /** Stream's copy of the person's question arriving in the channel. */
  arrives(text: string): void {
    this.update({ messages: [...this.get().messages, { id: crypto.randomUUID(), text, custom: {} } as LocalMessage] });
  }
}

/** A message as Stream holds it: the person's, or the agent's reply. */
function message(text: string, isReply = false): LocalMessage {
  // stream-chat 10 keeps fields it does not define under `custom`, and every message has one.
  return { id: crypto.randomUUID(), text, custom: isReply ? { ai_generated: true } : {} } as unknown as LocalMessage;
}

/**
 * A router whose session opens when the test says, and that records what it deleted, beside a
 * Stream channel that hands back `history` at once.
 */
function router(history: LocalMessage[] = []) {
  const chat = new FakeChat();
  const removed: string[] = [];
  let open: (() => void) | undefined;
  const opened = new Promise<void>((resolve) => {
    open = resolve;
  });
  const deps: ChatScreenDeps = {
    open: async () => {
      await opened;
      return chat as unknown as AgentChat;
    },
    history: async () => history,
    remove: async (id) => {
      removed.push(id);
    },
    voiceOf: () => {
      throw new Error("no call in these tests");
    },
  };
  return { chat, removed, deps, open: () => open?.() };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("ChatScreen", () => {
  it("sends nothing more while a new conversation's session is still opening", async () => {
    const { chat, deps, open } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);

    const first = screen.send("Write me a poem about Lisbon");
    expect(await screen.send("and one about Porto")).toBe(false);
    open();
    expect(await first).toBe(true);

    expect(chat.asked).toEqual(["Write me a poem about Lisbon"]);
  });

  it("says a conversation from the list is loading until its session is open", async () => {
    const { deps, open } = router();
    const screen = new ChatScreen("session-1", deps, () => undefined);
    expect(screen.get().isLoading).toBe(true);

    open();
    await settle();
    expect(screen.get().isLoading).toBe(false);
    expect(screen.get().chat).toBeDefined();
  });

  it("stops saying it is loading when the conversation could not be opened", async () => {
    const { deps } = router();
    const failing: ChatScreenDeps = { ...deps, open: async () => Promise.reject(new Error("the router is down")) };
    const screen = new ChatScreen("session-1", failing, () => undefined);

    await settle();
    expect(screen.get().isLoading).toBe(false);
    expect(screen.get().failure).toContain("the router is down");
  });

  it("shows a conversation's messages from Stream before its session is open", async () => {
    const earlier = [message("Plan a weekend in Lisbon"), message("Here is a plan", true)];
    const { deps } = router(earlier);
    const screen = new ChatScreen("session-1", deps, () => undefined, "agent:session-1");

    await settle();
    expect(screen.get().history).toEqual(earlier);
    expect(screen.get().isLoading).toBe(false);
    expect(screen.get().chat).toBeUndefined();
  });

  it("reads the history of a conversation opened from a link, by its session id alone", async () => {
    const earlier = [message("Plan a weekend in Lisbon")];
    const asked: Array<[string, string | undefined]> = [];
    const { deps } = router();
    const linked: ChatScreenDeps = {
      ...deps,
      history: async (sessionId, conversationId) => {
        asked.push([sessionId, conversationId]);
        return earlier;
      },
    };
    const screen = new ChatScreen("session-1", linked, () => undefined);

    await settle();
    expect(asked).toEqual([["session-1", undefined]]);
    expect(screen.get().history).toEqual(earlier);
  });

  it("lets the chat's own messages take over once its session is open", async () => {
    const { chat, deps, open } = router([message("Plan a weekend in Lisbon")]);
    const screen = new ChatScreen("session-1", deps, () => undefined, "agent:session-1");
    await settle();

    open();
    await settle();
    expect(screen.get().chat).toBe(chat);
    expect(screen.get().history).toBeUndefined();
  });

  it("keeps a question asked over the history until Stream's copy of it arrives", async () => {
    const earlier = [message("Plan a weekend in Lisbon"), message("Here is a plan", true)];
    const { chat, deps, open } = router(earlier);
    const screen = new ChatScreen("session-1", deps, () => undefined, "agent:session-1");
    await settle();

    const sending = screen.send("Make it three days");
    // The chat opens holding what the history held: the question has not arrived yet.
    chat.update({ messages: [...earlier] });
    open();
    await sending;
    // Anything the chat says meanwhile, such as the session starting to respond.
    chat.update({});
    expect(screen.get().pending?.text).toBe("Make it three days");

    chat.arrives("Make it three days");
    expect(screen.get().pending).toBeUndefined();
  });

  it("ignores history that arrives after the session opened", async () => {
    const { chat, deps, open } = router();
    let giveHistory: (messages: LocalMessage[]) => void = () => undefined;
    const late: ChatScreenDeps = { ...deps, history: () => new Promise((resolve) => (giveHistory = resolve)) };
    open();
    const screen = new ChatScreen("session-1", late, () => undefined, "agent:session-1");
    await settle();
    expect(screen.get().chat).toBe(chat);

    giveHistory([message("Stale")]);
    await settle();
    expect(screen.get().history).toBeUndefined();
  });

  it("does not say a new conversation is loading", () => {
    const { deps } = router();
    expect(new ChatScreen(undefined, deps, () => undefined).get().isLoading).toBe(false);
  });

  it("waits for a reply that is still being written before sending the next question", async () => {
    const { chat, deps, open } = router();
    open();
    const screen = new ChatScreen("session-1", deps, () => undefined);
    await settle();
    chat.answering = true;

    expect(await screen.send("one more thing")).toBe(false);
    expect(chat.asked).toEqual([]);
  });

  it("shows a question at once, until Stream's copy of it arrives", async () => {
    const { chat, deps, open } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);

    const sending = screen.send("What's on my calendar today?");
    expect(screen.get().pending?.text).toBe("What's on my calendar today?");
    open();
    await sending;
    chat.arrives("What's on my calendar today?");

    expect(screen.get().pending).toBeUndefined();
  });

  it("deletes a new conversation's session the person left without sending", async () => {
    const { chat, removed, deps, open } = router();
    const screen = new ChatScreen(undefined, deps, () => undefined);

    screen.draftChanged("Remind me");
    open();
    await settle();
    screen.leave();
    await settle();

    expect(chat.closed).toBe(true);
    expect(removed).toEqual(["session-1"]);
  });

  it("names a new conversation to the list once it is used, and keeps it on leaving", async () => {
    const { removed, deps, open } = router();
    const started: string[] = [];
    const screen = new ChatScreen(undefined, deps, (id) => started.push(id));

    screen.draftChanged("Plan");
    open();
    await screen.send("Plan a weekend in Lisbon");
    screen.leave();
    await settle();

    expect(started).toEqual(["session-1"]);
    expect(removed).toEqual([]);
  });
});
