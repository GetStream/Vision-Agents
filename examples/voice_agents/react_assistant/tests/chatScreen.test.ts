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
    this.update({ messages: [...this.get().messages, { id: crypto.randomUUID(), text } as LocalMessage] });
  }
}

/** A router whose session opens when the test says, and that records what it deleted. */
function router() {
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
