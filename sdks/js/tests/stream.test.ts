import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { Client } from "../src/index.js";
import { registerChat, type ChatClient } from "../src/stream.js";

/** The user each chat client built here was connected as. */
const connectedAs: unknown[] = [];

/** stream-chat 9's shape: no `userId`, only a `userID` set once connected. */
class Chat9 implements ChatClient {
  userID?: string;
  channel(type: string, id: string): unknown {
    return { type, id };
  }
  async connectUser(user: { id: string }): Promise<void> {
    connectedAs.push(user);
    this.userID = user.id;
  }
  async disconnectUser(): Promise<void> {}
}

/** stream-chat 10's shape: `userId` is a getter, there before anybody connects. */
class Chat10 extends Chat9 {
  get userId(): string | undefined {
    return this.userID;
  }
}

/** The user the chat client the SDK builds is connected as, after `setUser(user)`. */
async function connect(Chat: typeof Chat9, user: { id: string; [field: string]: unknown }): Promise<unknown> {
  registerChat(Chat);
  connectedAs.length = 0;
  const api = new Client({ url: "http://127.0.0.1:9", customerId: "local", apiKey: "key" });
  await api.setUser(user, "token");
  await api.peers.chat();
  return connectedAs[0];
}

// The user `setUser` takes keeps the app's own fields top level, as stream-chat 9 does. 10
// takes them only under `custom`, and would not keep one written anywhere else.
describe("the user the built chat client connects as", () => {
  it("is the user as given, for stream-chat 9", async () => {
    const user = { id: "jim", name: "Jim", image: "https://example.com/jim.png", team: "blue" };

    assert.deepEqual(await connect(Chat9, user), user);
  });

  it("has the app's own fields under custom, for stream-chat 10", async () => {
    const user = { id: "jim", name: "Jim", image: "https://example.com/jim.png", language: "en", team: "blue" };

    assert.deepEqual(await connect(Chat10, user), {
      id: "jim",
      name: "Jim",
      image: "https://example.com/jim.png",
      language: "en",
      custom: { team: "blue" },
    });
  });

  it("keeps what the app already put under custom, for stream-chat 10", async () => {
    const user = { id: "jim", custom: { team: "blue" }, mood: "fine" };

    assert.deepEqual(await connect(Chat10, user), { id: "jim", custom: { mood: "fine", team: "blue" } });
  });

  it("sends no custom when there is nothing of the app's, for stream-chat 10", async () => {
    assert.deepEqual(await connect(Chat10, { id: "jim", name: "Jim" }), { id: "jim", name: "Jim" });
  });
});
