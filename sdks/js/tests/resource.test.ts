import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import {
  Client,
  ConfigurationError,
  GUEST_STORAGE_KEY,
  RouterError,
  Session,
  type CreateSessionOptions,
  type GuestStore,
  type Schemas,
} from "../src/index.js";
import { TestRouter, type Connection } from "./router.js";

/** A session as the router describes one, with only the fields a test cares about set. */
function session(over: Partial<Schemas["Session"]> = {}): Schemas["Session"] {
  return {
    id: "session-1",
    call_id: "",
    call_type: "",
    user_id: "jlahey",
    agent_id: "docs",
    state: "live",
    created_at: new Date().toISOString(),
    ...over,
  };
}

describe("the agent handle", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("costs no request, because a name is not something to look up to hold", () => {
    const agent = api.agent("docs");

    assert.equal(agent.name, "docs");
    assert.equal(router.received.length, 0);
  });

  it("resolves the name to a config through the index rather than by listing every one", async () => {
    router.serve("GET", "/v1/agents/configs", {
      body: [{ id: "config-1", name: "docs" }],
    });

    const config = await api.agent("docs").config();

    assert.equal(config?.id, "config-1");
    assert.equal(router.last.query.get("name"), "docs");
  });

  it("says nothing is configured under a name rather than inventing one", async () => {
    router.serve("GET", "/v1/agents/configs", { body: [] });

    assert.equal(await api.agent("docs").config(), undefined);
  });
});

describe("sessions", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  /** Opens a session against the test router, holding the socket it opens. */
  async function open(
    options: CreateSessionOptions = {},
  ): Promise<{ session: Session; connection: Connection }> {
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: session({ conversation_id: "agent:support-7" }),
    });
    const opening = api.agent("docs").sessions.create(options);
    const connection = await router.socket();
    return { session: await opening, connection };
  }

  it("names the agent and holds the conversation in writing unless a call was given", async () => {
    const { session: held } = await open({ title: "Is Stream better than Sendbird" });

    const body = router.received[0]?.body as Record<string, unknown>;
    assert.equal(body["agent"], "docs");
    assert.equal(body["text"], true, "a session resource is a conversation");
    assert.equal(body["title"], "Is Stream better than Sendbird");
    assert.equal(held.conversationId, "agent:support-7");
    await held.close();
  });

  it("joins a call rather than writing when one is named", async () => {
    const { session: held } = await open({ call_id: "call-9" });

    const body = router.received[0]?.body as Record<string, unknown>;
    assert.equal(body["call_id"], "call-9");
    assert.equal(body["text"], undefined, "a call is not held in writing");
    await held.close();
  });

  it("spells the model overwrites the way the wire does", async () => {
    const { session: held } = await open({ modelOverwrites: { thinking: "high" } });

    const body = router.received[0]?.body as Record<string, unknown>;
    assert.deepEqual(body["model_overwrites"], { thinking: "high" });
    await held.close();
  });

  it("puts a query on the wire as the parameters the router reads", async () => {
    router.serve("GET", "/v1/agents/sessions", { body: { items: [session()], has_more: false } });
    const after = new Date("2026-01-01T00:00:00.000Z");

    await api.agent("docs").sessions.query({
      project: "Health",
      state: "closed",
      custom: { tab: "docs", seat: 4 },
      createdAfter: after,
      limit: 10,
      cursor: "page-2",
    });

    const query = router.last.query;
    assert.equal(query.get("agent"), "docs", "an agent's sessions are the agent's own");
    assert.equal(query.get("project"), "Health");
    assert.equal(query.get("state"), "closed");
    assert.equal(query.get("custom"), '{"tab":"docs","seat":4}');
    assert.equal(query.get("created_after"), after.toISOString());
    assert.equal(query.get("limit"), "10");
    assert.equal(query.get("cursor"), "page-2");
    assert.equal(query.get("user_id"), null, "a filter nobody set is not sent empty");
  });

  it("searches on the search path, carrying the same filters", async () => {
    router.serve("GET", "/v1/agents/sessions/search", { body: { items: [session()], has_more: false } });

    await api.agent("docs").sessions.search("sendbird", { project: "Health" });

    assert.equal(router.last.path, "/v1/agents/sessions/search");
    assert.equal(router.last.query.get("q"), "sendbird");
    assert.equal(router.last.query.get("project"), "Health");
    assert.equal(router.last.query.get("agent"), "docs");
  });

  it("forks into a session of its own and keeps watching it", async () => {
    const { session: parent } = await open();
    router.serve("POST", "/v1/agents/sessions/session-1/fork", {
      status: 201,
      body: session({ id: "session-2", forked_from: "session-1", title: "asked again" }),
    });

    const forking = parent.fork({ title: "asked again", modelOverwrites: { thinking: "high" } });
    const connection = await router.socket();
    const forked = await forking;

    assert.equal(forked.id, "session-2");
    assert.equal(forked.created.forked_from, "session-1");
    assert.match(connection.path, /session-2\/events$/, "the fork is watched, not the parent");
    const body = router.last.body as Record<string, unknown>;
    assert.equal(body["title"], "asked again");
    assert.deepEqual(body["model_overwrites"], { thinking: "high" });

    await forked.close();
    await parent.close();
  });

  it("changes a running session and hands back the session as it now is", async () => {
    const { session: held } = await open();
    router.serve("PATCH", "/v1/agents/sessions/session-1", {
      status: 200,
      body: session({ llm: "llm-thinking" }),
    });

    const updated = await held.update({ title: "Order 1042", llm: "llm-thinking", thinking: "high" });

    assert.equal(updated.id, "session-1");
    assert.equal(updated.llm, "llm-thinking");
    const [sent] = router.requestsTo("PATCH", "/v1/agents/sessions/session-1");
    assert.ok(sent, "no update reached the router");
    assert.deepEqual(
      sent.body,
      { title: "Order 1042", llm: "llm-thinking", thinking: "high" },
      "only what was named is sent",
    );

    await held.close();
  });

  it("deletes what one session remembered without ending it", async () => {
    const { session: held } = await open();
    router.serve("DELETE", "/v1/agents/sessions/session-1/memories", { status: 204 });

    await held.deleteMemories();

    assert.equal(router.requestsTo("DELETE", "/v1/agents/sessions/session-1/memories").length, 1);
    assert.equal(
      router.requestsTo("DELETE", "/v1/agents/sessions/session-1").length,
      0,
      "deleting what a session remembered must not end it",
    );
    await held.close();
  });

  it("deletes an ended session's memories by its id", async () => {
    router.serve("DELETE", "/v1/agents/sessions/session-9/memories", { status: 204 });

    await api.agent("docs").sessions.deleteMemories("session-9");

    assert.equal(router.last.path, "/v1/agents/sessions/session-9/memories");
  });

  it("says why the router would not delete a session's memories", async () => {
    router.serve("DELETE", "/v1/agents/sessions/someone-elses/memories", {
      status: 404,
      body: { error: "unknown session" },
    });

    await assert.rejects(
      api.agent("docs").sessions.deleteMemories("someone-elses"),
      (error: unknown) => error instanceof RouterError && error.status === 404,
    );
  });
});

describe("memories", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("truncates everything remembered about one user", async () => {
    router.serve("DELETE", "/v1/agents/users/user%20123/memories", { status: 204 });

    await api.memories.truncate("user 123");

    assert.equal(router.last.method, "DELETE");
    assert.equal(router.last.path, "/v1/agents/users/user%20123/memories");
  });

  it("refuses to truncate nobody before asking the router", async () => {
    await assert.rejects(api.memories.truncate(""), ConfigurationError);
    assert.equal(router.received.length, 0);
  });
});

describe("responses", () => {
  let router: TestRouter;
  let api: Client;
  let held: Session;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
    router.serve("POST", "/v1/agents/sessions", { status: 201, body: session() });
    const opening = api.agent("docs").sessions.create();
    await router.socket();
    held = await opening;
  });

  afterEach(async () => {
    await held.close();
    await router.stop();
  });

  it("asks the agent something and hands back a handle on that one turn", async () => {
    router.serve("POST", "/v1/agents/sessions/session-1/responses", {
      status: 202,
      body: {
        id: "response-1",
        session_id: "session-1",
        status: "running",
        said: "Is Stream better than Sendbird?",
        created_at: new Date().toISOString(),
      },
    });

    const answering = await held.responses.create("Is Stream better than Sendbird?");

    assert.equal(answering.id, "response-1");
    assert.equal(answering.status, "running");
    const body = router.last.body as Record<string, unknown>;
    assert.equal(body["text"], "Is Stream better than Sendbird?");
    assert.equal(body["command_id"], undefined, "a session keeping no conversation names no command");
  });

  it("names each question on a session that keeps its conversation", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: session({ id: "session-2", conversation_id: "agent:support-1" }),
    });
    const kept = await api.agent("docs").sessions.create({ text: true, watch: false });
    router.serve("POST", "/v1/agents/sessions/session-2/responses", {
      status: 202,
      body: { id: "response-1", session_id: "session-2", status: "running", created_at: new Date().toISOString() },
    });

    await kept.responses.create("First question");
    const first = (router.last.body as Record<string, unknown>)["command_id"];
    await kept.responses.create("Second question");
    const second = (router.last.body as Record<string, unknown>)["command_id"];
    await kept.responses.create("Retried question", { commandId: "request-7" });

    assert.match(String(first), /^[0-9a-f-]{36}$/);
    assert.notEqual(first, second, "two questions are two commands");
    assert.equal((router.last.body as Record<string, unknown>)["command_id"], "request-7");
    await kept.close();
  });

  it("reads one turn's items rather than the whole conversation's", async () => {
    router.serve("POST", "/v1/agents/sessions/session-1/responses", {
      status: 202,
      body: {
        id: "response-1",
        session_id: "session-1",
        status: "running",
        created_at: new Date().toISOString(),
      },
    });
    router.serve("GET", "/v1/agents/sessions/session-1/responses/items", { body: { items: [], has_more: false } });

    const answering = await held.responses.create("Anything");
    await answering.items.all();

    assert.equal(router.last.query.get("response_id"), "response-1");
  });

  it("reads the whole conversation's items when nothing narrows it", async () => {
    router.serve("GET", "/v1/agents/sessions/session-1/responses/items", { body: { items: [], has_more: false } });

    await held.responses.items.all();

    assert.equal(router.last.query.get("response_id"), null);
  });

  it("follows the cursor until a page says there is no more", async () => {
    const page = 2;
    router.serve("GET", "/v1/agents/sessions/session-1/responses/items", (_, calls) => ({
      body:
        calls === 0
          ? { items: [item(0), item(1)], has_more: true, next_cursor: "after-1" }
          : { items: [item(2)], has_more: false },
    }));

    const read: number[] = [];
    for await (const one of held.responses.items.unwind({ limit: page })) {
      read.push(one.ordinal);
    }

    assert.deepEqual(read, [0, 1, 2]);
    const asked = router.requestsTo("GET", "/v1/agents/sessions/session-1/responses/items");
    assert.equal(asked.length, 2, "the last page says so, so nothing is asked again");
    assert.equal(asked[1]?.query.get("cursor"), "after-1", "the second page picks up where the first ended");
  });

  it("rewinds to the response an item belongs to, since an item is what a transcript shows", async () => {
    // A stored text conversation cannot be rewound, so this is a call's, read back.
    const call = api.agent("docs").sessions.responses("call-1");
    router.serve("POST", "/v1/agents/sessions/call-1/rewind", { status: 204 });

    await call.rewind(item(3));
    assert.deepEqual(router.last.body, { response_id: "response-1" });

    await call.rewind("response-2");
    assert.deepEqual(router.last.body, { response_id: "response-2" });
  });

  it("says why the router would not rewind", async () => {
    router.serve("POST", "/v1/agents/sessions/session-1/rewind", {
      status: 400,
      body: { error: "session: a stored text conversation cannot be rewound; fork it at the response" },
    });

    await assert.rejects(
      () => held.responses.rewind("response-1"),
      (error: unknown) =>
        error instanceof RouterError && error.status === 400 && /cannot be rewound/.test(error.message),
    );
  });

  it("refuses to rewind to a response a session that records nothing never had", async () => {
    await assert.rejects(() => held.responses.rewind(""), ConfigurationError);
    assert.equal(router.requestsTo("POST", "/v1/agents/sessions/session-1/rewind").length, 0);
  });

  it("forks at a response, carrying the history only that far", async () => {
    router.serve("POST", "/v1/agents/sessions/session-1/fork", {
      status: 201,
      body: session({ id: "session-2", forked_from: "session-1" }),
    });

    const forked = await held.fork({ response_id: "response-1", watch: false });

    assert.equal(forked.id, "session-2");
    assert.deepEqual(router.last.body, { response_id: "response-1" });
  });
});

function item(ordinal: number): Schemas["AgentResponseItem"] {
  return {
    response_id: "response-1",
    session_id: "session-1",
    ordinal,
    kind: "answer",
    at: new Date().toISOString(),
  };
}

describe("guest users", () => {
  let router: TestRouter;
  let api: Client;

  /** Somewhere to remember a guest, standing in for a cookie a server does not have. */
  function memory(): GuestStore & { value: string | undefined } {
    return {
      value: undefined,
      read() {
        return this.value;
      },
      write(value: string) {
        this.value = value;
      },
      clear() {
        this.value = undefined;
      },
    };
  }

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
    router.serve("POST", "/v1/agents/guests", {
      status: 201,
      body: { id: "guest-1", token: "token-for-guest", name: "Guest" },
    });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("mints a guest and remembers them", async () => {
    const store = memory();

    const guest = await api.guestUser({ name: "Visitor" }, store);

    assert.equal(guest.id, "guest-1");
    assert.equal((router.last.body as Record<string, unknown>)["name"], "Visitor");
    assert.equal(JSON.parse(store.value ?? "{}").id, "guest-1");
  });

  it("gives back the remembered guest rather than minting a second", async () => {
    const store = memory();
    store.write(JSON.stringify({ id: "guest-7", token: "token-for-seven" }));

    const guest = await api.guestUser({}, store);

    assert.equal(guest.id, "guest-7", "reloading the page is the same visitor");
    assert.equal(router.received.length, 0);
  });

  it("mints a fresh one when the person says they are not that guest", async () => {
    const store = memory();
    store.write(JSON.stringify({ id: "guest-7", token: "token-for-seven" }));

    const guest = await api.guestUser({ fresh: true }, store);

    assert.equal(guest.id, "guest-1");
  });

  it("mints again rather than throwing when what was remembered is not a guest", async () => {
    const store = memory();
    store.write("something else was under the key");

    const guest = await api.guestUser({}, store);

    assert.equal(guest.id, "guest-1");
  });

  it("forgets a guest, which is what signing in has to do", async () => {
    const store = memory();
    await api.guestUser({}, store);

    api.forgetGuest(store);

    assert.equal(store.value, undefined);
  });

  it("refuses a claim from a caller that is not the app's own backend", async () => {
    const page = new Client({ url: router.url, apiKey: "vak_live_x" });
    await page.setUser({ id: "jlahey" }, "token-for-jim");

    assert.throws(() => page.claimGuestUser("guest-1", "jlahey"), ConfigurationError);
    assert.equal(router.received.length, 0, "it did not reach the router to be refused there");
  });

  it("claims a guest onto an account from a backend", async () => {
    router.serve("POST", "/v1/agents/guests/claim", {
      body: { guest_id: "guest-1", user_id: "jlahey", sessions_moved: 3 },
    });

    const claimed = await api.claimGuestUser("guest-1", { id: "jlahey", name: "Jim Lahey" });

    assert.equal(claimed.sessions_moved, 3);
    assert.deepEqual(router.last.body, { guest_id: "guest-1", user_id: "jlahey" });
  });

  it("keeps the guest under one key, so two tabs are one visitor", () => {
    assert.equal(GUEST_STORAGE_KEY, "stream-vision-agents-guest");
  });
});
