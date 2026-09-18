import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import {
  Client,
  ConfigurationError,
  GUEST_STORAGE_KEY,
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
    router.serve("GET", "/v1/agents/sessions", { body: [session()] });
    const after = new Date("2026-01-01T00:00:00.000Z");

    await api.agent("docs").sessions.query({
      project: "Health",
      state: "closed",
      custom: { tab: "docs", seat: 4 },
      createdAfter: after,
      limit: 10,
    });

    const query = router.last.query;
    assert.equal(query.get("agent"), "docs", "an agent's sessions are the agent's own");
    assert.equal(query.get("project"), "Health");
    assert.equal(query.get("state"), "closed");
    assert.equal(query.get("custom"), '{"tab":"docs","seat":4}');
    assert.equal(query.get("created_after"), after.toISOString());
    assert.equal(query.get("limit"), "10");
    assert.equal(query.get("user_id"), null, "a filter nobody set is not sent empty");
  });

  it("searches on the search path, carrying the same filters", async () => {
    router.serve("GET", "/v1/agents/sessions/search", { body: [session()] });

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
    assert.equal(
      (router.last.body as Record<string, unknown>)["text"],
      "Is Stream better than Sendbird?",
    );
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
    router.serve("GET", "/v1/agents/sessions/session-1/responses/items", { body: [] });

    const answering = await held.responses.create("Anything");
    await answering.items.all();

    assert.equal(router.last.query.get("response_id"), "response-1");
  });

  it("reads the whole conversation's items when nothing narrows it", async () => {
    router.serve("GET", "/v1/agents/sessions/session-1/responses/items", { body: [] });

    await held.responses.items.all();

    assert.equal(router.last.query.get("response_id"), null);
  });

  it("pages until a short page says there is no more", async () => {
    const page = 2;
    router.serve("GET", "/v1/agents/sessions/session-1/responses/items", (_, calls) => ({
      body: calls === 0 ? [item(0), item(1)] : [item(2)],
    }));

    const read: number[] = [];
    for await (const one of held.responses.items.unwind({ limit: page })) {
      read.push(one.ordinal);
    }

    assert.deepEqual(read, [0, 1, 2]);
    const asked = router.requestsTo("GET", "/v1/agents/sessions/session-1/responses/items");
    assert.equal(asked.length, 2, "a short page is the last page, so nothing is asked again");
    assert.equal(asked[1]?.query.get("offset"), "2", "the second page picks up where the first ended");
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
