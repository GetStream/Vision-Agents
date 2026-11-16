import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Backend, Client, RouterError, Socket, SocketClosedError, signToken } from "../src/index.js";
import { TestRouter, claimsOf } from "./router.js";

/** A token function that answers token-1, token-2, … and counts how often it was asked. */
function counting(): { source: () => Promise<string>; asked: () => number } {
  let asked = 0;
  return {
    source: async () => {
      asked += 1;
      return `token-${asked}`;
    },
    asked: () => asked,
  };
}

describe("a token function", () => {
  let router: TestRouter;

  beforeEach(async () => {
    router = await TestRouter.start();
  });

  afterEach(async () => {
    await router.stop();
  });

  it("is asked once, not once per request", async () => {
    router.serve("GET", "/v1/agents/sessions/sess_1", { body: { id: "sess_1" } });
    const tokens = counting();
    const api = new Client({ url: router.url, apiKey: "vak_live_x" });
    await api.setUser("jlahey", tokens.source);

    for (let i = 0; i < 3; i += 1) {
      await api.get("/v1/agents/sessions/{id}", { path: { id: "sess_1" } });
    }

    assert.equal(tokens.asked(), 1);
    assert.deepEqual(
      router.received.map((r) => r.headers["authorization"]),
      ["Bearer token-1", "Bearer token-1", "Bearer token-1"],
    );
  });

  it("is asked once by requests that need it at the same time", async () => {
    router.serve("GET", "/v1/agents/sessions/sess_1", { body: { id: "sess_1" } });
    let asked = 0;
    let release: (token: string) => void = () => {};
    const api = new Client({ url: router.url, apiKey: "vak_live_x" });
    await api.setUser("jlahey", () => {
      asked += 1;
      return new Promise<string>((resolve) => (release = resolve));
    });

    const requests = [1, 2, 3].map(() =>
      api.get("/v1/agents/sessions/{id}", { path: { id: "sess_1" } }),
    );
    await new Promise((resolve) => setImmediate(resolve));
    release("token-1");
    await Promise.all(requests);

    assert.equal(asked, 1);
  });

  it("is asked again after a 401, and the request sent once more", async () => {
    router.serve("GET", "/v1/agents/sessions/sess_1", (request) =>
      request.headers["authorization"] === "Bearer token-1"
        ? { status: 401, body: { error: { message: "token expired" } } }
        : { body: { id: "sess_1" } },
    );
    const tokens = counting();
    const api = new Client({ url: router.url, apiKey: "vak_live_x" });
    await api.setUser("jlahey", tokens.source);

    const session = await api.get("/v1/agents/sessions/{id}", { path: { id: "sess_1" } });

    assert.equal(session.id, "sess_1");
    assert.equal(tokens.asked(), 2);
    assert.deepEqual(
      router.received.map((r) => r.headers["authorization"]),
      ["Bearer token-1", "Bearer token-2"],
    );

    // The fresh one is kept for what comes next.
    await api.get("/v1/agents/sessions/{id}", { path: { id: "sess_1" } });
    assert.equal(tokens.asked(), 2);
  });

  it("is not asked a third time when the fresh token is refused too", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 401,
      body: { error: { message: "not a token of this app" } },
    });
    const tokens = counting();
    const api = new Client({ url: router.url, apiKey: "vak_live_x" });
    await api.setUser("jlahey", tokens.source);

    await assert.rejects(
      () => api.post("/v1/agents/sessions", { body: { agent: "john" } }),
      (raised: RouterError) => raised.status === 401,
    );
    assert.equal(tokens.asked(), 2);
    assert.equal(router.received.length, 2);
    assert.deepEqual(router.received[1]?.body, { agent: "john" }, "the body is sent again");
  });

  it("asks nothing again for a token that was handed in as it is", async () => {
    router.serve("GET", "/v1/agents/sessions/sess_1", { status: 401, body: {} });
    const api = new Client({ url: router.url, apiKey: "vak_live_x" });
    await api.setUser("jlahey", "token-for-jim");

    await assert.rejects(
      () => api.get("/v1/agents/sessions/{id}", { path: { id: "sess_1" } }),
      (raised: RouterError) => raised.status === 401,
    );
    assert.equal(router.received.length, 1);
  });

  it("does not retry a router that is never shown the token", async () => {
    // Reached by customer id, the token is for Stream's own clients, so a fresh one would
    // send the very same request again.
    router.serve("GET", "/v1/agents/sessions/sess_1", { status: 401, body: {} });
    const tokens = counting();
    const api = new Client({ url: router.url, customerId: "local", apiKey: "vak_live_x" });
    await api.setUser("jlahey", tokens.source);

    await assert.rejects(() => api.get("/v1/agents/sessions/{id}", { path: { id: "sess_1" } }));
    assert.equal(router.received.length, 1);
    assert.equal(tokens.asked(), 0);
  });

  it("is asked again before a socket is opened with a token about to expire", async () => {
    const now = Math.floor(Date.now() / 1000);
    const tokens = [
      await signToken({ user_id: "jlahey", exp: now + 10 }, "shh"),
      await signToken({ user_id: "jlahey", exp: now + 3600 }, "shh"),
    ];
    let asked = 0;
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x" });
    await backend.setUser("jlahey", () => tokens[asked++] ?? "");

    await backend.headers();
    const url = new URL(await backend.socketURL("/v1/agents/sessions/x/events"));

    assert.equal(asked, 2, "the first expires within the margin");
    assert.equal(url.searchParams.get("token"), tokens[1]);
    await backend.headers();
    assert.equal(asked, 2, "the second is kept");
  });

  it("is asked again after a 401 through Stream's proxy, the hosted router's own way", async () => {
    router.serve("GET", "/v1/agents/sessions/sess_1", (request) =>
      request.headers["authorization"] === "Bearer token-1"
        ? { status: 401, body: {} }
        : { body: { id: "sess_1" } },
    );
    const tokens = counting();
    const api = new Client({ url: router.url, apiKey: "vak_live_x", authenticate: true });
    await api.setUser("jlahey", tokens.source);

    await api.get("/v1/agents/sessions/{id}", { path: { id: "sess_1" } });

    assert.equal(tokens.asked(), 2);
    assert.deepEqual(
      router.received.map((r) => [r.headers["api_key"], r.headers["authorization"]]),
      [
        ["vak_live_x", "Bearer token-1"],
        ["vak_live_x", "Bearer token-2"],
      ],
    );
  });

  it("is asked once more, not once per request, when requests are refused together", async () => {
    router.serve("GET", "/v1/agents/sessions/sess_1", (request) =>
      request.headers["authorization"] === "Bearer token-1"
        ? { status: 401, body: {} }
        : { body: { id: "sess_1" } },
    );
    const tokens = counting();
    const api = new Client({ url: router.url, apiKey: "vak_live_x" });
    await api.setUser("jlahey", tokens.source);
    await api.backend.headers();

    const sessions = await Promise.all(
      [1, 2, 3, 4].map(() => api.get("/v1/agents/sessions/{id}", { path: { id: "sess_1" } })),
    );

    assert.equal(sessions.length, 4);
    assert.equal(tokens.asked(), 2, "token-1, then one token-2 for every retry");
  });

  it("keeps a newer token when a refusal of an older one arrives late", async () => {
    const tokens = counting();
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x" });
    await backend.setUser("jlahey", tokens.source);

    await backend.headers();
    assert.equal(backend.expireToken("token-1"), true);
    assert.equal((await backend.headers())["Authorization"], "Bearer token-2");

    // A request sent with token-1 before the refresh is refused only now.
    assert.equal(backend.expireToken("token-1"), true, "the retry goes with token-2");
    assert.equal((await backend.headers())["Authorization"], "Bearer token-2");
    assert.equal(tokens.asked(), 2);
  });

  it("dials a refused socket once more with a fresh token", async () => {
    router.refuseSockets((query) => query.get("token") === "token-1");
    const tokens = counting();
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x" });
    await backend.setUser("jlahey", tokens.source);
    await backend.headers();

    const opening = Socket.dial(backend, "/v1/agents/sessions/x/events");
    const connection = await router.socket();
    const socket = await opening;

    assert.equal(connection.query.get("token"), "token-2");
    assert.equal(tokens.asked(), 2);
    socket.close();
  });

  it("does not dial a refused socket again with a token handed in as it is", async () => {
    router.refuseSockets(() => true);
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x" });
    await backend.setUser("jlahey", "token-for-jim");

    await assert.rejects(() => Socket.dial(backend, "/v1/agents/sessions/x/events"), SocketClosedError);
  });

  it("leaves the credentials out of the error a refused socket raises, since errors end up in logs", async () => {
    router.refuseSockets(() => true);
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x" });
    await backend.setUser("jlahey", "token-for-jim");

    await assert.rejects(() => Socket.dial(backend, "/v1/agents/sessions/x/events"), (raised: unknown) => {
      assert.ok(raised instanceof SocketClosedError);
      assert.match(raised.message, /\/v1\/agents\/sessions\/x\/events/, "still says where");
      assert.doesNotMatch(raised.message, /token-for-jim/);
      assert.doesNotMatch(raised.message, /vak_live_x|api_key|token=/);
      return true;
    });
  });

  it("signs the server's own token once and sends it until it is about to expire", async (t) => {
    t.mock.timers.enable({ apis: ["Date"], now: Date.now() });
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x", apiSecret: "secret" });

    const first = (await backend.headers())["Authorization"];
    // Long enough that a token signed again would carry another iat.
    t.mock.timers.tick(5 * 60 * 1000);
    const second = (await backend.headers())["Authorization"];
    const socket = new URL(await backend.socketURL("/v1/dispatch")).searchParams.get("token");

    assert.ok(first);
    assert.equal(second, first);
    assert.equal(`Bearer ${socket}`, first);
  });

  it("signs a new server token once the one it holds is about to expire", async (t) => {
    t.mock.timers.enable({ apis: ["Date"], now: Date.now() });
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x", apiSecret: "secret" });
    const first = (await backend.headers())["Authorization"] ?? "";

    t.mock.timers.tick(60 * 60 * 1000 - 30_000);
    const later = (await backend.headers())["Authorization"] ?? "";

    assert.notEqual(later, first);
    const expiry = claimsOf(later.replace(/^Bearer /, ""))["exp"] as number;
    assert.ok(expiry * 1000 - Date.now() > 60 * 60 * 1000 - 5_000, "the new one runs for a fresh hour");
  });

  it("is not shared with a copy acting for another user", async () => {
    let release: (token: string) => void = () => {};
    let asked = 0;
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x", authenticate: true });
    await backend.setUser("jlahey", () => {
      asked += 1;
      return asked === 1 ? new Promise<string>((resolve) => (release = resolve)) : `token-${asked}`;
    });

    const pending = backend.headers();
    const acting = backend.actingFor("ana");
    release("token-1");
    await pending;

    // The copy asks for its own rather than holding the original's fetch forever.
    assert.equal((await acting.headers())["Authorization"], "Bearer token-2");
    assert.equal(acting.expireToken("token-2"), true);
    assert.equal((await acting.headers())["Authorization"], "Bearer token-3");
  });

  it("hands Stream's clients the fetch already under way rather than starting another", async () => {
    let release: (token: string) => void = () => {};
    let asked = 0;
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x" });
    await backend.setUser("jlahey", () => {
      asked += 1;
      return new Promise<string>((resolve) => (release = resolve));
    });

    const request = backend.headers();
    const chat = backend.streamCredentials(true);
    release("token-1");

    assert.equal((await chat)?.token, "token-1");
    assert.equal((await request)["Authorization"], "Bearer token-1");
    assert.equal(asked, 1);
  });

  it("is forgotten with the user it was for", async () => {
    let release: (token: string) => void = () => {};
    const backend = new Backend({ url: router.url, apiKey: "vak_live_x" });
    await backend.setUser("jlahey", () => new Promise<string>((resolve) => (release = resolve)));

    const pending = backend.headers();
    await backend.setUser("ana", "token-for-ana");
    release("token-for-jim");
    await pending;

    assert.equal((await backend.headers())["Authorization"], "Bearer token-for-ana");
  });
});
