import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import type { Duplex } from "node:stream";
import { afterEach, beforeEach, describe, it } from "node:test";

import {
  Agent,
  Backend,
  Client,
  ConfigurationError,
  Dispatch,
  RouterError,
  Session,
  Socket,
  type Folder,
  type GuestStore,
  type Schemas,
} from "../src/index.js";
import { loadFolder, parseDeclaration, parseSkill } from "../src/node.js";
import * as stream from "../src/stream.js";
import { TestRouter } from "./router.js";

const row: Schemas["Session"] = {
  id: "session-1",
  user_id: "user-1",
  agent_id: "support",
  call_id: "",
  call_type: "",
  modality: "text",
  state: "live",
  created_at: "2026-01-01T00:00:00Z",
};

describe("production hardening", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local", maxRetries: 0 });
    router.serve("POST", "/v1/agents/sessions", { body: row });
    router.serve("GET", "/v1/agents/sessions/session-1", { body: row });
    router.serve("POST", "/v1/agents/sessions/session-1/stop", { status: 204 });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("opens conversations under the configuration the agent synced", async () => {
    router.serve("POST", "/v1/agents/sync", { body: { config: { id: "cfg-1", name: "support" } } });
    const agent = new Agent({ client: api, name: "support", instructions: "Follow the policy." });
    await agent.sync();
    const session = await agent.chat({ watch: false });
    assert.equal((router.last.body as Schemas["CreateSessionRequest"]).agent, "support");
    await session.close();
  });

  it("preserves an existing conversation when its watcher cannot connect", async () => {
    router.refuseSockets(() => true);
    await assert.rejects(() => api.agent("support").sessions.resume(row.id));
    assert.equal(router.requestsTo("POST", "/v1/agents/sessions/session-1/stop").length, 0);
  });

  it("stops a remote conversation even after its watcher was lost", async () => {
    const session = await Session.open(api, {});
    const connection = await router.socket();
    connection.socket.close(1011, "router restarting");
    await session.wait();
    await session.close();
    assert.equal(router.requestsTo("POST", "/v1/agents/sessions/session-1/stop").length, 1);
    await session.close();
    assert.equal(router.requestsTo("POST", "/v1/agents/sessions/session-1/stop").length, 1);
  });

  it("reports a failed HTTP shutdown and lets the caller try again", async () => {
    const session = await Session.open(api, {}, { watch: false });
    router.serve("POST", "/v1/agents/sessions/session-1/stop", {
      status: 503,
      body: { error: { message: "temporarily unavailable", code: "unavailable" } },
    });
    await assert.rejects(() => session.close(), (cause: unknown) => {
      assert.ok(cause instanceof RouterError);
      assert.equal(cause.status, 503);
      return true;
    });
    assert.equal(session.live, true);
    router.serve("POST", "/v1/agents/sessions/session-1/stop", { status: 204 });
    await session.close();
  });

  it("ends unwatched sessions locally after a successful HTTP shutdown", async () => {
    const session = await Session.open(api, {}, { watch: false });
    await session.close();
    assert.equal(session.live, false);
    assert.equal(session.state.get().kind, "ended");
    const events = [];
    for await (const event of session.events()) {
      events.push(event);
    }
    assert.deepEqual(events, []);
  });

  it("accepts shutdown when the router already stopped the session", async () => {
    const session = await Session.open(api, {}, { watch: false });
    router.serve("POST", "/v1/agents/sessions/session-1/stop", {
      status: 404,
      body: { error: { message: "no such session", code: "session_not_found" } },
    });
    await session.close();
    assert.equal(session.live, false);
    assert.equal(session.state.get().kind, "ended");
  });

  it("refuses two hosted agents whose tool names would select the wrong implementation", () => {
    const first = api.agent("first");
    const second = api.agent("second");
    first.tools.register({ name: "lookup", description: "First lookup", run: () => "first" });
    second.tools.register({ name: "lookup", description: "Second lookup", run: () => "second" });
    const worker = new Dispatch({ client: api }).host(first);
    assert.throws(() => worker.host(second), ConfigurationError);
  });

  it("also refuses hosted tool collisions introduced after hosting", async () => {
    const first = api.agent("first");
    const second = api.agent("second");
    first.tools.register({ name: "lookup", description: "First lookup", run: () => "first" });
    const worker = new Dispatch({ client: api }).host(first).host(second);
    second.tools.register({ name: "lookup", description: "Second lookup", run: () => "second" });
    await assert.rejects(() => worker.run(), ConfigurationError);
  });

  it("finishes a pending tool hold when the worker is aborted", async () => {
    const agent = api.agent("support");
    let aborted: boolean | undefined;
    agent.tools.register({
      name: "write",
      description: "Write after holding the call",
      run: async (_input, context) => {
        aborted = await context.disallowInterruptions().then(
          () => false,
          () => true,
        );
        return "finished";
      },
    });
    const stop = new AbortController();
    const worker = new Dispatch({ client: api }).host(agent);
    const running = worker.run(stop.signal);
    const connection = await router.socket();
    connection.send({ type: "ready" });
    await connection.next();
    connection.send({ type: "tool_call", id: "call-1", session_id: row.id, name: "write", arguments: "{}" });
    assert.equal((await connection.next())["type"], "tool_hold");
    stop.abort();
    await running;
    assert.equal(aborted, true);
  });

  it("never refreshes an old Stream connection with another user's token", async () => {
    const backend = new Backend({ url: router.url, apiKey: "key", apiSecret: "", customerId: "" });
    await backend.setUser("first", () => "first-token");
    const credentials = await backend.streamCredentials();
    assert.ok(credentials);
    const tokens = stream.streamTokens(backend, credentials, "chat");
    assert.equal(await tokens(), "first-token");
    await backend.setUser("second", () => "second-token");
    await assert.rejects(tokens, ConfigurationError);
  });

  it("resyncs when the destination fingerprint differs despite a matching local stamp", async () => {
    let hash = "";
    const folder: Folder = {
      path: "support", name: "support", instructions: "current policy", guardrail: "",
      skills: [], knowledge: [], knowledgeURLs: [],
      stamp: { read: async () => hash, write: async (value) => { hash = value; } },
    };
    router.serve("POST", "/v1/agents/sync", { body: { unchanged: false, config: { id: "cfg", name: "support" } } });
    const agent = new Agent({ folder, client: api });
    await agent.sync();
    router.serve("GET", "/v1/agents/configs", {
      body: [{ id: "cfg", name: "support", instructions: "old policy", sync_hash: "different" }],
    });
    const result = await agent.sync();
    assert.equal(result.unchanged, false);
    assert.equal(router.requestsTo("POST", "/v1/agents/sync").length, 2);
  });

  it("retries a safe request whose response body was truncated", async () => {
    const retrying = new Client({ url: router.url, customerId: "local", maxRetries: 1 });
    router.serve("GET", "/v1/agents/configs", (_, calls) => calls === 0 ? {
      text: "[",
      headers: { "Content-Type": "application/json", "Content-Length": "100", Connection: "close" },
    } : { body: [{ id: "recovered" }] });
    assert.equal((await retrying.get("/v1/agents/configs"))[0]?.id, "recovered");
    assert.equal(router.requestsTo("GET", "/v1/agents/configs").length, 2);
  });

  it("keeps request metadata when a response cannot be read and retries are exhausted", async () => {
    router.serve("GET", "/v1/agents/configs", {
      text: "[",
      headers: { "Content-Type": "application/json", "Content-Length": "100", Connection: "close", "X-Request-Id": "truncated-1" },
    });
    await assert.rejects(() => api.get("/v1/agents/configs"), (cause: unknown) => {
      assert.ok(cause instanceof RouterError);
      assert.equal(cause.status, 0);
      assert.equal(cause.operation, "GET /v1/agents/configs");
      assert.equal(cause.requestId, "truncated-1");
      assert.ok(cause.cause instanceof Error);
      return true;
    });
  });

  it("reports malformed JSON with request context without retrying an invalid response", async () => {
    const retrying = new Client({ url: router.url, customerId: "local", maxRetries: 2 });
    router.serve("GET", "/v1/agents/configs", {
      text: "not JSON", headers: { "Content-Type": "application/json", "X-Request-Id": "invalid-1" },
    });
    await assert.rejects(() => retrying.get("/v1/agents/configs"), (cause: unknown) => {
      assert.ok(cause instanceof RouterError);
      assert.equal(cause.status, 200);
      assert.equal(cause.requestId, "invalid-1");
      assert.ok(cause.cause instanceof SyntaxError);
      return true;
    });
    assert.equal(router.requestsTo("GET", "/v1/agents/configs").length, 1);
  });

  it("reads text exports without attempting to parse their lines as one JSON value", async () => {
    const text = '{"table":"configs"}\n{"table":"skills"}\n';
    router.serve("GET", "/v1/data/export", { text });
    const exported: string = await api.get("/v1/data/export");
    assert.equal(exported, text);
    router.serve("GET", "/v1/data/export", { text: "" });
    assert.equal(await api.get("/v1/data/export"), "");
  });

  it("returns an event stream as bytes instead of parsing it as JSON", async () => {
    router.serve("GET", "/v1/agents/logs/stream", {
      text: 'event: logs\ndata: {"message":"ready"}\n\n',
      headers: { "Content-Type": "text/event-stream" },
    });
    const events: ReadableStream<Uint8Array> = await api.get("/v1/agents/logs/stream");
    const reader = events.getReader();
    assert.match(new TextDecoder().decode((await reader.read()).value), /event: logs/);
    await reader.cancel();
  });

  it("honours an explicitly empty skill list over the folder's skills", async () => {
    router.serve("POST", "/v1/agents/sync", { body: { config: { id: "cfg", name: "support" } } });
    const folder: Folder = {
      path: "support", name: "support", instructions: "policy", guardrail: "",
      skills: [{ name: "research", description: "Research", instructions: "Investigate" }],
      knowledge: [], knowledgeURLs: [],
    };
    await new Agent({ folder, client: api, harness: { skills: [] } }).sync();
    assert.deepEqual((router.last.body as Schemas["SyncAgentRequest"]).skills ?? [], []);
  });

  it("shares a guest refresh rather than spending the same secret twice", async () => {
    let stored = JSON.stringify({ id: "returning", token: "expired", expires_at: "2000-01-01T00:00:00Z", refresh_secret: "one-use" });
    const store: GuestStore = { read: () => stored, write: (value) => { stored = value; }, clear: () => { stored = ""; } };
    router.serve("POST", "/v1/agents/guests", (_, calls) => ({ body: {
      id: calls === 0 ? "returning" : "stranger", token: "fresh", refresh_secret: "next",
    } }));
    const guests = await Promise.all([api.guestUser({}, store), api.guestUser({}, store)]);
    assert.deepEqual(guests.map((guest) => guest.id), ["returning", "returning"]);
    assert.equal(router.requestsTo("POST", "/v1/agents/guests").length, 1);
    assert.equal(JSON.parse(stored).id, "returning");
  });

  it("closes an unread socket instead of retaining an unbounded event queue", async () => {
    const socket = await Socket.dial(api.backend, "/v1/dispatch");
    const connection = await router.socket();
    const closed = once(connection.socket, "close");
    for (let index = 0; index < 1100; index++) {
      connection.send({ type: "event", index });
    }
    await closed;
    assert.equal(socket.open, false);
    assert.match(socket.closedBy?.message ?? "", /reader fell behind/);
    const messages = [];
    for await (const message of socket.messages()) {
      messages.push(message);
    }
    assert.deepEqual(messages, []);
  });
});

describe("folder validation", () => {
  it("refuses an unreadable guardrail instead of replacing its policy with empty text", async () => {
    const root = await mkdtemp("/tmp/stream-agents-guardrail-");
    try {
      await writeFile(`${root}/agent.yaml`, "name: support\n");
      await mkdir(`${root}/guardrail.md`);
      await assert.rejects(() => loadFolder(root), (cause: unknown) => {
        assert.ok(cause instanceof Error && "code" in cause);
        assert.equal(cause.code, "EISDIR");
        return true;
      });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("preserves hashes inside quoted declaration values", () => {
    for (const quote of ['"', "'"]) {
      assert.equal(parseDeclaration(`name: ${quote}Support #1${quote} # comment`).name, "Support #1");
    }
  });

  it("rejects duration strings containing unmatched fragments", () => {
    for (const deadline of ["broken1sjunk", "-1s", "1s junk", "1m2"]) {
      assert.throws(() => parseSkill("work", `---\ndescription: Work\ndeadline: ${deadline}\n---\nDo work.`), ConfigurationError);
    }
    assert.equal(parseSkill("work", "---\ndescription: Work\ndeadline: 1m500ms\n---\nDo work.").deadlineMs, 60_500);
  });
});

describe("socket establishment", () => {
  it("cancels a stalled upgrade when dispatch is stopped", async (t) => {
    const server = createServer();
    let raw: Duplex | undefined;
    const upgraded = new Promise<void>((resolve) => server.on("upgrade", (_, socket) => {
      raw = socket;
      socket.resume();
      resolve();
    }));
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    t.after(async () => {
      raw?.destroy();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    });
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    const worker = new Dispatch({ client: { url: `http://127.0.0.1:${address.port}`, customerId: "local" } }).onCall(() => undefined);
    const running = worker.run();
    await upgraded;
    const disconnected = once(raw!, "end");
    worker.stop();
    await running;
    await disconnected;
  });
});
