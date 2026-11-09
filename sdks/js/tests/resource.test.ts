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
import { inPage, offPage } from "./device.js";
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
    modality: "text",
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

  it("updates a config by sending only what changed, so the rest of it is kept", async () => {
    router.serve("GET", "/v1/agents/configs", { body: [{ id: "config-1", name: "docs" }] });
    router.serve("PATCH", "/v1/agents/configs/config-1", {
      body: { id: "config-1", name: "docs", guardrail: "Only Acme.", instructions: "Be brief." },
    });

    const updated = await api.agent("docs").updateConfig({ guardrail: "Only Acme." });

    assert.equal(updated.instructions, "Be brief.");
    assert.deepEqual(router.requestsTo("PATCH", "/v1/agents/configs/config-1")[0]?.body, {
      guardrail: "Only Acme.",
    });
  });

  it("refuses to update an agent nothing is configured under", async () => {
    router.serve("GET", "/v1/agents/configs", { body: [] });

    await assert.rejects(
      () => api.agent("docs").updateConfig({ guardrail: "Only Acme." }),
      ConfigurationError,
    );
  });
});

describe("sessions", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
    router.serve("POST", "/v1/agents/sessions/session-1/stop", { status: 204 });
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

  it("names the agent and holds the conversation in writing unless voice was asked for", async () => {
    const { session: held } = await open({ title: "Is Stream better than Sendbird" });

    const body = router.received[0]?.body as Record<string, unknown>;
    assert.equal(body["agent"], "docs");
    assert.equal(body["start_voice"], undefined, "a session resource is a conversation");
    assert.equal(body["title"], "Is Stream better than Sendbird");
    assert.equal(held.conversationId, "agent:support-7");
    assert.equal(held.voice.started, false);
    await held.close();
  });

  it("starts voice as it opens when asked, on the session's own call", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: session({ id: "chosen-id_1", call_id: "chosen-id_1" }),
    });
    const opening = api.agent("docs").sessions.create({ id: "chosen-id_1", start_voice: true });
    await router.socket();
    const held = await opening;

    const body = router.received[0]?.body as Record<string, unknown>;
    assert.equal(body["start_voice"], true);
    assert.equal(body["id"], "chosen-id_1", "a session id is whatever the caller picks");
    assert.equal(held.voice.started, true);
    await held.close();
  });

  it("resumes a conversation by the id of the session it was held in", async () => {
    router.serve("GET", "/v1/agents/sessions/session-4", {
      body: session({ id: "session-4", conversation_id: "agent:support-4", state: "ended" }),
    });
    const resuming = api.agent("docs").sessions.resume("session-4");
    const connection = await router.socket();
    const held = await resuming;

    assert.equal(held.id, "session-4");
    assert.equal(held.conversationId, "agent:support-4");
    assert.equal(connection.path, "/v1/agents/sessions/session-4/events");
    assert.equal(
      router.received.some((one) => one.method === "POST" && one.path === "/v1/agents/sessions"),
      false,
      "carrying a conversation on creates no session",
    );
    await held.close();
  });

  it("spells the model overwrites the way the wire does", async () => {
    const { session: held } = await open({ modelOverwrites: { thinking: "high" } });

    const body = router.received[0]?.body as Record<string, unknown>;
    assert.deepEqual(body["model_overwrites"], { thinking: "high" });
    await held.close();
  });

  /** The bodies of every session the test opened, in order. */
  function created(): Record<string, unknown>[] {
    return router
      .requestsTo("POST", "/v1/agents/sessions")
      .map((one) => one.body as Record<string, unknown>);
  }

  it("opens a session from a page in the device's own zone", async () => {
    await inPage("Europe/Lisbon", async () => {
      const { session: held } = await open();
      await held.close();
    });
    await inPage("UTC", async () => {
      const { session: held } = await open();
      await held.close();
    });

    assert.deepEqual(
      created().map((body) => body["timezone"]),
      ["Europe/Lisbon", "UTC"],
    );
  });

  it("sends the app's own zone over the device's", async () => {
    await inPage("Europe/Lisbon", async () => {
      const { session: held } = await open({ timezone: "America/New_York" });
      await held.close();
    });

    assert.equal(created()[0]?.["timezone"], "America/New_York");
  });

  it("sends no zone from a page that opts out with an empty one", async () => {
    await inPage("Europe/Lisbon", async () => {
      const { session: held } = await open({ timezone: "" });
      await held.close();
    });

    assert.equal("timezone" in (created()[0] ?? {}), false);
  });

  it("sends no zone when what the browser reports does not read as one, so the session still opens", async () => {
    const reported = [undefined, "", "GMT+05:30", "+05:30", "Etc/Unknown", 42, new RangeError("no ICU")];
    for (const zone of reported) {
      await inPage(zone, async () => {
        const { session: held } = await open();
        await held.close();
      });
    }

    const bodies = created();
    assert.equal(bodies.length, reported.length);
    for (const [index, body] of bodies.entries()) {
      assert.equal("timezone" in body, false, `${String(reported[index])} was sent`);
    }
  });

  it("reads no zone off a machine that is not a page, whatever it holds", async () => {
    const asUser = new Client({ url: router.url, apiKey: "key", token: "token-for-jlahey" });
    await offPage("Europe/Lisbon", async () => {
      const { session: asBackend } = await open();
      await asBackend.close();

      router.serve("POST", "/v1/agents/sessions", { status: 201, body: session() });
      const opening = asUser.agent("docs").sessions.create();
      await router.socket();
      await (await opening).close();
    });

    const bodies = created();
    assert.equal(bodies.length, 2);
    for (const body of bodies) {
      assert.equal("timezone" in body, false, "a server's zone is not the caller's");
    }
  });

  it("sends a zone the app names off a page, too", async () => {
    await offPage("Europe/Lisbon", async () => {
      const { session: held } = await open({ timezone: "Asia/Tokyo" });
      await held.close();
    });

    assert.equal(created()[0]?.["timezone"], "Asia/Tokyo");
  });

  it("puts a query on the wire as the filter the router reads", async () => {
    router.serve("POST", "/v1/agents/sessions/query", {
      body: { items: [session()], has_more: false },
    });

    await api.agent("docs").sessions.query({
      projectId: "health",
      modality: "voice",
      state: "live",
      agentId: "support-7",
      limit: 10,
      cursor: "page-2",
    });

    assert.deepEqual(
      router.last.body,
      {
        filter: {
          agent: "docs",
          project_id: "health",
          modality: "voice",
          state: "live",
          agent_id: "support-7",
        },
        limit: 10,
        cursor: "page-2",
      },
      "an agent's sessions are the agent's own, and a filter nobody set is not sent empty",
    );
  });

  it("searches through the same query, as a text match", async () => {
    router.serve("POST", "/v1/agents/sessions/query", {
      body: { items: [session()], has_more: false },
    });

    await api.agent("docs").sessions.search("sendbird", { userId: "ana" });

    assert.deepEqual(router.last.body, {
      filter: { agent: "docs", user_id: "ana", text: { $q: "sendbird" } },
    });
  });

  it("stops an unwatched session on close rather than deleting it, so what it said is kept", async () => {
    router.serve("POST", "/v1/agents/sessions", { status: 201, body: session() });
    router.serve("POST", "/v1/agents/sessions/session-1/stop", { status: 204 });
    const held = await api.agent("docs").sessions.create({ watch: false });

    await held.close();

    assert.equal(router.requestsTo("POST", "/v1/agents/sessions/session-1/stop").length, 1);
    assert.equal(router.requestsTo("DELETE", "/v1/agents/sessions/session-1").length, 0);
  });

  it("deletes a conversation, held or not, only when asked", async () => {
    const { session: held } = await open();
    router.serve("DELETE", "/v1/agents/sessions/session-1", { status: 204 });
    router.serve("DELETE", "/v1/agents/sessions/session-9", { status: 204 });

    await held.delete();
    await api.agent("docs").sessions.delete("session-9");

    assert.equal(router.requestsTo("DELETE", "/v1/agents/sessions/session-1").length, 1);
    assert.equal(router.requestsTo("DELETE", "/v1/agents/sessions/session-9").length, 1);
    await held.close();
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
      body: { error: { message: "unknown session", type: "not_found", code: "not_found" } },
    });

    await assert.rejects(
      api.agent("docs").sessions.deleteMemories("someone-elses"),
      (error: unknown) => error instanceof RouterError && error.status === 404,
    );
  });

  it("filters a query by when sessions were made and by their labels, and sorts it", async () => {
    router.serve("POST", "/v1/agents/sessions/query", {
      body: { items: [session()], has_more: false },
    });

    await api.agent("docs").sessions.query({
      createdSince: new Date("2026-10-01T00:00:00Z"),
      createdBefore: new Date("2026-11-01T00:00:00Z"),
      custom: { team: "support" },
      sort: "updated_at",
    });

    assert.deepEqual(router.last.body, {
      filter: {
        agent: "docs",
        created_at: { $gte: "2026-10-01T00:00:00.000Z", $lt: "2026-11-01T00:00:00.000Z" },
        custom: { team: "support" },
      },
      sort: [{ field: "updated_at" }],
    });
  });

  it("sends one end of a date range alone, and no empty labels", async () => {
    router.serve("POST", "/v1/agents/sessions/query", {
      body: { items: [], has_more: false },
    });

    await api.agent("docs").sessions.search("lisbon", {
      createdSince: new Date("2026-10-01T00:00:00Z"),
      custom: {},
      sort: "relevance",
    });

    assert.deepEqual(router.last.body, {
      filter: {
        agent: "docs",
        created_at: { $gte: "2026-10-01T00:00:00.000Z" },
        text: { $q: "lisbon" },
      },
      sort: [{ field: "relevance" }],
    });
  });

  it("renames, voices, stops and forks a session it is not holding, by its id", async () => {
    router.serve("PATCH", "/v1/agents/sessions/session-9", { body: session({ id: "session-9", title: "Trip" }) });
    router.serve("POST", "/v1/agents/sessions/session-9/voice", { body: session({ id: "session-9", call_id: "c" }) });
    router.serve("DELETE", "/v1/agents/sessions/session-9/voice", { body: session({ id: "session-9" }) });
    router.serve("POST", "/v1/agents/sessions/session-9/stop", { status: 204 });
    router.serve("POST", "/v1/agents/sessions/session-9/fork", {
      status: 201,
      body: session({ id: "session-10", conversation_id: "agent:session-10" }),
    });
    const sessions = api.agent("docs").sessions;

    const renamed = await sessions.update("session-9", { title: "Trip" });
    await sessions.startVoice("session-9");
    await sessions.stopVoice("session-9");
    await sessions.close("session-9");
    const forking = sessions.fork("session-9", { title: "Trip, again", modelOverwrites: { llm: "openai/gpt-5" } });
    const connection = await router.socket();
    const forked = await forking;

    assert.equal(renamed.title, "Trip");
    assert.deepEqual(router.requestsTo("PATCH", "/v1/agents/sessions/session-9")[0]?.body, { title: "Trip" });
    assert.equal(router.requestsTo("POST", "/v1/agents/sessions/session-9/voice").length, 1);
    assert.equal(router.requestsTo("DELETE", "/v1/agents/sessions/session-9/voice").length, 1);
    assert.equal(router.requestsTo("POST", "/v1/agents/sessions/session-9/stop").length, 1);
    assert.deepEqual(router.requestsTo("POST", "/v1/agents/sessions/session-9/fork")[0]?.body, {
      title: "Trip, again",
      model_overwrites: { llm: "openai/gpt-5" },
    });
    assert.equal(forked.id, "session-10");
    assert.equal(connection.path, "/v1/agents/sessions/session-10/events");
    await forked.close();
  });

  it("remembers for the user a page signed in as, whoever the page names", async () => {
    const page = new Client({ url: router.url, apiKey: "key" });
    await page.setUser("jlahey", "user-token");
    router.serve("POST", "/v1/agents/sessions", { status: 201, body: session() });

    const held = await page.agent("docs").sessions.create({
      watch: false,
      memory: { user_id: "someone-else", app_id: "assistant" },
    });

    const body = router.requestsTo("POST", "/v1/agents/sessions")[0]?.body as Record<string, unknown>;
    assert.deepEqual(body["memory"], { user_id: "jlahey", app_id: "assistant" });
    await held.close();
  });

  it("lets a backend remember for the user it names, or the one it acts for", async () => {
    router.serve("POST", "/v1/agents/sessions", { status: 201, body: session() });

    const held = await api.agent("docs").sessions.create({ watch: false, memory: { user_id: "ana" } });

    const body = router.requestsTo("POST", "/v1/agents/sessions")[0]?.body as Record<string, unknown>;
    assert.deepEqual(body["memory"], { user_id: "ana" });
    await held.close();
  });

  it("remembers for the user a backend acts for when it names none", async () => {
    const acting = new Client(api.backend.actingFor("ana"));
    router.serve("POST", "/v1/agents/sessions", { status: 201, body: session() });

    const held = await acting.agent("docs").sessions.create({ watch: false, memory: {} });

    const body = router.requestsTo("POST", "/v1/agents/sessions")[0]?.body as Record<string, unknown>;
    assert.deepEqual(body["memory"], { user_id: "ana" });
    await held.close();
  });

  it("names where a sign-in goes back to in the spelling the phones use", async () => {
    router.serve("POST", "/v1/agents/sessions", { status: 201, body: session() });

    const held = await api.agent("docs").sessions.create({ watch: false, returnUrl: "http://localhost:5173/connected.html" });

    const body = router.requestsTo("POST", "/v1/agents/sessions")[0]?.body as Record<string, unknown>;
    assert.equal(body["return_url"], "http://localhost:5173/connected.html");
    assert.equal(body["returnUrl"], undefined);
    await held.close();
  });

  it("refuses memory for nobody before asking the router", async () => {
    await assert.rejects(api.agent("docs").sessions.create({ memory: {} }), ConfigurationError);
    assert.equal(router.requestsTo("POST", "/v1/agents/sessions").length, 0);
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

describe("simulations", () => {
  let router: TestRouter;
  let api: Client;

  const request: Schemas["SimulationRequest"] = {
    name: "capital",
    config_id: "config-1",
    scenario: "Ask what the capital of France is.",
    assertion: "The agent says Paris.",
  };

  const simulation: Schemas["Simulation"] = {
    ...request,
    id: "simulation-1",
    mode: "text",
    variations: 1,
    max_turns: 12,
    created_at: new Date().toISOString(),
  };

  function run(over: Partial<Schemas["SimulationRun"]> = {}): Schemas["SimulationRun"] {
    return {
      id: "run-1",
      simulation_id: "simulation-1",
      state: "running",
      cases: 1,
      passed: 0,
      failed: 0,
      started_at: new Date().toISOString(),
      ...over,
    };
  }

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("creates a simulation from the request as written", async () => {
    router.serve("POST", "/v1/agents/simulations", { status: 201, body: simulation });

    const created = await api.simulations.create(request);

    assert.equal(created.id, "simulation-1");
    assert.deepEqual(router.last.body, request, "nothing the caller left out is filled in");
  });

  it("replaces a simulation in place, by its id", async () => {
    router.serve("PUT", "/v1/agents/simulations/simulation-1", {
      body: { ...simulation, assertion: "The agent says Paris, France." },
    });

    const updated = await api.simulations.update("simulation-1", {
      ...request,
      assertion: "The agent says Paris, France.",
    });

    assert.equal(updated.assertion, "The agent says Paris, France.");
    assert.equal((router.last.body as Record<string, unknown>)["assertion"], "The agent says Paris, France.");
  });

  it("starts a run and reads it until it settles", async () => {
    router.serve("POST", "/v1/agents/simulations/simulation-1/run", { status: 202, body: run() });
    router.serve("GET", "/v1/agents/simulation-runs/run-1", { body: run({ state: "passed", passed: 1 }) });

    const started = await api.simulations.run("simulation-1");
    const settled = await api.simulations.runs.get(started.id);

    assert.equal(started.state, "running", "a run answers before its conversations are had");
    assert.equal(settled.state, "passed");
    assert.equal(settled.passed, 1);
  });

  it("lists one simulation's runs, and leaves out a filter nobody set", async () => {
    router.serve("GET", "/v1/agents/simulation-runs", { body: [run()] });

    await api.simulations.runs.list({ simulationId: "simulation-1" });
    assert.equal(router.last.query.get("simulation_id"), "simulation-1");
    assert.equal(router.last.query.get("limit"), null);

    await api.simulations.runs.list();
    assert.equal(router.last.query.get("simulation_id"), null);
  });

  it("cancels a run and deletes a simulation", async () => {
    router.serve("POST", "/v1/agents/simulation-runs/run-1/cancel", { body: run({ state: "cancelled" }) });
    router.serve("DELETE", "/v1/agents/simulations/simulation-1", { status: 204 });

    const cancelled = await api.simulations.runs.cancel("run-1");
    await api.simulations.delete("simulation-1");

    assert.equal(cancelled.state, "cancelled");
    assert.equal(router.requestsTo("DELETE", "/v1/agents/simulations/simulation-1").length, 1);
  });

  it("says why the router would not run a simulation", async () => {
    router.serve("POST", "/v1/agents/simulations/gone/run", {
      status: 404,
      body: { error: { message: "unknown simulation", type: "not_found", code: "not_found" } },
    });

    await assert.rejects(
      api.simulations.run("gone"),
      (error: unknown) => error instanceof RouterError && error.status === 404,
    );
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
    assert.match(
      String(body["request_id"]),
      /^[A-Za-z0-9_-]{1,128}$/,
      "a session keeping no conversation is named a request too, since the router ignores it there",
    );
  });

  it("names each question afresh, so a retry is answered once and the next is not mistaken for it", async () => {
    router.serve("POST", "/v1/agents/sessions/session-1/responses", {
      status: 202,
      body: { id: "response-1", session_id: "session-1", status: "running", created_at: new Date().toISOString() },
    });

    await held.responses.create("First question");
    const first = (router.last.body as Record<string, unknown>)["request_id"];
    await held.responses.create("Second question");
    const second = (router.last.body as Record<string, unknown>)["request_id"];

    assert.match(String(first), /^[0-9a-f-]{36}$/);
    assert.notEqual(first, second, "two questions are two requests");
  });

  it("names a request for a question with images too, which a conversation in Chat needs", async () => {
    router.serve("POST", "/v1/agents/sessions/session-1/responses", {
      status: 202,
      body: { id: "response-1", session_id: "session-1", status: "running", created_at: new Date().toISOString() },
    });

    await held.responses.create("What is this?", { images: [{ url: "https://example.com/a.png" }] });

    const body = router.last.body as Record<string, unknown>;
    assert.deepEqual(body["images"], [{ url: "https://example.com/a.png" }]);
    assert.match(String(body["request_id"]), /^[0-9a-f-]{36}$/);
  });

  it("caps one answer and sets how long it thinks, and sends neither unasked", async () => {
    router.serve("POST", "/v1/agents/sessions/session-1/responses", {
      status: 202,
      body: { id: "response-1", session_id: "session-1", status: "running", created_at: new Date().toISOString() },
    });

    await held.responses.create("Think hard", { maxOutputTokens: 512, reasoningEffort: "high" });
    const tuned = router.last.body as Record<string, unknown>;
    await held.responses.create("Plain");
    const plain = router.last.body as Record<string, unknown>;

    assert.equal(tuned["max_output_tokens"], 512);
    assert.equal(tuned["reasoning_effort"], "high");
    assert.match(String(tuned["request_id"]), /^[0-9a-f-]{36}$/);
    assert.deepEqual(Object.keys(plain).sort(), ["request_id", "text"]);
  });

  it("sends what the person did in a reply's UI with their words, still under a request id", async () => {
    router.serve("POST", "/v1/agents/sessions/session-1/responses", {
      status: 202,
      body: { id: "response-1", session_id: "session-1", status: "running", created_at: new Date().toISOString() },
    });

    await held.responses.create("Update the plan", {
      uiAction: { v: 1, message_id: "reply-1", format: "a2ui", name: "update_plan", payload: { pace: "relaxed" } },
    });
    const acted = router.last.body as Record<string, unknown>;
    await held.responses.create("Plain", { uiAction: {} });
    const empty = router.last.body as Record<string, unknown>;

    assert.deepEqual(acted["ui_action"], {
      v: 1,
      message_id: "reply-1",
      format: "a2ui",
      name: "update_plan",
      payload: { pace: "relaxed" },
    });
    assert.match(String(acted["request_id"]), /^[0-9a-f-]{36}$/, "the router keeps the action only under a request id");
    assert.deepEqual(Object.keys(empty).sort(), ["request_id", "text"], "an empty action is not sent");
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
      body: { error: { message: "session: a stored text conversation cannot be rewound; fork it at the response", type: "invalid_request", code: "invalid_request" } },
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

  it("remembers the refresh secret a minted guest is handed", async () => {
    router.serve("POST", "/v1/agents/guests", {
      status: 201,
      body: { id: "guest-1", token: "token-for-guest", refresh_secret: "secret-1" },
    });
    const store = memory();

    await api.guestUser({}, store);

    assert.equal(JSON.parse(store.value ?? "{}").refresh_secret, "secret-1");
  });

  it("comes back with the refresh secret once the remembered token has expired", async () => {
    router.serve("POST", "/v1/agents/guests", {
      status: 201,
      body: {
        id: "guest-7",
        token: "fresh-token",
        expires_at: "2999-01-01T00:00:00Z",
        refresh_secret: "secret-2",
      },
    });
    const store = memory();
    store.write(
      JSON.stringify({
        id: "guest-7",
        token: "expired-token",
        expires_at: "2020-01-01T00:00:00Z",
        refresh_secret: "secret-1",
      }),
    );

    const guest = await api.guestUser({}, store);

    assert.deepEqual(router.last.body, { id: "guest-7", refresh_secret: "secret-1" });
    assert.equal(guest.token, "fresh-token");
    const stored = JSON.parse(store.value ?? "{}");
    assert.equal(stored.token, "fresh-token");
    assert.equal(stored.refresh_secret, "secret-2", "each secret works once, so the next is kept");
  });

  it("remembers whoever comes back when a spent secret is answered with a new guest", async () => {
    router.serve("POST", "/v1/agents/guests", {
      status: 201,
      body: { id: "guest-8", token: "new-token", refresh_secret: "secret-9" },
    });
    const store = memory();
    store.write(
      JSON.stringify({
        id: "guest-7",
        token: "expired-token",
        expires_at: "2020-01-01T00:00:00Z",
        refresh_secret: "spent",
      }),
    );

    const guest = await api.guestUser({ name: "Visitor" }, store);
    const again = await api.guestUser({}, store);

    assert.equal(guest.id, "guest-8");
    assert.equal(again.id, "guest-8");
    assert.equal(router.received.length, 1);
    assert.deepEqual(router.last.body, { id: "guest-7", refresh_secret: "spent", name: "Visitor" });
    assert.equal(JSON.parse(store.value ?? "{}").refresh_secret, "secret-9");
  });

  it("mints a new guest for an expired one with no refresh secret, naming no id", async () => {
    const store = memory();
    store.write(
      JSON.stringify({ id: "guest-7", token: "expired-token", expires_at: "2020-01-01T00:00:00Z" }),
    );

    const guest = await api.guestUser({}, store);

    assert.equal(guest.id, "guest-1");
    assert.deepEqual(router.last.body, {}, "an id it cannot prove is not sent");
  });

  it("hands back a remembered guest whose token has not expired without asking", async () => {
    const store = memory();
    store.write(
      JSON.stringify({ id: "guest-7", token: "live-token", expires_at: "2999-01-01T00:00:00Z" }),
    );

    const guest = await api.guestUser({}, store);

    assert.equal(guest.id, "guest-7");
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

  it("asks anonymously from a page holding only the app key", async () => {
    const page = new Client({ url: router.url, apiKey: "vak_live_x" });

    const guest = await page.guestUser({}, memory());

    assert.equal(guest.id, "guest-1");
    assert.equal(router.last.headers["stream-auth-type"], "anonymous");
    assert.equal(router.last.headers["x-api-key"], "vak_live_x");
    assert.equal(router.last.headers["authorization"], undefined);
  });

  it("says to mint the guest on the backend when the router admits no anonymous callers", async () => {
    router.serve("POST", "/v1/agents/guests", {
      status: 401,
      body: { code: 2, message: "anonymous access is not available on the acceleration api" },
    });
    const page = new Client({ url: router.url, apiKey: "vak_live_x" });

    await assert.rejects(() => page.guestUser({}, memory()), (raised: unknown) => {
      assert.ok(raised instanceof ConfigurationError);
      assert.match(raised.message, /setUser/);
      return true;
    });
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

describe("scheduled tasks", () => {
  let router: TestRouter;
  let api: Client;

  /** A task as the router describes one, with only the fields a test cares about set. */
  function task(over: Partial<Schemas["ScheduledTask"]> = {}): Schemas["ScheduledTask"] {
    return {
      id: "task-1",
      title: "Call mum",
      state: "active",
      ...over,
    } as Schemas["ScheduledTask"];
  }

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("lists the user's tasks with the filters set, and leaves out the rest", async () => {
    router.serve("GET", "/v1/agents/scheduled-tasks", { body: { items: [task()], has_more: false } });

    const page = await api.scheduledTasks.list({ state: "paused", configId: "config-1", limit: 10, cursor: "c2" });
    assert.equal(page.items[0]?.id, "task-1");
    assert.equal(router.last.query.get("state"), "paused");
    assert.equal(router.last.query.get("config_id"), "config-1");
    assert.equal(router.last.query.get("session_id"), null);
    assert.equal(router.last.query.get("limit"), "10");
    assert.equal(router.last.query.get("cursor"), "c2");

    await api.scheduledTasks.list();
    assert.equal([...router.last.query.keys()].length, 0);
  });

  it("reads, snoozes, pauses, resumes and cancels a task, and pages its runs", async () => {
    router.serve("GET", "/v1/agents/scheduled-tasks/task-1", { body: task() });
    router.serve("PATCH", "/v1/agents/scheduled-tasks/task-1", { body: task() });
    router.serve("POST", "/v1/agents/scheduled-tasks/task-1/pause", { body: task({ state: "paused" }) });
    router.serve("POST", "/v1/agents/scheduled-tasks/task-1/resume", { body: task() });
    router.serve("DELETE", "/v1/agents/scheduled-tasks/task-1", { status: 204 });
    router.serve("GET", "/v1/agents/scheduled-tasks/task-1/runs", { body: { items: [], has_more: false } });

    await api.scheduledTasks.get("task-1");
    await api.scheduledTasks.update("task-1", { next: { in: "PT15M" } });
    const paused = await api.scheduledTasks.pause("task-1");
    await api.scheduledTasks.resume("task-1");
    await api.scheduledTasks.runs("task-1", { limit: 5 });
    assert.equal(router.last.query.get("limit"), "5");
    await api.scheduledTasks.delete("task-1");

    assert.equal(paused.state, "paused");
    assert.deepEqual(router.requestsTo("PATCH", "/v1/agents/scheduled-tasks/task-1")[0]?.body, {
      next: { in: "PT15M" },
    });
    assert.equal(router.requestsTo("POST", "/v1/agents/scheduled-tasks/task-1/resume").length, 1);
    assert.equal(router.requestsTo("DELETE", "/v1/agents/scheduled-tasks/task-1").length, 1);
  });
});

describe("search", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("searches under a stored config with its cost labels, and sends neither when unset", async () => {
    router.serve("POST", "/v1/search", { body: { answer: "Prices went up 5%.", results: [] } });

    const answer = await api.router({ config: "healthcare", tags: { team: "web" } }).search("what changed in the pricing page");
    assert.deepEqual(router.last.body, {
      query: "what changed in the pricing page",
      config_id: "healthcare",
      tags: { team: "web" },
    });
    assert.equal((answer as { answer?: string }).answer, "Prices went up 5%.");

    await api.router().search("weather in Lisbon");
    assert.deepEqual(router.last.body, { query: "weather in Lisbon" });
  });
});
