import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";

import {
  Agent,
  Client,
  ConfigurationError,
  Edge,
  USER_KEY,
  daytona,
  userIdOf,
  type Folder,
  type Schemas,
} from "../src/index.js";
import { loadFolder } from "../src/node.js";
import { TestRouter } from "./router.js";

const session = {
  id: "sess_1",
  call_id: "demo",
  call_type: "agent",
  user_id: "john",
  agent_id: "john",
  state: "live",
  created_at: "2026-01-01T00:00:00Z",
};

describe("Agent", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
    router.serve("POST", "/v1/agents/sessions", { status: 201, body: session });
  });

  afterEach(async () => {
    await router.stop();
  });

  /** The body of the session request the agent made. */
  function requested(): Schemas["CreateSessionRequest"] {
    const made = router.received.find((one) => one.path === "/v1/agents/sessions");
    assert.ok(made, "the agent created no session");
    return made.body as Schemas["CreateSessionRequest"];
  }

  /** Opens a text session and closes it, so the assertion is about what was sent. */
  async function chatting(agent: Agent): Promise<void> {
    const opening = agent.chat();
    const connection = await router.socket();
    const held = await opening;
    connection.socket.close();
    await held.wait();
  }

  it("refuses an agent with no name, since the name is what it is stored as", () => {
    assert.throws(() => new Agent({ client: api }), ConfigurationError);
  });

  it("derives who it joins a call as from its name", () => {
    assert.equal(new Agent({ name: "Dr. John Smith", client: api }).userId, "dr--john-smith");
    assert.equal(userIdOf("!!!"), "vision-agent");
  });

  it("holds a conversation in writing without joining a call", async () => {
    await chatting(new Agent({ name: "John", instructions: "Be brief.", client: api }));

    const body = requested();
    assert.equal(body.text, true);
    assert.equal(body.call_id, undefined);
    assert.equal(body.instructions, "Be brief.");
    assert.equal(body.user_id, "john");
    assert.equal(body.user_name, "John");
  });

  it("renders the models it was declared with into the session", async () => {
    await chatting(
      new Agent({
        name: "John",
        client: api,
        pipeline: {
          llm: "llm-fast",
          stt: "en-low-latency",
          tts: "sonic_36",
          voice: "amy",
          language: "en",
          greeting: "Hello",
          backchannel: true,
          maxTokens: 200,
          toolTimeoutMs: 5_000,
        },
      }),
    );

    const body = requested();
    assert.equal(body.llm, "llm-fast");
    assert.equal(body.tts, "sonic_36");
    assert.equal(body.voice, "amy");
    assert.deepEqual(body.languages, ["en"]);
    assert.equal(body.greeting, "Hello");
    assert.equal(body.backchannel, true);
    assert.equal(body.max_tokens, 200);
    assert.equal(body.tool_timeout_ms, 5_000);
  });

  it("leaves out what it was not told, so the backend's own default stands", async () => {
    await chatting(new Agent({ name: "John", client: api }));

    const body = requested();
    assert.equal("llm" in body, false);
    assert.equal("backchannel" in body, false);
    assert.equal("skills" in body, false);
  });

  it("splits the memory filter into who it is about and what narrows it", async () => {
    await chatting(
      new Agent({
        name: "John",
        client: api,
        memoryFilter: { [USER_KEY]: "ana", region: "eu" },
      }),
    );

    assert.deepEqual(requested().memory, { user_id: "ana", filter: { region: "eu" } });
  });

  it("carries the cost labels onto the session", async () => {
    await chatting(new Agent({ name: "John", client: api, costTracking: { team: "support" } }));

    assert.deepEqual(requested().tags, { team: "support" });
  });

  it("renders a harness as the configuration the backend takes", async () => {
    await chatting(
      new Agent({
        name: "John",
        client: api,
        harness: {
          useSkills: true,
          subagents: { default: "llm-slow" },
          tasks: 2,
          vm: daytona(),
          skills: [{ name: "think", description: "Think", instructions: "Work it out." }],
        },
      }),
    );

    const body = requested();
    assert.equal(body.subagent, "llm-slow");
    assert.equal(body.tasks, 2);
    assert.equal(body.sandbox, "daytona");
    assert.deepEqual(body.skills, [
      { name: "think", description: "Think", instructions: "Work it out." },
    ]);
  });

  it("turns delegation off when asked for no skills, which is not the same as saying nothing", async () => {
    await chatting(new Agent({ name: "John", client: api, harness: { useSkills: false } }));

    assert.deepEqual(requested().skills, []);
  });

  it("refuses a harness that would mean something different on every run", () => {
    assert.throws(
      () =>
        new Agent({
          name: "John",
          client: api,
          harness: { skills: [{ name: "x", description: "", instructions: "Go." }] },
        }),
      ConfigurationError,
    );
    assert.throws(
      () => new Agent({ name: "John", client: api, harness: { tasks: -1 } }),
      ConfigurationError,
    );
  });

  it("resolves a config named by name to its id, and looks it up once", async () => {
    router.serve("GET", "/v1/agents/configs", {
      body: [{ id: "cfg_7", name: "support", created_at: "2026-01-01T00:00:00Z" }],
    });
    const agent = new Agent({ name: "John", client: api, pipeline: { config: "support" } });

    await chatting(agent);
    await chatting(agent);

    assert.equal(requested().config_id, "cfg_7");
    assert.equal(
      router.received.filter((one) => one.path === "/v1/agents/configs").length,
      1,
      "the config was looked up more than once",
    );
  });

  it("passes a config name nothing is stored under straight through", async () => {
    router.serve("GET", "/v1/agents/configs", { body: [] });

    await chatting(new Agent({ name: "John", client: api, pipeline: { config: "cfg_raw" } }));

    assert.equal(requested().config_id, "cfg_raw");
  });

  it("fills in from a directory what was not written in code", async () => {
    const folder: Folder = {
      path: "/agents/jean",
      name: "jean",
      instructions: "Be brief.",
      guardrail: "Refuse medical advice.",
      skills: [{ name: "think", description: "Think", instructions: "Work it out." }],
      knowledge: [],
      knowledgeURLs: [],
    };

    const agent = new Agent({ folder, client: api });

    assert.equal(agent.name, "jean");
    assert.equal(agent.instructions, "Be brief.");
    assert.equal(agent.guardrail, "Refuse medical advice.");

    await chatting(agent);
    assert.deepEqual(requested().skills, [
      { name: "think", description: "Think", instructions: "Work it out." },
    ]);
  });

  it("lets what is written in code win over the directory", () => {
    const folder: Folder = {
      path: "/agents/jean",
      name: "jean",
      instructions: "Be brief.",
      guardrail: "",
      skills: [],
      knowledge: [],
      knowledgeURLs: [],
    };

    const agent = new Agent({ folder, name: "Jeanne", instructions: "Be warm.", client: api });

    assert.equal(agent.name, "Jeanne");
    assert.equal(agent.instructions, "Be warm.");
  });

  it("answers a call that arrived on the dispatch socket without creating one", async () => {
    const agent = new Agent({ name: "John", client: api });

    const opening = agent.answer({
      callId: "call_1",
      callType: "agent",
      calledNumber: "+15551234567",
      callerNumber: "+15557654321",
      custom: {},
    });
    const connection = await router.socket();
    const held = await opening;

    const body = requested();
    assert.equal(body.call_id, "call_1");
    assert.equal(body.call_type, "agent");
    assert.deepEqual(body.phone, { number: "+15551234567" });
    assert.equal(body.text, undefined, "a call is not held in writing");

    connection.socket.close();
    await held.wait();
  });

  it("rings somebody and joins the call it placed, navigating", async () => {
    router.serve("POST", "/v1/phone/calls", {
      status: 202,
      body: { vendor: "twilio", vendor_call_id: "vc_1", call_id: "demo", call_type: "agent" },
    });
    const agent = new Agent({
      name: "John",
      client: api,
      costTracking: { team: "sales" },
      edge: new Edge({ apiKey: "key", apiSecret: "secret", fetch: streamCallCreated }),
    });

    const opening = agent.startCall("+15551234567", "+15557654321");
    const connection = await router.socket();
    const held = await opening;

    const placed = router.received.find((one) => one.path === "/v1/phone/calls")?.body as {
      from: string;
      to: string;
      call_id: string;
      tags: Record<string, string>;
    };
    assert.equal(placed.from, "+15551234567");
    assert.equal(placed.to, "+15557654321");
    assert.deepEqual(placed.tags, { team: "sales" });

    const body = requested();
    assert.equal(body.navigating, true);
    assert.deepEqual(body.phone, { number: "+15551234567", vendor_call_id: "vc_1" });
    assert.equal(body.call_id, placed.call_id);

    connection.socket.close();
    await held.wait();
  });

  it("builds a link to the call it joined, as a listener of its own", async () => {
    const agent = new Agent({
      name: "John",
      client: api,
      edge: new Edge({ apiKey: "key", apiSecret: "secret", fetch: streamCallCreated }),
    });

    const opening = agent.join({ id: "demo" });
    const connection = await router.socket();
    const held = await opening;

    const url = new URL(await agent.monitorURL(held));
    assert.equal(url.pathname, "/video/demos/join/demo");
    assert.equal(url.searchParams.get("user_name"), "Monitor");

    connection.socket.close();
    await held.wait();
  });

  it("refuses to place a call with nobody to ring", async () => {
    const agent = new Agent({ name: "John", client: api });

    await assert.rejects(() => agent.startCall("+15551234567", ""), ConfigurationError);
    await assert.rejects(() => agent.waitForCall(""), ConfigurationError);
  });

  describe("sync", () => {
    it("stores everything the agent is in one request", async () => {
      router.serve("POST", "/v1/agents/sync", {
        body: {
          unchanged: false,
          config: { id: "cfg_1", name: "jean", created_at: "2026-01-01T00:00:00Z" },
        },
      });
      const folder: Folder = {
        path: "/agents/jean",
        name: "jean",
        settings: { llm: "llm-smart", stt: "flux", tags: { team: "support" } },
        instructions: "Be brief.",
        guardrail: "Refuse medical advice.",
        skills: [{ name: "think", description: "Think", instructions: "Work it out." }],
        knowledge: [{ source: "pricing.md", text: "The plans cost money." }],
        knowledgeURLs: [{ url: "https://example.com", title: "Home" }],
      };

      const result = await new Agent({
        folder,
        client: api,
        pipeline: { llm: "llm-fast", tts: "sonic_36" },
      }).sync();

      assert.equal(result.config.id, "cfg_1");

      const body = router.received.find((one) => one.path === "/v1/agents/sync")
        ?.body as Schemas["SyncAgentRequest"];
      assert.equal(body.name, "jean");
      assert.equal(body.instructions, "Be brief.");
      assert.equal(body.guardrail, "Refuse medical advice.");
      assert.equal(body.llm, "llm-fast", "what the code set wins over agent.yaml");
      assert.equal(body.stt, "flux", "what only agent.yaml says still goes");
      assert.deepEqual(body.tags, { team: "support" });
      assert.equal(body.skills?.length, 1);
      assert.equal(body.knowledge?.length, 1);
      assert.deepEqual(body.knowledge_urls, [{ url: "https://example.com", title: "Home" }]);
      assert.match(body.hash, /^[0-9a-f]{64}$/);
      assert.equal(router.received.length, 1, "the pages go in the same request");
    });

    it("fingerprints the same agent the same way twice, and a changed one differently", async () => {
      router.serve("POST", "/v1/agents/sync", {
        body: {
          unchanged: false,
          config: { id: "cfg_1", name: "john", created_at: "2026-01-01T00:00:00Z" },
        },
      });

      await new Agent({ name: "john", instructions: "Be brief.", client: api }).sync();
      await new Agent({ name: "john", instructions: "Be brief.", client: api }).sync();
      await new Agent({ name: "john", instructions: "Be warm.", client: api }).sync();

      const hashes = router.received
        .filter((one) => one.path === "/v1/agents/sync")
        .map((one) => (one.body as Schemas["SyncAgentRequest"]).hash);

      assert.equal(hashes[0], hashes[1]);
      assert.notEqual(hashes[1], hashes[2]);
    });

    it("only reads back a directory nothing has touched since .agent_sync was written", async () => {
      const stored = { id: "cfg_1", name: "jean", created_at: "2026-01-01T00:00:00Z" };
      router.serve("POST", "/v1/agents/sync", { body: { unchanged: false, config: stored } });
      router.serve("GET", "/v1/agents/configs", { body: [stored] });
      const root = join(await mkdtemp(join(tmpdir(), "vision-agents-")), "jean");
      await mkdir(root);
      await writeFile(join(root, "agent.yaml"), "name: jean\n");
      await writeFile(join(root, "instructions.md"), "Be brief.\n");

      await new Agent({ folder: await loadFolder(root), client: api }).sync();
      const again = await new Agent({ folder: await loadFolder(root), client: api }).sync();
      await writeFile(join(root, "instructions.md"), "Be warm.\n");
      await new Agent({ folder: await loadFolder(root), client: api }).sync();

      assert.equal(again.unchanged, true);
      assert.equal(again.config.id, "cfg_1");
      assert.equal(router.requestsTo("POST", "/v1/agents/sync").length, 2);
      assert.equal(router.requestsTo("GET", "/v1/agents/configs")[0]?.query.get("name"), "jean");
      const recorded = JSON.parse(await readFile(join(root, ".agent_sync"), "utf8")) as {
        hash: string;
        synced_at: string;
      };
      const last = router.requestsTo("POST", "/v1/agents/sync").at(-1);
      assert.equal(recorded.hash, (last?.body as Schemas["SyncAgentRequest"]).hash);
      assert.match(recorded.synced_at, /^\d{4}-\d\d-\d\dT/);
    });
  });
});

describe("Edge", () => {
  it("refuses to create a call without the credentials to create one with", () => {
    assert.throws(() => new Edge({ apiKey: "", apiSecret: "" }), ConfigurationError);
  });

  it("refuses a call nobody created", async () => {
    const edge = new Edge({ apiKey: "key", apiSecret: "secret", fetch: streamCallCreated });

    await assert.rejects(() => edge.createCall({}, { id: "" }), ConfigurationError);
  });

  it("names a call after a random string when it is given none", async () => {
    const edge = new Edge({ apiKey: "key", apiSecret: "secret", fetch: streamCallCreated });

    const call = await edge.createCall({}, { id: "john" });

    assert.match(call.id, /^[0-9a-f]{16}$/);
    assert.equal(call.type, "agent");
  });

  it("creates the call as the agent, authenticated as the app", async () => {
    const asked: { url: string; headers: Headers; body: unknown }[] = [];
    const edge = new Edge({
      apiKey: "key",
      apiSecret: "secret",
      fetch: async (url, init) => {
        asked.push({
          url: String(url),
          headers: new Headers(init?.headers),
          body: JSON.parse(String(init?.body)),
        });
        return Response.json({});
      },
    });

    await edge.createCall({ id: "demo", type: "agent" }, { id: "john", name: "John" });

    const made = asked[0];
    assert.ok(made);
    assert.match(made.url, /\/api\/v2\/video\/call\/agent\/demo\?api_key=key$/);
    assert.equal(made.headers.get("stream-auth-type"), "jwt");
    assert.ok(made.headers.get("authorization"));
    assert.deepEqual(made.body, { data: { created_by_id: "john" } });
  });

  it("builds a link a person can open to hear the agent", async () => {
    const edge = new Edge({ apiKey: "key", apiSecret: "secret" });

    const url = new URL(
      await edge.monitorURLFor({ id: "demo", type: "agent" }, { id: "monitor", name: "Monitor" }),
    );

    assert.equal(url.pathname, "/video/demos/join/demo");
    assert.equal(url.searchParams.get("api_key"), "key");
    assert.equal(url.searchParams.get("user_name"), "Monitor");
    assert.equal(url.searchParams.get("skip_lobby"), "true");
    assert.ok(url.searchParams.get("token"));
  });

  it("refuses to build a link to a conversation held in writing", async () => {
    const edge = new Edge({ apiKey: "key", apiSecret: "secret" });

    await assert.rejects(
      () => edge.monitorURLFor({ id: "", type: "agent" }, { id: "monitor" }),
      ConfigurationError,
    );
  });
});

/** Stream answering that the call is there, which is all the SDK reads from it. */
const streamCallCreated: typeof fetch = () => Promise.resolve(Response.json({}));
