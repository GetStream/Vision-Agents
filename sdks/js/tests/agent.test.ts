import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";
import { inspect } from "node:util";

import {
  Agent,
  Client,
  ConfigurationError,
  Edge,
  USER_KEY,
  daytona,
  userIdOf,
  type ConnectorDeclaration,
  type Declaration,
  type Folder,
  type Schemas,
} from "../src/index.js";
import { loadFolder } from "../src/node.js";
import { inPage } from "./device.js";
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

    const body = requested() as Record<string, unknown>;
    assert.equal(body["start_voice"], undefined);
    assert.equal(body["instructions"], undefined, "instructions reach the backend with sync");
    assert.equal(body["user_id"], "john");
    assert.equal(body["user_name"], "John");
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
    assert.deepEqual(body.greeting, { text: "Hello" });
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

  it("puts a session in the agent's zone when the call names none", async () => {
    await chatting(new Agent({ name: "John", client: api, timezone: "Europe/Lisbon" }));

    assert.equal(requested().timezone, "Europe/Lisbon");
  });

  it("lets a call name the zone of the person on it, over the agent's", async () => {
    const agent = new Agent({ name: "John", client: api, timezone: "Europe/Lisbon" });

    for (const timezone of ["America/New_York", ""]) {
      const opening = agent.chat({ timezone });
      const connection = await router.socket();
      const held = await opening;
      connection.socket.close();
      await held.wait();
    }
    const answering = agent.answer(
      {
        callId: "sess_1",
        callType: "agent",
        sessionId: "sess_1",
        calledNumber: "+15551234567",
        callerNumber: "+15557654321",
        custom: {},
      },
      { timezone: "Asia/Tokyo" },
    );
    const connection = await router.socket();
    const held = await answering;
    connection.socket.close();
    await held.wait();

    const zones = router
      .requestsTo("POST", "/v1/agents/sessions")
      .map((one) => (one.body as Schemas["CreateSessionRequest"]).timezone);
    assert.deepEqual(zones, ["America/New_York", "Europe/Lisbon", "Asia/Tokyo"]);
  });

  it("sends no zone when neither the agent nor the call names one, whatever the machine is in", async () => {
    await inPage("Europe/Lisbon", () => chatting(new Agent({ name: "John", client: api })));

    assert.equal("timezone" in requested(), false);
  });

  it("stores a harness on the config rather than on each session, since a session runs its config's", async () => {
    router.serve("POST", "/v1/agents/sync", {
      body: {
        unchanged: false,
        config: { id: "cfg_1", name: "John", created_at: "2026-01-01T00:00:00Z" },
      },
    });
    const agent = new Agent({
      name: "John",
      client: api,
      harness: {
        name: "default",
        subagents: { default: "llm-slow" },
        vm: daytona(),
        skills: [{ name: "think", description: "Think", instructions: "Work it out." }],
      },
    });

    await agent.sync();
    await chatting(agent);

    const synced = router.requestsTo("POST", "/v1/agents/sync")[0]?.body as Schemas["SyncAgentRequest"];
    assert.equal(synced.harness, "default");
    assert.equal(synced.subagent, "llm-slow");
    assert.equal(synced.sandbox, "daytona");
    assert.deepEqual(synced.skills, [
      { name: "think", description: "Think", instructions: "Work it out.", config_id: "" },
    ]);
    const body = requested() as Record<string, unknown>;
    for (const key of ["harness", "subagent", "sandbox", "skills", "tasks"]) {
      assert.equal(key in body, false, `${key} went on the session`);
    }
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

  it("fills in from a directory what was not written in code", () => {
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

  it("answers a call that arrived on the dispatch socket in the session it names", async () => {
    const agent = new Agent({ name: "John", client: api });

    const opening = agent.answer({
      callId: "sess_1",
      callType: "agent",
      sessionId: "sess_1",
      calledNumber: "+15551234567",
      callerNumber: "+15557654321",
      custom: {},
    });
    const connection = await router.socket();
    const held = await opening;

    const body = requested();
    assert.equal(body.id, "sess_1");
    assert.equal(body.start_voice, true, "a call is not held in writing");
    assert.deepEqual(body.phone, { number: "+15551234567" });

    connection.socket.close();
    await held.wait();
  });

  it("refuses a call that names no session, since there is no call to join", async () => {
    const agent = new Agent({ name: "John", client: api });
    const call = {
      callId: "",
      callType: "default",
      sessionId: "",
      calledNumber: "+15551234567",
      callerNumber: "",
      custom: {},
    };

    await assert.rejects(() => agent.answer(call), ConfigurationError);
    assert.equal(router.received.length, 0);
  });

  it("rings somebody and joins the session the call was placed for, navigating", async () => {
    router.serve("POST", "/v1/phone/calls", {
      status: 202,
      body: { status: "ringing", vendor: "twilio", vendor_call_id: "vc_1", session_id: "sess_1" },
    });
    const agent = new Agent({ name: "John", client: api, costTracking: { team: "sales" } });

    const opening = agent.startCall("+15551234567", "+15557654321");
    const connection = await router.socket();
    const held = await opening;

    const placed = router.received.find((one) => one.path === "/v1/phone/calls")?.body;
    assert.deepEqual(placed, { from: "+15551234567", to: "+15557654321", tags: { team: "sales" } });

    const body = requested();
    assert.equal(body.id, "sess_1");
    assert.equal(body.start_voice, true);
    assert.equal(body.navigating, true);
    assert.deepEqual(body.phone, { number: "+15551234567", vendor_call_id: "vc_1" });

    connection.socket.close();
    await held.wait();
  });

  it("answers the next caller to a number in the session the router opened the call for", async () => {
    router.serve("POST", "/v1/phone/numbers/%2B15551234567/attach", {
      body: { number: "+15551234567", trunk_id: "t_1", sip_uri: "sip:x" },
    });
    const agent = new Agent({ name: "John", client: api });

    const waiting = agent.waitForCall("+15551234567");
    const dispatch = await router.socket();
    assert.equal(dispatch.path, "/v1/dispatch");
    assert.equal(dispatch.query.get("capacity"), "1");
    dispatch.send({
      type: "call",
      work_id: "work_1",
      call_id: "sess_1",
      call_type: "agent",
      session_id: "sess_1",
      called_number: "+15551234567",
    });
    const connection = await router.socket();
    connection.send({ type: "heard", text: "hello?" });
    const held = await waiting;

    const attached = router.received.find((one) => one.path.endsWith("/attach"))?.body;
    assert.deepEqual(attached, {}, "a number is attached to the agent, not to a call");
    const body = requested();
    assert.equal(body.id, "sess_1");
    assert.equal(body.start_voice, true);
    assert.deepEqual(body.phone, { number: "+15551234567" });

    connection.socket.close();
    await held.wait();
  });

  it("builds a link to the session's own call, as a listener of its own", async () => {
    const agent = new Agent({
      name: "John",
      client: api,
      edge: new Edge({ apiKey: "key", apiSecret: "secret", fetch: streamCallCreated }),
    });

    const opening = agent.join();
    const connection = await router.socket();
    const held = await opening;

    assert.equal(requested().start_voice, true);
    const url = new URL(await agent.monitorURL(held));
    assert.equal(url.pathname, "/video/demos/join/sess_1");
    assert.equal(url.searchParams.get("user_name"), "Monitor");

    connection.socket.close();
    await held.wait();
  });

  it("has no link to give for a conversation held in writing", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: { ...session, call_id: "", call_type: "" },
    });
    const agent = new Agent({
      name: "John",
      client: api,
      edge: new Edge({ apiKey: "key", apiSecret: "secret", fetch: streamCallCreated }),
    });

    const opening = agent.chat();
    const connection = await router.socket();
    const held = await opening;

    await assert.rejects(() => agent.monitorURL(held), ConfigurationError);

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
        settings: {
          llm: "llm-smart",
          stt: "flux",
          greeting: { text: "Hello", mode: "variation" },
          harness: "default",
          tags: { team: "support" },
          dispatch: { incoming_call: "disabled", text: "enabled" },
        },
        instructions: "Be brief.",
        guardrail: "Refuse medical advice.",
        skills: [{ name: "think", description: "Think", instructions: "Work it out." }],
        knowledge: [{ source: "pricing.md", text: "The plans cost money." }],
        knowledgeURLs: [{ url: "https://example.com", title: "Home", refresh_hours: 24 }],
        simulations: [{ name: "order", scenario: "Order a wrap.", assertion: "One wrap." }],
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
      assert.deepEqual(body.dispatch, { incoming_call: "disabled", text: "enabled" });
      assert.deepEqual(body.greeting, { text: "Hello", mode: "variation" });
      assert.equal(body.harness, "default");
      assert.equal(body.skills?.length, 1);
      assert.equal(body.knowledge?.length, 1);
      assert.deepEqual(body.knowledge_urls, [
        { url: "https://example.com", title: "Home", refresh_hours: 24 },
      ]);
      assert.deepEqual(body.simulations, [
        { name: "order", scenario: "Order a wrap.", assertion: "One wrap." },
      ]);
      assert.match(body.hash, /^[0-9a-f]{64}$/);
      assert.equal(router.received.length, 1, "the pages go in the same request");
    });

    it("sends the region agent.yaml names, and none when it names none", async () => {
      router.serve("POST", "/v1/agents/sync", {
        body: {
          unchanged: false,
          config: { id: "cfg_1", name: "jean", created_at: "2026-01-01T00:00:00Z" },
        },
      });
      const folder = (settings: Declaration): Folder => ({
        path: "/agents/jean",
        name: "jean",
        settings,
        instructions: "Be brief.",
        guardrail: "",
        skills: [],
        knowledge: [],
        knowledgeURLs: [],
      });

      await new Agent({ folder: folder({ region: "eu" }), client: api }).sync();
      await new Agent({ folder: folder({}), client: api }).sync();

      const bodies = router.received
        .filter((one) => one.path === "/v1/agents/sync")
        .map((one) => one.body as Schemas["SyncAgentRequest"]);
      assert.equal(bodies[0]?.region, "eu");
      assert.equal(bodies[1]?.region, undefined, "the router keeps the stored region");
    });

    it("sends an empty simulations/ as an empty list and a missing one not at all", async () => {
      router.serve("POST", "/v1/agents/sync", {
        body: {
          unchanged: false,
          config: { id: "cfg_1", name: "jean", created_at: "2026-01-01T00:00:00Z" },
        },
      });
      const folder: Folder = {
        path: "/agents/jean",
        name: "jean",
        instructions: "Be brief.",
        guardrail: "",
        skills: [],
        knowledge: [],
        knowledgeURLs: [],
      };

      await new Agent({ folder, client: api }).sync();
      await new Agent({ folder: { ...folder, simulations: [] }, client: api }).sync();

      const [without, emptied] = router
        .requestsTo("POST", "/v1/agents/sync")
        .map((one) => one.body as Schemas["SyncAgentRequest"]);
      assert.equal(without && "simulations" in without, false, "the stored ones would be deleted");
      assert.deepEqual(emptied?.simulations, [], "the stored ones would never be deleted");
      assert.notEqual(without?.hash, emptied?.hash, "emptying simulations/ would not sync");
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

    describe("connectors", () => {
      const stored = { id: "cfg_1", name: "jean", created_at: "2026-01-01T00:00:00Z" };
      const folder = (connectors: ConnectorDeclaration[] | undefined): Folder => ({
        path: "/agents/jean",
        name: "jean",
        settings: connectors ? { connectors } : {},
        instructions: "Be brief.",
        guardrail: "",
        skills: [],
        knowledge: [],
        knowledgeURLs: [],
      });
      const connection = (id: string, label: string): Schemas["Connection"] =>
        ({ id, label }) as Schemas["Connection"];

      beforeEach(() => {
        router.serve("POST", "/v1/agents/sync", { body: { unchanged: false, config: stored } });
      });

      async function synced(connectors: ConnectorDeclaration[] | undefined): Promise<Record<string, unknown>> {
        await new Agent({ folder: folder(connectors), client: api }).sync();
        return router.requestsTo("POST", "/v1/agents/sync").at(-1)?.body as Record<string, unknown>;
      }

      it("sends a connector named alone as a session binding of every tool, and lists no connections", async () => {
        const body = await synced([{ name: "linear" }, { name: "github", tools: ["list_*"], required: false }]);

        assert.deepEqual(body["connectors"], [
          { name: "linear", connector_id: "linear", tools: [], tool_patterns: ["*"], connection: { type: "session" } },
          {
            name: "github",
            connector_id: "github",
            tools: [],
            tool_patterns: ["list_*"],
            required: false,
            connection: { type: "session" },
          },
        ]);
        assert.equal(router.requestsTo("GET", "/v1/agents/connections").length, 0);
      });

      async function syncedFrom(declaration: string): Promise<Record<string, unknown>> {
        const root = join(await mkdtemp(join(tmpdir(), "vision-agents-")), "jean");
        await mkdir(root);
        await writeFile(join(root, "agent.yaml"), declaration);
        await writeFile(join(root, "instructions.md"), "Be brief.\n");
        await new Agent({ folder: await loadFolder(root), client: api }).sync();
        return router.requestsTo("POST", "/v1/agents/sync").at(-1)?.body as Record<string, unknown>;
      }

      it("sends no connectors when agent.yaml has none, and an empty list when it lists none", async () => {
        const without = await synced(undefined);
        const emptied = await synced([]);

        assert.equal("connectors" in without, false, "the stored bindings would be removed");
        assert.deepEqual(emptied["connectors"], [], "the stored bindings would never be removed");
        assert.notEqual(without["hash"], emptied["hash"]);
      });

      it("fixes a connection by its id before its label, reading every page", async () => {
        router.serve("GET", "/v1/agents/connections", (request) => ({
          body: request.query.get("cursor")
            ? { items: [connection("acct", "other")], has_more: false }
            : { items: [connection("wrong", "acct")], has_more: true, next_cursor: "2" },
        }));

        const body = await synced([{ name: "issues", connector: "linear", connection: "acct" }]);

        const listed = router.requestsTo("GET", "/v1/agents/connections");
        assert.equal(listed.length, 2);
        assert.equal(listed[0]?.query.get("owner_type"), "app");
        assert.equal(listed[0]?.query.get("connector_id"), "linear");
        assert.deepEqual((body["connectors"] as Schemas["AgentConnectorBinding"][])[0]?.connection, {
          type: "fixed",
          connection_id: "acct",
        });
      });

      it("fixes the one connection with that label, and stops at a cursor it has read", async () => {
        router.serve("GET", "/v1/agents/connections", (request) => ({
          body: request.query.get("cursor")
            ? { items: [connection("the-one", "work")], has_more: true, next_cursor: "again" }
            : { items: [connection("other", "nope")], has_more: true, next_cursor: "again" },
        }));

        const body = await synced([{ name: "issues", connection: "work" }]);

        assert.equal(router.requestsTo("GET", "/v1/agents/connections").length, 2);
        assert.deepEqual((body["connectors"] as Schemas["AgentConnectorBinding"][])[0]?.connection, {
          type: "fixed",
          connection_id: "the-one",
        });
      });

      it("refuses a connection nobody has, or a label two share, and sends nothing", async () => {
        router.serve("GET", "/v1/agents/connections", {
          body: { items: [connection("a", "work"), connection("b", "work")], has_more: false },
        });

        await assert.rejects(
          () => new Agent({ folder: folder([{ name: "issues", connection: "missing" }]), client: api }).sync(),
          /connector "issues": no app connection with id or label "missing"/,
        );
        await assert.rejects(
          () => new Agent({ folder: folder([{ name: "issues", connection: "work" }]), client: api }).sync(),
          /connector "issues": "work" matches 2 app connections/,
        );
        assert.equal(router.requestsTo("POST", "/v1/agents/sync").length, 0);
      });
    });

    describe("channels", () => {
      const stored = { id: "cfg_1", name: "jean", created_at: "2026-01-01T00:00:00Z" };

      beforeEach(() => {
        router.serve("POST", "/v1/agents/sync", { body: { unchanged: false, config: stored } });
      });

      async function synced(declaration: string): Promise<Record<string, unknown>> {
        const root = join(await mkdtemp(join(tmpdir(), "vision-agents-")), "jean");
        await mkdir(root);
        await writeFile(join(root, "agent.yaml"), declaration);
        await writeFile(join(root, "instructions.md"), "Be brief.\n");
        await new Agent({ folder: await loadFolder(root), client: api }).sync();
        return router.requestsTo("POST", "/v1/agents/sync").at(-1)?.body as Record<string, unknown>;
      }

      it("refuses plugins and mcp_servers that name something, and sends nothing", async () => {
        await assert.rejects(() => synced("name: jean\nplugins:\n  - weather\n"), /plugins is no longer read/);
        await assert.rejects(
          () => synced("name: jean\nmcp_servers:\n  - name: a\n    url: https://a.example.com\n"),
          /mcp_servers is no longer read/,
        );
        assert.equal(router.requestsTo("POST", "/v1/agents/sync").length, 0);
      });
    });

    describe("of a directory's files", () => {
      /** Writes an agent directory under a temporary one, and hands back its path. */
      async function agentDir(files: Record<string, string>): Promise<string> {
        const root = join(await mkdtemp(join(tmpdir(), "vision-agents-")), "jean");
        for (const [name, content] of Object.entries(files)) {
          await mkdir(join(root, name, ".."), { recursive: true });
          await writeFile(join(root, name), content);
        }
        return root;
      }

      /** Answers a sync as the router does: the config, under a hash of what was sent. */
      function serveSync(leftOut: Schemas["LeftOutConnector"][] = []): { sync_hash: string } {
        const stored = {
          id: "cfg_1", name: "jean", created_at: "2026-01-01T00:00:00Z", sync_hash: "",
          ...(leftOut.length > 0 ? { left_out_connectors: leftOut } : {}),
        };
        router.serve("POST", "/v1/agents/sync/files", (request) => {
          stored.sync_hash = createHash("sha256").update(JSON.stringify(request.body)).digest("hex");
          return { body: { unchanged: false, config: stored } };
        });
        router.serve("GET", "/v1/agents/configs", { body: [stored] });
        return stored;
      }

      function synced(): Schemas["SyncAgentFilesRequest"][] {
        return router
          .requestsTo("POST", "/v1/agents/sync/files")
          .map((one) => one.body as Schemas["SyncAgentFilesRequest"]);
      }

      it("sends a loaded directory's files as they are, and leaves reading them to the router", async () => {
        const held = {
          "agent.yaml": "llm: llm-smart\ntags:\n  team: support\nconnectors:\n  - {name: linear, connection: company}\nui: {}\n",
          "instructions.md": "Be brief.\n",
          "guardrail.md": "Refuse medical advice.\n",
          "skills/think.md": "---\ndescription: Think\n---\nWork it out.\n",
          "skills/pdf/SKILL.md": "---\ndescription: Fill a form\n---\nRun the script.\n",
          "skills/pdf/scripts/fill.py": "print('filled')\n",
          "knowledge/pricing.md": "The plans cost money.\n",
          "knowledge/urls.yaml": "- https://example.com\n",
          "simulations/order.yaml": "- name: order\n  scenario: Order a wrap.\n  assertion: One wrap.\n",
          "ui/blueprints.yaml": "- name: booking\n  when: booking a table\n  order: [CardHeader]\n",
        };
        const root = await agentDir(held);
        serveSync();

        const result = await new Agent({ folder: await loadFolder(root), client: api }).sync();

        assert.equal(result.config.id, "cfg_1");
        assert.equal(router.received.length, 1, "the whole directory goes in one request, and no connection is looked up here");
        const [body] = synced();
        assert.deepEqual(Object.keys(body ?? {}).sort(), ["files", "name"], "what the files say is the router's to read");
        assert.equal(body?.name, "jean");
        assert.deepEqual(Object.fromEntries((body?.files ?? []).map((file) => [file.path, file.content])), held);
        assert.deepEqual((body?.files ?? []).map((file) => file.path), Object.keys(held).sort(), "in path order");
      });

      it("only reads back a directory nothing has touched since .agent_sync was written", async () => {
        serveSync();
        const root = await agentDir({ "agent.yaml": "name: jean\n", "instructions.md": "Be brief.\n" });

        await new Agent({ folder: await loadFolder(root), client: api }).sync();
        const again = await new Agent({ folder: await loadFolder(root), client: api }).sync();
        await writeFile(join(root, "instructions.md"), "Be warm.\n");
        const edited = await new Agent({ folder: await loadFolder(root), client: api }).sync();

        assert.equal(again.unchanged, true);
        assert.equal(again.config.id, "cfg_1");
        assert.equal(synced().length, 2);
        assert.equal(router.requestsTo("GET", "/v1/agents/configs")[0]?.query.get("name"), "jean");
        const recorded = JSON.parse(await readFile(join(root, ".agent_sync"), "utf8")) as {
          hash: string;
          synced_at: string;
        };
        assert.match(recorded.hash, /^[0-9a-f]{64}$/);
        assert.equal(
          (recorded as { sync_hash?: string }).sync_hash,
          edited.config.sync_hash,
          "the stamp keeps the hash the router stored it under, as the Go SDK writes it",
        );
        assert.match(recorded.synced_at, /^\d{4}-\d\d-\d\dT/);
      });

      it("sends a directory again while a connector of it is left out, so the sync that finds its connection is made", async () => {
        serveSync([{ name: "linear", connector_id: "linear", connection: "company" }]);
        const root = await agentDir({ "agent.yaml": "name: jean\nconnectors:\n  - {name: linear, connection: company}\n" });

        await new Agent({ folder: await loadFolder(root), client: api }).sync();
        await new Agent({ folder: await loadFolder(root), client: api }).sync();

        assert.equal(synced().length, 2);
      });

      it("sends a stamped directory again once the stored config is no longer what was synced", async () => {
        const stored = serveSync();
        const root = await agentDir({ "agent.yaml": "name: jean\n" });

        await new Agent({ folder: await loadFolder(root), client: api }).sync();
        await new Agent({ folder: await loadFolder(root), client: api }).sync();
        // Somebody imports another directory in the dashboard, or edits the config, which clears it.
        stored.sync_hash = "imported";
        await new Agent({ folder: await loadFolder(root), client: api }).sync();

        assert.equal(synced().length, 2, "once, not again while the config was as synced, and again once it was not");
      });

      it("sends the files the format names and nothing else of the folder", async () => {
        serveSync();
        const root = await agentDir({
          "agent.yaml": "name: jean\nui:\n  blueprints: design/recipes.yaml\n",
          "design/recipes.yaml": "- name: booking\n  when: booking\n  order: [CardHeader]\n",
          "design/other.yaml": "not named by agent.yaml\n",
          "ui/components.json": "[]\n",
          "knowledge/pricing.md": "Pro.\n",
          "knowledge/data.csv": "a,b\n",
          "simulations/lunch.yaml": "- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n",
          "simulations/notes.md": "not a simulations file\n",
          // What a project keeps beside its agent, none of which is the agent.
          "index.ts": "export {};\n",
          "package-lock.json": "{}\n",
          ".env": "SECRET=1\n",
          "keys/prod.pem": "-----BEGIN PRIVATE KEY-----\n",
          "knowledge/node_modules/left-pad/index.js": "module.exports = 1;\n",
        });
        await writeFile(join(root, "knowledge", "logo.png"), new Uint8Array([0x89, 0x50, 0xff, 0xfe]));

        await new Agent({ folder: await loadFolder(root), client: api }).sync();

        assert.deepEqual(synced()[0]?.files.map((file) => file.path), [
          "agent.yaml",
          "design/recipes.yaml",
          "knowledge/data.csv",
          "knowledge/pricing.md",
          "simulations/lunch.yaml",
          "ui/components.json",
        ]);
      });

      it("reads the files when it syncs, so an edit after the folder was loaded is what is sent", async () => {
        serveSync();
        const root = await agentDir({ "agent.yaml": "name: jean\n", "instructions.md": "Be brief.\n" });
        const folder = await loadFolder(root);
        await writeFile(join(root, "knowledge.md"), "beside the directory\n");
        await mkdir(join(root, "knowledge"));
        await writeFile(join(root, "knowledge", "pricing.md"), "Pro.\n");

        await new Agent({ folder, client: api }).sync();

        assert.deepEqual(synced()[0]?.files.map((file) => file.path), ["agent.yaml", "instructions.md", "knowledge/pricing.md"]);
      });

      it("refuses a directory over what a sync sends, by name, before it sends anything", async () => {
        serveSync();
        const files: Record<string, string> = { "agent.yaml": "name: jean\n" };
        for (let index = 0; index < 9; index++) {
          files[`knowledge/${index}.md`] = "a".repeat(1 << 20);
        }
        const folder = await loadFolder(await agentDir(files));

        await assert.rejects(
          () => new Agent({ folder, client: api }).sync(),
          (cause: unknown) => cause instanceof ConfigurationError && /holds more than the 8388608 bytes a sync sends/.test(cause.message),
        );
        assert.equal(synced().length, 0);
      });

      it("stamps a request with the fingerprint the Go SDK takes of it", async () => {
        serveSync();
        let stamp = "";
        const folder: Folder = {
          path: "/agents/jean", name: "jean", instructions: "", guardrail: "", skills: [], knowledge: [], knowledgeURLs: [],
          files: async () => [{ path: "agent.yaml", content: "name: jean\n" }],
          stamp: { read: async () => stamp, write: async (value) => { stamp = value; } },
        };

        await new Agent({ folder, client: api, costTracking: { b: "<2>", a: "1" } }).sync();

        // sha256 of {"files":[{"content":"name: jean\n","path":"agent.yaml"}],"name":"jean","tags":{"a":"1","b":"<2>"}},
        // which sdks/go/agents/folder_test.go holds the Go SDK to as well.
        assert.equal(stamp.split(":")[0], "8e57238c214f215939fb6a466b9791aae8abbfd7df14e752f9d0d767aaed3ad9");
      });

      it("sends what the code set beside the directory, which wins over its files", async () => {
        const root = await agentDir({ "agent.yaml": "name: jean\nllm: llm-smart\n", "instructions.md": "Be brief.\n" });
        const written = {
          instructions: "Be warm.",
          guardrail: "Refuse medical advice.",
          pipeline: { llm: "llm-fast", tts: "sonic_36", greeting: "Hello", greetingMode: "variation" as const, video: { max_frames: 4 } },
          costTracking: { team: "support", customer: "123" },
          harness: {
            name: "default" as const,
            subagents: { default: "llm-slow" },
            vm: daytona(),
            skills: [
              { name: "think", description: "Think", instructions: "Work it out.", deadlineMs: 30_000 },
              { name: "fill", description: "Fill a form", instructions: "Run it.", files: { "fill.py": "print()\n" } },
            ],
          },
        };
        serveSync();

        await new Agent({ folder: await loadFolder(root), client: api, ...written }).sync();
        const again = await new Agent({ folder: await loadFolder(root), client: api, ...written }).sync();

        assert.equal(again.unchanged, true);
        assert.equal(synced().length, 1);
        const [body] = synced();
        assert.equal(body?.instructions, "Be warm.");
        assert.equal(body?.guardrail, "Refuse medical advice.");
        assert.equal(body?.llm, "llm-fast", "what the code set wins over agent.yaml");
        assert.equal(body?.tts, "sonic_36");
        assert.deepEqual(body?.greeting, { text: "Hello", mode: "variation" });
        assert.deepEqual(body?.video, { max_frames: 4 });
        assert.deepEqual(body?.tags, { team: "support", customer: "123" });
        assert.equal(body?.harness, "default");
        assert.equal(body?.subagent, "llm-slow");
        assert.equal(body?.sandbox, "daytona");
        assert.deepEqual(body?.skills?.map((skill) => skill.name), ["think", "fill"]);
        assert.deepEqual(
          body?.files.map((file) => file.content),
          ["name: jean\nllm: llm-smart\n", "Be brief.\n"],
          "the files go as they are",
        );
      });

      it("sends an explicitly empty skill list, so the directory's skills are not the agent's", async () => {
        serveSync();
        const root = await agentDir({ "agent.yaml": "name: jean\n", "skills/think.md": "---\ndescription: Think\n---\nWork it out.\n" });

        await new Agent({ folder: await loadFolder(root), client: api, harness: { skills: [] } }).sync();

        assert.deepEqual(synced()[0]?.skills, []);
      });

      it("syncs a folder built by hand, with no files, as the fields it holds", async () => {
        router.serve("POST", "/v1/agents/sync", {
          body: { unchanged: false, config: { id: "cfg_1", name: "jean", created_at: "2026-01-01T00:00:00Z" } },
        });
        const folder: Folder = {
          path: "/agents/jean", name: "jean", instructions: "Be brief.", guardrail: "",
          settings: { llm: "llm-smart" }, skills: [], knowledge: [], knowledgeURLs: [],
        };

        await new Agent({ folder, client: api }).sync();

        const body = router.requestsTo("POST", "/v1/agents/sync")[0]?.body as Schemas["SyncAgentRequest"];
        assert.equal(body.llm, "llm-smart");
        assert.match(body.hash, /^[0-9a-f]{64}$/);
        assert.equal(synced().length, 0);
      });
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

  it("never prints the secret it signs with", () => {
    const edge = new Edge({ apiKey: "key", apiSecret: "shh-edge-secret" });

    for (const shown of [inspect(edge), JSON.stringify(edge)]) {
      assert.ok(!shown.includes("shh-edge-secret"), shown);
    }
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
