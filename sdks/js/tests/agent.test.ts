import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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
  type ConnectorDeclaration,
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

    it("sends a skill folder's files, and fingerprints a skill moved into a folder the same until it brings one", async () => {
      router.serve("POST", "/v1/agents/sync", {
        body: {
          unchanged: false,
          config: { id: "cfg_1", name: "jean", created_at: "2026-01-01T00:00:00Z" },
        },
      });
      // Nothing stored, so a stamp that still matches syncs again and the request is seen.
      router.serve("GET", "/v1/agents/configs", { body: [] });
      const root = join(await mkdtemp(join(tmpdir(), "vision-agents-")), "jean");
      const skill = "---\ndescription: Fill PDF forms\ndeadline: 2m\n---\nFill the form.\n";
      await mkdir(join(root, "skills"), { recursive: true });
      await writeFile(join(root, "agent.yaml"), "name: jean\n");
      await writeFile(join(root, "skills", "pdf.md"), skill);
      await new Agent({ folder: await loadFolder(root), client: api }).sync();

      await rm(join(root, "skills", "pdf.md"));
      await mkdir(join(root, "skills", "pdf", "scripts"), { recursive: true });
      await writeFile(join(root, "skills", "pdf", "SKILL.md"), skill);
      await new Agent({ folder: await loadFolder(root), client: api }).sync();

      await writeFile(join(root, "skills", "pdf", "scripts", "fill.py"), "print('filled')\n");
      await new Agent({ folder: await loadFolder(root), client: api }).sync();

      const [asFile, asFolder, withFile] = router
        .requestsTo("POST", "/v1/agents/sync")
        .map((one) => one.body as Schemas["SyncAgentRequest"]);
      assert.equal(asFolder?.skills?.[0] && "files" in asFolder.skills[0], false);
      assert.deepEqual(withFile?.skills?.[0]?.files, { "scripts/fill.py": "print('filled')\n" });
      assert.equal(asFile?.hash, asFolder?.hash, "moving a skill into a folder would sync it again");
      assert.notEqual(asFolder?.hash, withFile?.hash, "a file added to a skill would not sync");
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

    describe("mcp_servers and channels", () => {
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

      it("sends them the way the Go SDK does", async () => {
        const body = await synced(
          [
            "name: jean",
            "mcp_servers:",
            "  - name: crm",
            '    url: "https://crm.example.com/mcp"',
            "    tools: [list_*, get_deal]",
            "    scopes:",
            "      - read",
            "    user: true",
            "  - name: notes",
            "    url: https://notes.example.com/mcp",
            "    tools: []",
            "    user: false",
            "channels:",
            "  whatsapp:",
            '    number: "+15551234567" # the main line',
            "  sms:",
            "    number: +15557654321",
            "  imessage:",
            "  identity: link # by link",
            "",
          ].join("\n"),
        );

        assert.deepEqual(body["mcp_servers"], [
          {
            name: "crm",
            url: "https://crm.example.com/mcp",
            tools: ["list_*", "get_deal"],
            scopes: ["read"],
            user: true,
          },
          { name: "notes", url: "https://notes.example.com/mcp" },
        ]);
        assert.deepEqual(body["channels"], {
          whatsapp: { number: "+15551234567" },
          sms: { number: "+15557654321" },
          identity: "link",
        });
      });

      it("sends no mcp_servers for an empty list, and empty channels for an empty mapping", async () => {
        const body = await synced("name: jean\nmcp_servers: []\nchannels: {}\n");

        assert.equal("mcp_servers" in body, false);
        assert.deepEqual(body["channels"], {});
      });

      it("fingerprints a change to either", async () => {
        const one = await synced("name: jean\nchannels:\n  sms:\n    number: +15550000001\n");
        const other = await synced("name: jean\nchannels:\n  sms:\n    number: +15550000002\n");
        const servers = await synced("name: jean\nmcp_servers:\n  - name: a\n    url: https://a.example.com\n");

        assert.notEqual(one["hash"], other["hash"]);
        assert.notEqual(one["hash"], servers["hash"]);
      });

      it("sends how the sandbox is built and what it may reach", async () => {
        const body = await synced(
          [
            "name: jean",
            "sandbox: daytona",
            "sandbox_options:",
            "  setup:",
            "    - pip install bpy==5.2.2",
            "  timeout: 5m",
            "  memory_gb: 4",
            "  network: true",
            "  allowed_domains: [api.example.com]",
            "  network_allow_list: [10.0.0.0/8]",
            "",
          ].join("\n"),
        );

        assert.deepEqual(body["sandbox_options"], {
          setup: ["pip install bpy==5.2.2"],
          timeout_ms: 300_000,
          memory_gb: 4,
          network: true,
          allowed_domains: ["api.example.com"],
          network_allow_list: ["10.0.0.0/8"],
        });
      });

      it("sends the sandbox's snapshot and region", async () => {
        const body = await synced(
          "name: jean\nsandbox: daytona\nsandbox_options:\n  snapshot: daytona-large\n  region: eu\n",
        );

        assert.deepEqual(body["sandbox_options"], { snapshot: "daytona-large", region: "eu" });
      });

      it("sends the same body as before for an agent.yaml that names neither", async () => {
        const body = await synced("name: jean\nllm: llm-fast\nplugins:\n  - weather\n");

        assert.deepEqual(body, {
          name: "jean",
          instructions: "Be brief.",
          llm: "llm-fast",
          plugins: ["weather"],
          // Probed on the base, before agent.yaml could declare either.
          hash: "693edb6e1c1bf06d590d73d9b06ec2b7defcb319627b098b9a0c1f56e2b38561",
        });
      });
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
