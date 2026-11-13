import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";

import { AgentChat, agentChatTransport, type AgentTransportFeed } from "../src/chat.js";
import {
  Client,
  ConfigurationError,
  Conversation,
  Session,
  mergeOpenUIStatements,
} from "../src/index.js";
import { loadFolder } from "../src/node.js";
import { TestRouter } from "./router.js";

describe("Conversation dynamic UI streaming", () => {
  it("folds ui_delta and ui_completed frames onto the agent turn and merges openui patches", () => {
    const conversation = new Conversation(true);

    conversation.observe({ type: "responding", turn_id: "t1" }, 100);
    conversation.observe({ type: "response_delta", turn_id: "t1", text: "Table at 19:15." }, 110);
    conversation.observe(
      { type: "ui_delta", turn_id: "t1", format: "openui", delta: '$party = 2\nroot = Card([hdr])\n' },
      120,
    );
    assert.deepEqual(conversation.get().turns[0]?.ui, {
      format: "openui",
      status: "streaming",
      content: "$party = 2\nroot = Card([hdr])\n",
    });

    conversation.observe(
      {
        type: "ui_completed",
        turn_id: "t1",
        format: "openui",
        content: '$party = 2\nroot = Card([hdr])\nhdr = CardHeader("Kaneo")',
      },
      150,
    );
    conversation.observe({ type: "responded", turn_id: "t1", text: "Table at 19:15." }, 160);
    assert.deepEqual(conversation.get().turns[0]?.ui, {
      format: "openui",
      status: "complete",
      content: '$party = 2\nroot = Card([hdr])\nhdr = CardHeader("Kaneo")',
    });

    conversation.observe({ type: "responding", turn_id: "t2" }, 200);
    conversation.observe(
      { type: "ui_delta", turn_id: "t2", format: "openui", mode: "patch", delta: "$party = 4" },
      210,
    );
    assert.deepEqual(conversation.get().turns[1]?.ui, {
      format: "openui",
      status: "streaming",
      content: '$party = 4\nroot = Card([hdr])\nhdr = CardHeader("Kaneo")',
    });

    conversation.observe(
      { type: "ui_completed", turn_id: "t2", format: "openui", mode: "patch", content: "$party = 4" },
      220,
    );
    assert.equal(conversation.get().turns[1]?.ui?.status, "complete");
  });

  it("removes component statements assigned null while keeping reactive $variables", () => {
    const merged = mergeOpenUIStatements(
      '$note = "Window"\nroot = Card([hdr, banner])\nhdr = CardHeader("Kaneo")\nbanner = Callout("info", "Busy", "12m")',
      "$note = null\nbanner = null\nroot = Card([hdr])",
    );
    assert.equal(merged, '$note = null\nroot = Card([hdr])\nhdr = CardHeader("Kaneo")');
  });
});

describe("AgentChat fast-lane LiveUI overlay over real WebSocket", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_ui",
        call_id: "sess_ui",
        call_type: "agent",
        conversation_id: "agent:sess_ui",
        user_id: "alex",
        agent_id: "juno",
        modality: "text",
        state: "live",
        created_at: "2026-11-12T00:00:00Z",
      },
    });
    router.serve("POST", "/v1/agents/sessions/sess_ui/responses", {
      status: 201,
      body: { id: "resp_1", session_id: "sess_ui", status: "in_progress", created_at: "2026-11-12T00:00:00Z" },
    });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("streams 0ms WebSocket ui_delta into adapted Chat messages and hands off cleanly when Chat settles", async () => {
    type WireMessage = { id: string; text?: string; custom?: Record<string, unknown> };
    const handlers = new Set<(event: { type: string; message?: WireMessage }) => void>();
    const channel = {
      state: { messages: [] as WireMessage[] },
      on(handler: (event: { type: string; message?: WireMessage }) => void) {
        handlers.add(handler);
        return { unsubscribe: () => void handlers.delete(handler) };
      },
      async watch() {
        return { messages: this.state.messages };
      },
      async query() {
        return { messages: [] };
      },
      getClient() {
        return { on: () => ({ unsubscribe() {} }) };
      },
      push(type: string, message: WireMessage) {
        if (type === "message.new") {
          this.state.messages = [...this.state.messages, message];
        } else if (type === "message.updated") {
          this.state.messages = this.state.messages.map((m) => (m.id === message.id ? message : m));
        }
        for (const h of handlers) h({ type, message });
      },
    };

    const opening = Session.open(api, { start_voice: false });
    const connection = await router.socket();
    const opened = await opening;

    const chat = new AgentChat<WireMessage>({
      sessionId: "sess_ui",
      open: async () => {
        const proxy = Object.create(opened) as Session;
        proxy.chat = async () => ({ client: {} as never, channel: channel as never });
        proxy.close = async () => {};
        return proxy;
      },
    });
    await chat.prepare();

    const rendered: WireMessage[][] = [];
    const feed: AgentTransportFeed<WireMessage> = {
      isConnected: true,
      updateMessages: (msgs) => rendered.push([...msgs]),
      updateHasOlderMessages: () => {},
      updateActivity: () => {},
    };
    const transport = agentChatTransport(chat, { adapt: (msg) => msg });
    const disconnect = transport.connect(feed);

    channel.push("message.new", {
      id: "reply_1",
      text: "",
      custom: { ai_generated: true, generating: true },
    });

    connection.send({ type: "responding", turn_id: "turn_fast" });
    connection.send({
      type: "ui_delta",
      turn_id: "turn_fast",
      format: "openui",
      delta: 'root = Card([hdr])\nhdr = CardHeader("Kaneo")',
    });

    await waitUntil(() => {
      const last = rendered.at(-1)?.[0];
      const ui = last?.custom?.["ai_ui"] as { status?: string; content?: string } | undefined;
      return ui?.status === "streaming" && ui.content?.includes("Kaneo") === true;
    });

    const settledReply: WireMessage = {
      id: "reply_1",
      text: "Here is Kaneo.",
      custom: {
        ai_generated: true,
        generating: false,
        ai_ui: {
          v: 1,
          format: "openui",
          status: "complete",
          content: 'root = Card([hdr])\nhdr = CardHeader("Kaneo")',
        },
      },
    };
    channel.push("message.updated", settledReply);

    assert.equal(
      rendered.at(-1)?.[0],
      settledReply,
      "once Stream Chat's durable ai_ui is complete, the exact settled message is handed over untouched",
    );

    disconnect();
    await chat.close();
    await opened.close();
  });
});

describe("loadFolder declarative ui catalog and blueprints", () => {
  let root: string;

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "va-js-ui-"));
  });

  afterEach(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it("compiles deterministic instructions from ui allowlist, portable components, and blueprints", async () => {
    const dir = join(root, "concierge");
    await mkdir(join(dir, "ui"), { recursive: true });
    await writeFile(
      join(dir, "agent.yaml"),
      [
        "name: concierge",
        "llm: llm-fast",
        "ui:",
        "  format: openui",
        "  keep: [CardHeader, Form, Chips, FollowUpBlock]",
        "  patch_mode: true",
        "  voice_ui: true",
        "  rules:",
        "    - Always pair RestaurantCard with AvailabilityGauge.",
      ].join("\n"),
    );
    await writeFile(join(dir, "instructions.md"), "You are Juno.");
    await writeFile(
      join(dir, "ui", "components.json"),
      JSON.stringify([
        {
          name: "restaurant_card",
          description: "Venue summary card.",
          parameters: {
            type: "object",
            properties: { name: { type: "string" }, rating: { type: "number" } },
            required: ["name", "rating"],
            "x-order": ["name", "rating"],
          },
          events: [{ name: "inspect", description: "Tapping inspects the menu." }],
        },
      ]),
    );
    await writeFile(
      join(dir, "ui", "blueprints.yaml"),
      [
        "- name: venue_booking",
        "  when: booking a table",
        "  order: [RestaurantCard, Form, FollowUpBlock]",
        "  example: |",
        '    $slot = "19:15"',
        "    root = Card([venue, reserve, next])",
        '    venue = RestaurantCard("Kaneo", 4.9)',
        '    reserve = Form("book", btns, [slotField])',
        '    slotField = FormControl("Slot", Chips("slot", "single", [ChipItem("19:15", "19:15")]))',
        '    btns = Buttons([Button("Book", @ToAssistant("Book", {slot: $slot}), "primary")])',
        '    next = FollowUpBlock([FollowUpItem("Menu")])',
      ].join("\n"),
    );

    const first = await loadFolder(dir);
    const second = await loadFolder(dir);
    assert.equal(first.instructions, second.instructions);
    assert.match(first.instructions, /RestaurantCard\(name: string, rating: number\)/);
    assert.match(first.instructions, /\*\*venue_booking\*\*/);
    assert.match(first.instructions, /openui-patch/);
    assert.doesNotMatch(first.instructions, /BarChart\(/);
  });

  it("refuses a ui components path that leaves the agent directory", async () => {
    const dir = join(root, "escaped");
    await mkdir(dir, { recursive: true });
    await writeFile(join(root, "outside.json"), "[]");
    await writeFile(join(dir, "agent.yaml"), "name: escaped\nui:\n  components: ../outside.json\n");

    await assert.rejects(() => loadFolder(dir), ConfigurationError);
  });
});

async function waitUntil(predicate: () => boolean, timeoutMs = 2000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) {
      throw new Error("Timed out waiting for condition");
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
