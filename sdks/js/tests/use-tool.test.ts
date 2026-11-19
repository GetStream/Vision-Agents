import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Client, Session, Tools, type Tool } from "../src/index.js";
import { offerTool } from "../src/react.js";
import { TestRouter, type Connection, type Received } from "./router.js";
import { until } from "./stream-chat.js";

const TOOLS = "/v1/agents/sessions/sess_1/tools";

/**
 * What `useTool` does, driven the way React drives its effect: `mount` runs it, the function it
 * answers is the unmount, and a re-render only points `latest` at the new tool. The hook itself
 * is not rendered, since running an effect needs a DOM this suite has none of.
 */
describe("useTool", () => {
  let router: TestRouter;
  let session: Session;
  let connection: Connection;
  /** The names the router offers the model, kept the way it keeps them. */
  let offered: Set<string>;
  let failures: Error[];
  let queue: { current: Promise<void> };

  beforeEach(async () => {
    router = await TestRouter.start();
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_1",
        call_id: "sess_1",
        call_type: "agent",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });
    offered = new Set();
    failures = [];
    queue = { current: Promise.resolve() };
    router.serve("POST", TOOLS, ({ body }) => {
      const names = (body as { tools: { name: string }[] }).tools.map(({ name }) => name);
      if (names.some((name) => offered.has(name))) {
        return taken;
      }
      names.forEach((name) => offered.add(name));
      return { status: 204 };
    });
    router.serve("DELETE", `${TOOLS}/read_cart`, () => (offered.delete("read_cart") ? { status: 204 } : missing));
  });

  afterEach(async () => {
    await session.close();
    await router.stop();
  });

  async function open(tools?: Tools): Promise<void> {
    const opening = Session.open(new Client({ url: router.url, customerId: "local" }), { start_voice: true }, tools ? { tools } : {});
    connection = await router.socket();
    session = await opening;
  }

  function mount(latest: { current: Tool<Record<string, never>> }): () => void {
    return offerTool(session, latest, queue, (cause) => failures.push(cause));
  }

  async function call(id: string): Promise<unknown> {
    connection.send({ type: "tool_call", id, name: "read_cart", arguments: "{}" });
    const answered = await connection.next();
    assert.equal(answered["tool_call_id"], id);
    return answered["output"] ?? answered["error"];
  }

  function cart(items: string[]): Tool<Record<string, never>> {
    return { name: "read_cart", description: "What is in the cart", run: () => items.join(", ") };
  }

  it("offers the tool once and runs the latest closure after a re-render, so new state needs no re-add", async () => {
    await open();
    const latest = { current: cart(["socks"]) };
    mount(latest);
    await queue.current;

    latest.current = cart(["socks", "shoes"]);

    assert.equal(await call("call-1"), "socks, shoes");
    assert.deepEqual([...offered], ["read_cart"]);
    assert.equal(router.requestsTo("POST", TOOLS).length, 1);
    assert.deepEqual(failures, []);
  });

  it("withdraws the tool on unmount, so the model is no longer offered it", async () => {
    await open();
    const unmount = mount({ current: cart(["socks"]) });
    await queue.current;

    unmount();
    await queue.current;

    assert.deepEqual([...offered], []);
    assert.deepEqual(failures, []);
  });

  it("still withdraws a tool unmounted while its add was in flight", async () => {
    await open();
    let answer: (reply: { status: number }) => void = () => undefined;
    router.serve("POST", TOOLS, () => {
      offered.add("read_cart");
      return new Promise((resolve) => (answer = resolve));
    });
    const unmount = mount({ current: cart(["socks"]) });
    await until(() => router.requestsTo("POST", TOOLS).length === 1);

    unmount();
    answer({ status: 204 });
    await queue.current;

    assert.deepEqual([...offered], []);
    assert.deepEqual(failures, []);
  });

  it("ends StrictMode's mount, unmount and mount with exactly one offer", async () => {
    await open();
    const latest = { current: cart(["socks"]) };
    mount(latest)();
    mount(latest);
    await queue.current;

    assert.deepEqual([...offered], ["read_cart"]);
    assert.deepEqual(
      router.received.filter(({ path }) => path.startsWith(TOOLS)).map(({ method }: Received) => method),
      ["POST", "DELETE", "POST"],
    );
    assert.equal(await call("call-1"), "socks");
    assert.deepEqual(failures, []);
  });

  it("reports a refused add and leaves the session's own tool of that name alone on unmount", async () => {
    await open(new Tools().register(cart(["the session's own"])));
    offered.add("read_cart");
    const unmount = mount({ current: cart(["the component's"]) });
    await queue.current;

    unmount();
    await queue.current;

    assert.equal(failures.length, 1);
    assert.match(failures[0]!.message, /taken/);
    assert.deepEqual([...offered], ["read_cart"]);
    assert.equal(router.requestsTo("DELETE", `${TOOLS}/read_cart`).length, 0);
    assert.equal(await call("call-1"), "the session's own");
  });
});

const taken = {
  status: 409,
  body: { error: { message: "read_cart is taken", type: "conflict", code: "tool_name_taken", doc_url: "" } },
};

const missing = {
  status: 404,
  body: { error: { message: "no tool read_cart", type: "not_found", code: "tool_not_found", doc_url: "" } },
};
