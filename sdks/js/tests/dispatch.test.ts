import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import {
  Agent,
  Client,
  ConfigurationError,
  Dispatch,
  HostingRefusedError,
  Tools,
  type InboundCall,
  type InboundMessage,
} from "../src/index.js";
import { TestRouter, type Connection } from "./router.js";

describe("Dispatch", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("refuses a worker that can hold no calls", () => {
    assert.throws(() => new Dispatch({ client: api, capacity: 0 }), ConfigurationError);
  });

  it("refuses to run with nothing to do the work", async () => {
    const dispatch = new Dispatch({ client: api });

    await assert.rejects(() => dispatch.run(), ConfigurationError);
  });

  it("promises the router what it can hold, and learns what it is called", async () => {
    const dispatch = new Dispatch({ client: api, capacity: 2 }).onCall(() => undefined);

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "ready", worker_id: "worker_7" });
    await settle();

    assert.equal(connection.query.get("capacity"), "2");
    assert.equal(connection.query.get("customer_id"), "local");
    assert.equal(dispatch.workerId, "worker_7");

    dispatch.stop();
    await running;
  });

  it("hands an arriving call to the handler and tells the router it was taken", async () => {
    const answered: InboundCall[] = [];
    const dispatch = new Dispatch({ client: api }).onCall((call) => {
      answered.push(call);
    });

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({
      type: "call",
      call_id: "call_1",
      call_type: "agent",
      called_number: "+15551234567",
      caller_number: "+15557654321",
      custom: { campaign: "spring" },
      at: "2026-01-01T00:00:00Z",
    });

    assert.deepEqual(await connection.next(), { type: "accepted", call_id: "call_1" });
    assert.equal(answered.length, 1);
    assert.equal(answered[0]?.callId, "call_1");
    assert.equal(answered[0]?.calledNumber, "+15551234567");
    assert.deepEqual(answered[0]?.custom, { campaign: "spring" });
    assert.equal(answered[0]?.at?.toISOString(), "2026-01-01T00:00:00.000Z");

    dispatch.stop();
    await running;
  });

  it("reads a call that says as little as the router can send", async () => {
    const answered: InboundCall[] = [];
    const dispatch = new Dispatch({ client: api }).onCall((call) => {
      answered.push(call);
    });

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "call", call_id: "call_1" });
    await connection.next();

    assert.equal(answered[0]?.callType, "default");
    assert.deepEqual(answered[0]?.custom, {});
    assert.equal(answered[0]?.at, undefined);

    dispatch.stop();
    await running;
  });

  it("tells the router about a call it could not answer, with the reason", async () => {
    const dispatch = new Dispatch({ client: api }).onCall(() => {
      throw new Error("no agent config called john");
    });

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "call", call_id: "call_1" });

    assert.deepEqual(await connection.next(), {
      type: "rejected",
      call_id: "call_1",
      reason: "no agent config called john",
    });

    dispatch.stop();
    await running;
  });

  it("answers the next caller while the first is still talking", async () => {
    const held: (() => void)[] = [];
    const dispatch = new Dispatch({ client: api }).onCall(
      () => new Promise<void>((resolve) => held.push(resolve)),
    );

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "call", call_id: "call_1" });
    connection.send({ type: "call", call_id: "call_2" });
    await settle();

    assert.equal(held.length, 2, "the second call was not waiting behind the first");
    assert.equal(dispatch.active, 2);

    for (const finish of held) {
      finish();
    }
    await settle();
    assert.equal(dispatch.active, 0);

    dispatch.stop();
    await running;
  });

  it("hands a message to its own handler and reports nothing back", async () => {
    const written: InboundMessage[] = [];
    const dispatch = new Dispatch({ client: api }).onMessage((message) => {
      written.push(message);
    });

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({
      type: "message",
      channel_id: "chan_1",
      channel_type: "messaging",
      agent_id: "john",
      config_id: "cfg_1",
      text: "are you open today",
      message_id: "msg_1",
      user_id: "ana",
      user_name: "Ana",
    });
    await settle();

    assert.equal(written.length, 1);
    assert.equal(written[0]?.channelId, "chan_1");
    assert.equal(written[0]?.text, "are you open today");
    assert.equal(written[0]?.agentId, "john");

    dispatch.stop();
    await running;
  });

  it("ignores work it has no handler for rather than ending the connection", async () => {
    const written: InboundMessage[] = [];
    const dispatch = new Dispatch({ client: api }).onMessage((message) => {
      written.push(message);
    });

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "call", call_id: "call_1" });
    connection.send({ type: "something_new" });
    connection.send({ type: "message", channel_id: "chan_1", text: "still here" });
    await settle();

    assert.equal(written.length, 1);
    assert.equal(written[0]?.text, "still here");

    dispatch.stop();
    await running;
  });

  it("measures the round trip itself and reports how it is doing", async () => {
    const dispatch = new Dispatch({ client: api, reportEveryMs: 10 }).onCall(() => undefined);

    const running = dispatch.run();
    const connection = await router.socket();

    const ping = await connection.next();
    assert.equal(ping["type"], "ping");
    assert.equal(typeof ping["at"], "number");
    connection.send({ type: "pong", at: ping["at"] });

    const load = await connection.next();
    assert.equal(load["type"], "load");
    assert.equal(load["active_agents"], 0);
    assert.ok(Number(load["latency_ms"]) >= 0);

    dispatch.stop();
    await running;
  });

  it("waits for work already being handled before it stops", async () => {
    let finished = false;
    const dispatch = new Dispatch({ client: api }).onCall(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
      finished = true;
    });

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "call", call_id: "call_1" });
    await settle();

    connection.socket.close();
    await running;

    assert.equal(finished, true, "a call in progress was hung up on");
  });

  it("returns when the router closes the connection", async () => {
    const dispatch = new Dispatch({ client: api }).onCall(() => undefined);

    const running = dispatch.run();
    const connection = await router.socket();
    connection.socket.close();

    await running;
  });

  it("stops waiting when the caller's signal aborts", async () => {
    const stop = new AbortController();
    const dispatch = new Dispatch({ client: api }).onCall(() => undefined);

    const running = dispatch.run(stop.signal);
    await router.socket();
    stop.abort();

    await running;
  });

  it("declares the tools it hosts once the router is listening, and runs the calls sent down", async () => {
    const tools = new Tools().register<{ sdk: string }>({
      name: "investigate_sdk",
      description: "Read SDK source",
      parameters: { type: "object", properties: { sdk: { type: "string" } } },
      run: ({ sdk }) => `read ${sdk}`,
    });
    const dispatch = new Dispatch({ client: api }).host("stream-support", tools, {
      timeoutMs: 60_000,
    });

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "ready", worker_id: "worker_1" });

    assert.deepEqual(await connection.next(), {
      type: "host_tools",
      agent_id: "stream-support",
      tools: [
        {
          name: "investigate_sdk",
          description: "Read SDK source",
          parameters: { type: "object", properties: { sdk: { type: "string" } } },
        },
      ],
      timeout_ms: 60_000,
    });

    connection.send({ type: "hosting", agent_id: "stream-support", tools: ["investigate_sdk"] });
    connection.send({
      type: "tool_call",
      id: "call_1",
      session_id: "sess_1",
      name: "investigate_sdk",
      arguments: '{"sdk":"android"}',
    });
    assert.deepEqual(await connection.next(), {
      type: "tool_result",
      id: "call_1",
      output: "read android",
    });

    dispatch.stop();
    await running;
  });

  it("declares its tools again every time the router says it is listening", async () => {
    const tools = new Tools().register({ name: "lookup", description: "Look up", run: () => "" });
    const dispatch = new Dispatch({ client: api }).host("stream-support", tools);

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "ready", worker_id: "worker_1" });
    const first = await connection.next();
    connection.send({ type: "ready", worker_id: "worker_1" });
    const second = await connection.next();

    assert.equal(first["type"], "host_tools");
    assert.equal(first["timeout_ms"], 0, "a timeout nobody set should leave the router's default");
    assert.deepEqual(second, first);

    dispatch.stop();
    await running;
  });

  it("keeps reading while a hosted tool runs, and counts it as work in flight", async () => {
    const held: (() => void)[] = [];
    const tools = new Tools().register({
      name: "slow",
      description: "Takes a while",
      run: () => new Promise<string>((resolve) => held.push(() => resolve("done"))),
    });
    const dispatch = new Dispatch({ client: api }).host("stream-support", tools);

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "tool_call", id: "call_1", name: "slow", arguments: "" });
    connection.send({ type: "tool_call", id: "call_2", name: "slow", arguments: "" });
    await settle();

    assert.equal(held.length, 2, "the second call was waiting behind the first");
    assert.equal(dispatch.active, 2);

    for (const finish of held) {
      finish();
    }
    const answered = [await connection.next(), await connection.next()];
    assert.deepEqual(answered.map((frame) => frame["id"]).sort(), ["call_1", "call_2"]);
    assert.equal(answered[0]?.["output"], "done");

    dispatch.stop();
    await running;
  });

  it("tells the router a hosted tool failed, since a model is waiting on it", async () => {
    const tools = new Tools().register({
      name: "investigate_sdk",
      description: "Read SDK source",
      run: () => {
        throw new Error("the checkout is missing");
      },
    });
    const dispatch = new Dispatch({ client: api }).host("stream-support", tools);

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({ type: "tool_call", id: "call_1", name: "investigate_sdk", arguments: "{}" });
    assert.deepEqual(await connection.next(), {
      type: "tool_result",
      id: "call_1",
      error: "the checkout is missing",
    });

    connection.send({ type: "tool_call", id: "call_2", name: "deploy", arguments: "{}" });
    assert.deepEqual(await connection.next(), {
      type: "tool_result",
      id: "call_2",
      error: "this worker does not run deploy",
    });

    dispatch.stop();
    await running;
  });

  it("stops waiting when the router refuses its tools, saying for which agent and why", async () => {
    const tools = new Tools().register({ name: "lookup", description: "Look up", run: () => "" });
    const dispatch = new Dispatch({ client: api }).host("stream-support", tools);

    const running = dispatch.run();
    const connection = await router.socket();
    connection.send({
      type: "hosting_refused",
      agent_id: "stream-support",
      reason: "hosting no tools is not hosting",
    });

    await assert.rejects(running, (error: unknown) => {
      assert.ok(error instanceof HostingRefusedError);
      assert.equal(error.agentId, "stream-support");
      assert.equal(error.reason, "hosting no tools is not hosting");
      assert.match(error.message, /stream-support.*hosting no tools is not hosting/);
      return true;
    });
  });

  it("answers the second message on a channel from the session that answered the first", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_1",
        call_id: "",
        call_type: "agent",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });

    const dispatch = new Dispatch({ client: api });
    const message: InboundMessage = {
      channelId: "chan_1",
      channelType: "messaging",
      agentId: "john",
      configId: "cfg_1",
      text: "hello",
      messageId: "msg_1",
      userId: "ana",
      userName: "Ana",
    };

    let built = 0;
    const create = () => {
      built += 1;
      return new Agent({ name: "John", client: api });
    };

    const first = dispatch.sessionFor(message, create);
    const connection = await router.socket();
    const session = await first;
    const again = await dispatch.sessionFor({ ...message, messageId: "msg_2" }, create);

    assert.equal(built, 1, "a second conversation was started on the same channel");
    assert.equal(again, session);

    const body = router.received.find((one) => one.path === "/v1/agents/sessions")?.body as {
      text: boolean;
      conversation_id: string;
      agent_id: string;
    };
    assert.equal(body.text, true);
    assert.equal(body.conversation_id, "messaging:chan_1");
    assert.equal(body.agent_id, "john");

    connection.socket.close();
    await session.wait();
  });
});

/** Lets the event loop deliver whatever is already on the socket. */
function settle(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 25));
}
