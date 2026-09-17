import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Client, Session, Socket, Tools } from "../src/index.js";
import { TestRouter, type Connection } from "./router.js";

describe("Socket", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("yields the frames the router sends, in order", async () => {
    const opening = Socket.open(api.backend, await api.backend.socketURL("/v1/stt/stream"));
    const connection = await router.socket();
    const socket = await opening;

    connection.send({ type: "transcript", text: "one" });
    connection.send({ type: "transcript", text: "two" });

    const read: unknown[] = [];
    for await (const message of socket.messages()) {
      read.push(message);
      if (read.length === 2) {
        break;
      }
    }

    assert.deepEqual(read, [
      { type: "transcript", text: "one" },
      { type: "transcript", text: "two" },
    ]);
    socket.close();
  });

  it("hands audio back as bytes rather than as a frame", async () => {
    const opening = Socket.open(api.backend, await api.backend.socketURL("/v1/tts/stream"));
    const connection = await router.socket();
    const socket = await opening;

    connection.socket.send(Buffer.from([1, 2, 3]));

    for await (const message of socket.messages()) {
      assert.ok(message instanceof Uint8Array);
      assert.deepEqual([...message], [1, 2, 3]);
      break;
    }
    socket.close();
  });

  it("ends the stream when the router closes it", async () => {
    const opening = Socket.open(api.backend, await api.backend.socketURL("/v1/llm/stream"));
    const connection = await router.socket();
    const socket = await opening;

    connection.socket.close();

    const read: unknown[] = [];
    for await (const message of socket.messages()) {
      read.push(message);
    }
    assert.deepEqual(read, []);
    assert.equal(socket.open, false);
  });

  it("refuses a send on a socket that is no longer open", async () => {
    const opening = Socket.open(api.backend, await api.backend.socketURL("/v1/llm/stream"));
    await router.socket();
    const socket = await opening;
    socket.close();

    assert.throws(() => socket.send({ type: "respond" }));
  });

  it("reports an upgrade the router refused rather than resolving", async () => {
    await assert.rejects(() =>
      Socket.open(api.backend, "ws://127.0.0.1:1/v1/agents/sessions/x/events"),
    );
  });
});

describe("Session", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_1",
        call_id: "demo",
        call_type: "agent",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });
  });

  afterEach(async () => {
    await router.stop();
  });

  /** Opens a session and hands back both ends of it. */
  async function opened(tools?: Tools): Promise<[Session, Connection]> {
    const opening = Session.open(api, { call_id: "demo" }, tools ? { tools } : {});
    const connection = await router.socket();
    return [await opening, connection];
  }

  it("creates the session and watches the socket for it", async () => {
    const [session, connection] = await opened();

    assert.equal(session.id, "sess_1");
    assert.equal(connection.path, "/v1/agents/sessions/sess_1/events");
    assert.equal(connection.query.get("customer_id"), "local");
    await session.close();
  });

  it("asks for interim speech only when told to", async () => {
    const opening = Session.open(api, { call_id: "demo" }, { interim: true });
    const connection = await router.socket();
    const session = await opening;

    assert.equal(connection.query.get("interim"), "true");
    await session.close();
  });

  it("declares the registered tools on the session it creates", async () => {
    const tools = new Tools().register({
      name: "get_weather",
      description: "Get the weather somewhere",
      parameters: { type: "object", properties: { city: { type: "string" } } },
      run: () => "sunny",
    });

    const [session] = await opened(tools);

    const body = router.last.body as { tools: { name: string; description: string }[] };
    assert.equal(body.tools.length, 1);
    assert.equal(body.tools[0]?.name, "get_weather");
    await session.close();
  });

  it("closes the session in the backend when the socket cannot be watched", async () => {
    const unwatchable = new Client({ url: router.url, customerId: "local" });
    router.serve("DELETE", "/v1/agents/sessions/sess_1", { status: 204 });
    await router.stop();

    await assert.rejects(() => Session.open(unwatchable, { call_id: "demo" }));
  });

  it("hands the backend's events to whoever is reading them", async () => {
    const [session, connection] = await opened();

    connection.send({ type: "joined" });
    connection.send({
      type: "heard",
      text: "hello",
      participant: { id: "p1", user_id: "ana", name: "Ana" },
    });

    const read = [];
    for await (const event of session.events()) {
      read.push(event);
      if (read.length === 2) {
        break;
      }
    }

    assert.equal(read[0]?.kind, "joined");
    assert.equal(read[1]?.text, "hello");
    assert.deepEqual(read[1]?.participant, { id: "p1", userId: "ana", name: "Ana" });
    await session.close();
  });

  it("runs a tool the model asked for and answers with what it returned", async () => {
    const tools = new Tools().register<{ city: string }>({
      name: "get_weather",
      description: "Get the weather somewhere",
      run: (input) => ({ city: input.city, sky: "clear" }),
    });
    const [session, connection] = await opened(tools);

    connection.send({
      type: "tool_call",
      id: "call_1",
      name: "get_weather",
      arguments: JSON.stringify({ city: "Boulder" }),
    });

    const answer = await connection.next();
    assert.deepEqual(answer, {
      type: "tool_result",
      tool_call_id: "call_1",
      output: JSON.stringify({ city: "Boulder", sky: "clear" }),
    });
    await session.close();
  });

  it("tells the model a tool did not work, since it is mid-sentence waiting", async () => {
    const tools = new Tools().register({
      name: "lookup",
      description: "Look something up",
      run: () => {
        throw new Error("the database is down");
      },
    });
    const [session, connection] = await opened(tools);

    connection.send({ type: "tool_call", id: "call_1", name: "lookup", arguments: "{}" });

    assert.deepEqual(await connection.next(), {
      type: "tool_result",
      tool_call_id: "call_1",
      error: "the database is down",
    });
    await session.close();
  });

  it("reports a tool the model asked for that nothing is registered as", async () => {
    const [session, connection] = await opened(new Tools());

    connection.send({ type: "tool_call", id: "call_1", name: "unknown", arguments: "{}" });

    const answer = await connection.next();
    assert.match(String(answer["error"]), /nothing is registered as unknown/);
    await session.close();
  });

  it("aborts a running tool when the model cancels it", async () => {
    let aborted = false;
    const tools = new Tools().register({
      name: "slow",
      description: "Take a while",
      run: (_input, signal) =>
        new Promise((resolve) => {
          signal.addEventListener("abort", () => {
            aborted = true;
            resolve("abandoned");
          });
        }),
    });
    const [session, connection] = await opened(tools);

    connection.send({ type: "tool_call", id: "call_1", name: "slow", arguments: "{}" });
    await new Promise((resolve) => setTimeout(resolve, 20));
    connection.send({ type: "tool_cancel", id: "call_1" });

    await connection.next();
    assert.equal(aborted, true);
    await session.close();
  });

  it("does not hand a tool call to whoever is reading events", async () => {
    const tools = new Tools().register({
      name: "ping",
      description: "Answer",
      run: () => "pong",
    });
    const [session, connection] = await opened(tools);

    connection.send({ type: "tool_call", id: "call_1", name: "ping", arguments: "{}" });
    connection.send({ type: "responded", text: "done" });

    for await (const event of session.events()) {
      assert.equal(event.kind, "responded");
      break;
    }
    await session.close();
  });

  it("sends what the caller asked the conversation to do", async () => {
    const [session, connection] = await opened();

    session.say("one moment");
    assert.deepEqual(await connection.next(), { type: "say", text: "one moment" });

    session.respond("what is the weather", { interrupt: true });
    assert.deepEqual(await connection.next(), { type: "interrupt" });
    assert.deepEqual(await connection.next(), {
      type: "respond",
      text: "what is the weather",
    });

    session.setInstructions("be brief");
    assert.deepEqual(await connection.next(), {
      type: "instructions",
      instructions: "be brief",
    });

    await session.close();
  });

  it("ends the conversation over the socket it is being watched on", async () => {
    const [session, connection] = await opened();

    const closing = session.close();
    assert.deepEqual(await connection.next(), { type: "close" });
    await closing;

    assert.equal(session.live, false);
  });

  it("stops waiting when the conversation ends", async () => {
    const [session, connection] = await opened();

    connection.socket.close();

    await session.wait();
    assert.equal(session.live, false);
  });
});
