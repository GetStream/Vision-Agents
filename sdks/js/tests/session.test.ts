import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Client, ConfigurationError, Session, Socket, Tools } from "../src/index.js";
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
    const opening = Session.open(api, { start_voice: true }, tools ? { tools } : {});
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
    const opening = Session.open(api, { start_voice: true }, { interim: true });
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

  it("stops the session in the backend when the socket cannot be watched", async () => {
    const unwatchable = new Client({ url: router.url, customerId: "local" });
    router.serve("POST", "/v1/agents/sessions/sess_1/stop", { status: 204 });
    await router.stop();

    await assert.rejects(() => Session.open(unwatchable, { start_voice: true }));
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

  it("names the durable request and turn a tool was asked for by, so the result is taken", async () => {
    const tools = new Tools().register({ name: "lookup", description: "Look it up", run: () => "found" });
    const [session, connection] = await opened(tools);

    connection.send({
      type: "tool_call",
      id: "call_1",
      name: "lookup",
      arguments: "{}",
      request_id: "request-a",
      turn_id: "turn-a",
    });

    assert.deepEqual(await connection.next(), {
      type: "tool_result",
      tool_call_id: "call_1",
      request_id: "request-a",
      turn_id: "turn-a",
      output: "found",
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

  it("leaves a call for a tool held elsewhere to whoever holds it", async () => {
    const tools = new Tools().register({ name: "lookup", description: "Look it up", run: () => "found" });
    const [session, connection] = await opened(tools);

    // Every watcher is sent every call and the first answer wins, so answering a tool this
    // session does not hold would beat the backend that runs it.
    connection.send({ type: "tool_call", id: "call_1", name: "book_flight", arguments: "{}" });
    connection.send({ type: "tool_call", id: "call_2", name: "lookup", arguments: "{}" });

    assert.deepEqual(await connection.next(), { type: "tool_result", tool_call_id: "call_2", output: "found" });
    await session.close();
  });

  it("answers no call at all when it holds no tools", async () => {
    const [session, connection] = await opened();

    connection.send({ type: "tool_call", id: "call_1", name: "book_flight", arguments: "{}" });
    connection.send({ type: "joined" });
    // Frames are read in order, so once joined is read the call before it has been seen.
    for await (const event of session.events()) {
      if (event.kind === "joined") {
        break;
      }
    }
    session.say("one moment");

    assert.deepEqual(await connection.next(), { type: "say", text: "one moment" });
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

  describe("a function that asks first", () => {
    const refundCall = {
      type: "tool_call",
      id: "call_1",
      name: "issue_refund",
      arguments: JSON.stringify({ order_id: "A-1042", reason: "it arrived unopened" }),
      request_id: "m1",
      turn_id: "t1",
    };
    let ran: string[];
    const refunds = (): Tools =>
      new Tools().register<{ order_id: string }>({
        name: "issue_refund",
        description: "Refund an order",
        approval: { title: "Refund order A-1042?", reasonArgument: "reason" },
        run: (input) => {
          ran.push(input.order_id);
          return `Refunded ${input.order_id}`;
        },
      });

    beforeEach(() => {
      ran = [];
    });

    async function waiting(session: Session): Promise<void> {
      for await (const event of session.events()) {
        if (event.kind === "tool_call") {
          return;
        }
      }
    }

    it("waits for the person, then runs once allowed", async () => {
      const [session, connection] = await opened(refunds());

      connection.send(refundCall);
      await waiting(session);
      assert.deepEqual(session.approvals, [
        {
          id: "call_1",
          name: "issue_refund",
          arguments: refundCall.arguments,
          approval: { title: "Refund order A-1042?", reasonArgument: "reason" },
          reason: "it arrived unopened",
        },
      ]);
      assert.deepEqual(ran, []);

      session.decide("call_1", { allowed: true });

      assert.deepEqual(session.approvals, []);
      assert.deepEqual(await connection.next(), {
        type: "tool_approval",
        tool_call_id: "call_1",
        allowed: true,
        request_id: "m1",
        turn_id: "t1",
      });
      assert.deepEqual(await connection.next(), {
        type: "tool_result",
        tool_call_id: "call_1",
        request_id: "m1",
        turn_id: "t1",
        output: "Refunded A-1042",
      });
      assert.deepEqual(ran, ["A-1042"]);
      await session.close();
    });

    it("declined, never runs, tells the model so, and a second answer does nothing", async () => {
      const [session, connection] = await opened(refunds());

      connection.send(refundCall);
      await waiting(session);
      session.decide("call_1", { allowed: false, summary: "Kept the order" });
      session.decide("call_1", { allowed: true });
      session.say("still there?");

      assert.deepEqual(await connection.next(), {
        type: "tool_approval",
        tool_call_id: "call_1",
        allowed: false,
        summary: "Kept the order",
        request_id: "m1",
        turn_id: "t1",
      });
      assert.deepEqual(await connection.next(), {
        type: "tool_result",
        tool_call_id: "call_1",
        request_id: "m1",
        turn_id: "t1",
        error: "The person declined this call.",
      });
      assert.deepEqual(await connection.next(), { type: "say", text: "still there?" });
      assert.deepEqual(ran, []);
      await session.close();
    });

    it("stops waiting when the model cancels the call", async () => {
      const [session, connection] = await opened(refunds());

      connection.send(refundCall);
      await waiting(session);
      connection.send({ type: "tool_cancel", id: "call_1" });
      connection.send({ type: "responded", text: "never mind" });
      for await (const event of session.events()) {
        if (event.kind === "responded") {
          break;
        }
      }

      assert.deepEqual(session.approvals, []);
      session.decide("call_1", { allowed: true });
      session.say("still there?");
      assert.deepEqual(await connection.next(), { type: "say", text: "still there?" });
      await session.close();
    });
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

  it("reads what is being heard off a speaking event, before the spoke that ends it", async () => {
    const [session, connection] = await opened();
    const at = "2026-10-22T09:30:00.123456789Z";

    connection.send({ type: "speaking", turn_id: "turn-1", synthesis_id: "turn-1#0", text: "Hello there.", at });
    connection.send({ type: "spoke", turn_id: "turn-1", synthesis_id: "turn-1#0", characters: 12, audio_duration_ms: 900 });

    const read = [];
    for await (const event of session.events()) {
      read.push(event);
      if (read.length === 2) {
        break;
      }
    }
    const [speaking, spoke] = read;
    assert.equal(speaking?.kind, "speaking");
    assert.equal(speaking?.text, "Hello there.");
    assert.equal(speaking?.frame["at"], at);
    assert.equal(spoke?.kind, "spoke");
    assert.equal(spoke?.frame["synthesis_id"], speaking?.frame["synthesis_id"]);
    assert.equal(spoke?.frame["characters"], 12);
    await session.close();
  });

  it("follows the voice off the socket without taking anything from events()", async () => {
    const [session, connection] = await opened();
    const told: (string | undefined)[] = [];
    session.speech.subscribe(() => told.push(session.speech.get()?.synthesisId));

    const frames = [
      { type: "response_delta", turn_id: "turn-1", text: "Hello there." },
      { type: "speaking", turn_id: "turn-1", synthesis_id: "turn-1", text: "Hello there.", at: "2026-10-22T09:30:00Z" },
      { type: "spoke", turn_id: "turn-1", synthesis_id: "turn-1", characters: 12, audio_duration_ms: 900 },
    ];
    for (const frame of frames) {
      connection.send(frame);
    }

    const read: string[] = [];
    for await (const event of session.events()) {
      read.push(event.kind);
      if (read.length === frames.length) {
        break;
      }
    }
    assert.deepEqual(read, ["response_delta", "speaking", "spoke"], "every event is still the caller's");
    // Held while its text came first, on a voice session; then heard, then measured.
    assert.deepEqual(told, [undefined, "turn-1", "turn-1"]);
    assert.equal(session.speech.heard("turn-1", performance.now() + 900)?.done, true);
    await session.close();
  });

  it("sends what the caller asked the conversation to do", async () => {
    const [session, connection] = await opened();

    session.say("one moment");
    assert.deepEqual(await connection.next(), { type: "say", text: "one moment" });

    session.say("actually", { interrupt: true });
    assert.deepEqual(await connection.next(), { type: "interrupt" });
    assert.deepEqual(await connection.next(), { type: "say", text: "actually" });

    await session.close();
  });

  it("takes the agent off the call and puts it back, carrying on one conversation", async () => {
    const [session] = await opened();
    const row = { id: "sess_1", user_id: "john", agent_id: "john", state: "live", created_at: "2026-01-01T00:00:00Z" };
    router.serve("DELETE", "/v1/agents/sessions/sess_1/voice", { body: row });
    router.serve("POST", "/v1/agents/sessions/sess_1/voice", { body: { ...row, call_id: "sess_1" } });

    assert.equal(session.voice.started, true, "a session that joined its call has voice on");
    await session.voice.stop();
    assert.equal(router.last.method, "DELETE");
    assert.equal(router.last.path, "/v1/agents/sessions/sess_1/voice");
    assert.equal(session.voice.started, false);

    await session.voice.start();
    assert.equal(router.last.method, "POST");
    assert.equal(router.last.path, "/v1/agents/sessions/sess_1/voice");
    assert.equal(session.voice.started, true);
    await session.close();
  });

  it("holds no reply for a voice once the agent is off the call", async () => {
    const [session, connection] = await opened();
    const row = { id: "sess_1", user_id: "john", agent_id: "john", state: "live", created_at: "2026-01-01T00:00:00Z" };
    router.serve("DELETE", "/v1/agents/sessions/sess_1/voice", { body: row });
    router.serve("POST", "/v1/agents/sessions/sess_1/voice", { body: { ...row, call_id: "sess_1" } });
    const deltas = (async () => {
      const kinds: string[] = [];
      for await (const event of session.events()) {
        kinds.push(event.kind);
        if (kinds.length === 3) {
          return kinds;
        }
      }
      return kinds;
    })();

    connection.send({ type: "speaking", turn_id: "turn-0", synthesis_id: "turn-0", text: "Hi." });
    await session.voice.stop();
    connection.send({ type: "response_delta", turn_id: "turn-1", text: "In writing now." });
    await session.voice.start();
    connection.send({ type: "response_delta", turn_id: "turn-2", text: "Back on the call." });
    await deltas;

    assert.equal(session.speech.heard("turn-1"), undefined, "a reply in writing is shown as it is");
    assert.deepEqual(session.speech.heard("turn-2"), { chars: 0, done: false });
    await session.close();
  });

  it("abandons the reply to one request, or whatever is being said", async () => {
    const [session, connection] = await opened();

    session.interrupt({ requestId: "req_2" });
    assert.deepEqual(await connection.next(), { type: "interrupt", request_id: "req_2" });
    session.interrupt();
    assert.deepEqual(await connection.next(), { type: "interrupt" });

    await session.close();
  });

  it("opens the channel on the chat client the app handed over, and leaves it connected", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_1",
        conversation_id: "agent:support-1",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });
    const opening = Session.open(api, {});
    await router.socket();
    const session = await opening;
    let connects = 0;
    let disconnects = 0;
    const client = {
      channel: (type: string, id: string) => ({ type, id }),
      connectUser: async () => {
        connects++;
      },
      disconnectUser: async () => {
        disconnects++;
      },
    };
    api.use({ chat: client });

    const chat = await session.chat();
    await api.disconnect();

    assert.equal(chat.client, client);
    assert.deepEqual(chat.channel, { type: "agent", id: "support-1" });
    assert.equal(connects, 0, "no second connection");
    assert.equal(disconnects, 0, "the app's client is the app's to disconnect");
    await session.close();
  });

  it("joins the call on the video client the app handed over", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_1",
        call_id: "sess_1",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });
    const opening = Session.open(api, { start_voice: true });
    await router.socket();
    const session = await opening;
    const client = {
      streamClient: { userID: "john" },
      call: (type: string, id: string) => ({ type, id }),
      disconnectUser: async () => undefined,
    };
    api.use({ video: client });

    const video = await session.video();

    assert.equal(video.client, client);
    assert.deepEqual(video.call, { type: "agent", id: "sess_1" });
    await session.close();
  });

  it("refuses a chat client the app connected as somebody else", async () => {
    const page = new Client({ url: router.url, customerId: "local", apiKey: "vak_live_x" });
    await page.setUser({ id: "john" }, "token-for-john");
    page.use({
      chat: {
        userID: "jane",
        channel: (type: string, id: string) => ({ type, id }),
        connectUser: async () => undefined,
        disconnectUser: async () => undefined,
      },
    });
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_1",
        conversation_id: "agent:support-1",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });
    const opening = Session.open(page, {});
    await router.socket();
    const session = await opening;

    await assert.rejects(session.chat(), (error: Error) => {
      assert.ok(error instanceof ConfigurationError);
      assert.match(error.message, /connected as jane but setUser named john/);
      return true;
    });
    await session.close();
  });

  it("says what chat needs before connecting a client with no Stream key", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_1",
        conversation_id: "agent:support-1",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });
    const opening = Session.open(api, {});
    await router.socket();
    const session = await opening;

    await assert.rejects(session.chat(), (error: Error) => {
      assert.ok(error instanceof ConfigurationError);
      assert.match(error.message, /pass apiKey and call setUser/);
      return true;
    });
    await session.close();
  });

  it("names the entry point to import when chat was never opted into", async () => {
    // Each test file runs in its own process, and this one never imports the chat entry.
    const page = new Client({ url: router.url, customerId: "local", apiKey: "vak_live_x" });
    await page.setUser({ id: "john" }, "token-for-john");
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_1",
        conversation_id: "agent:support-1",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });
    const opening = Session.open(page, {});
    await router.socket();
    const session = await opening;

    await assert.rejects(session.chat(), (error: Error) => {
      assert.ok(error instanceof ConfigurationError);
      assert.match(error.message, /import "@stream-io\/calypso\/chat"/);
      return true;
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
