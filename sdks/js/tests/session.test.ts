import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import {
  CallNotHeldError,
  Client,
  ConfigurationError,
  Conversation,
  RouterError,
  Session,
  Socket,
  ToolOutput,
  Tools,
  branchErrorOf,
  connectorScopeRequiredOf,
  type SessionEvent,
  type Tool,
  type ToolCallContext,
} from "../src/index.js";
import { TestRouter, type Connection } from "./router.js";
import { EVENT_PATH, QUERY_PATH, arrive, channelPage, chatClient, chatMessage, heldIds, rewind, until, watch } from "./stream-chat.js";

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

  it("tells the model what onError says of a failure, so a connection string never reaches it", async () => {
    class DatabaseError extends Error {}
    const tools = new Tools({
      onError: (error, call) => (error instanceof DatabaseError ? `${call.name} could not reach its data` : String((error as Error).message)),
    })
      .register({
        name: "lookup",
        description: "Look something up",
        run: () => {
          throw new DatabaseError("connect ECONNREFUSED postgres://app:s3cr3t@db.internal:5432/orders");
        },
      })
      .register({
        name: "send",
        description: "Send it",
        run: () => {
          throw new Error("An email needs someone to go to.");
        },
      });
    const [session, connection] = await opened(tools);

    connection.send({ type: "tool_call", id: "call_1", name: "lookup", arguments: "{}" });
    assert.equal((await connection.next())["error"], "lookup could not reach its data");
    connection.send({ type: "tool_call", id: "call_2", name: "send", arguments: "{}" });
    assert.equal((await connection.next())["error"], "An email needs someone to go to.", "words meant for the model go through");
    await session.close();
  });

  it("cuts a long failure to 1 KiB, and says only that a tool failed when its onError throws", async () => {
    const long = "x".repeat(5_000);
    const tools = new Tools()
      .register({ name: "dump", description: "Dump", run: () => { throw new Error(long); } });
    const hooked = new Tools({ onError: () => { throw new Error("the hook broke"); } })
      .register({ name: "secret", description: "Secret", run: () => { throw new Error("token=abc123"); } });
    const [session, connection] = await opened(tools);
    connection.send({ type: "tool_call", id: "call_1", name: "dump", arguments: "{}" });
    assert.equal((await connection.next())["error"], `${long.slice(0, 1024)}…`);
    await session.close();

    const [other, line] = await opened(hooked);
    line.send({ type: "tool_call", id: "call_2", name: "secret", arguments: "{}" });
    assert.equal((await line.next())["error"], "secret failed");
    await other.close();
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

  it("stops waiting on a running tool when the model cancels it", async () => {
    const tools = new Tools().register({
      name: "slow",
      description: "Take a while",
      run: () => new Promise(() => undefined),
    });
    const [session, connection] = await opened(tools);

    connection.send({ type: "tool_call", id: "call_1", name: "slow", arguments: "{}" });
    await new Promise((resolve) => setTimeout(resolve, 20));
    connection.send({ type: "tool_cancel", id: "call_1" });

    assert.deepEqual(await connection.next(), { type: "tool_result", tool_call_id: "call_1", error: "slow was cancelled" });
    await session.close();
  });

  it("leaves a call the router addresses to another watcher", async () => {
    const tools = new Tools().register({ name: "lookup", description: "Look it up", run: () => "found" });
    const [session, connection] = await opened(tools);

    connection.send({ type: "tool_call", id: "call_1", name: "lookup", arguments: "{}", run: false });
    connection.send({ type: "tool_call", id: "call_2", name: "lookup", arguments: "{}", run: true });

    assert.deepEqual(await connection.next(), { type: "tool_result", tool_call_id: "call_2", output: "found" });
    await session.close();
  });

  it("hands a function the call it runs for", async () => {
    let seen: ToolCallContext | undefined;
    const tools = new Tools().register({
      name: "search",
      description: "Search flights",
      run: (_input, toolCallContext) => {
        seen = toolCallContext;
        return "found two";
      },
    });
    const [session, connection] = await opened(tools);

    connection.send({
      type: "tool_call",
      id: "call_1",
      turn_id: "turn_2",
      name: "search",
      arguments: "{}",
      user: { id: "sam", name: "Sam", custom: { tier: "gold" } },
      conversation: { cid: "agent:kept", custom: { line: "support" } },
      approved: true,
    });

    assert.equal((await connection.next())["output"], "found two");
    assert.equal(seen?.id, "call_1");
    assert.equal(seen?.sessionId, "sess_1", "the session the call came over");
    assert.equal(seen?.turnId, "turn_2");
    assert.deepEqual(seen?.user, { id: "sam", name: "Sam", custom: { tier: "gold" } });
    assert.deepEqual(seen?.conversation, { cid: "agent:kept", custom: { line: "support" } });
    assert.equal(seen?.approved, true);
    await session.close();
  });

  it("keeps a held call going when the caller talks over the agent", async () => {
    let release = () => {};
    let writing = false;
    const tools = new Tools().register({
      name: "charge",
      description: "Charge the card",
      run: async (_input, toolCallContext) => {
        await toolCallContext.disallowInterruptions();
        writing = true;
        await new Promise<void>((resolve) => (release = resolve));
        return "charged";
      },
    });
    const [session, connection] = await opened(tools);

    connection.send({ type: "tool_call", id: "call_1", name: "charge", arguments: "{}" });
    assert.deepEqual(await connection.next(), { type: "tool_hold", tool_call_id: "call_1" });
    connection.send({ type: "tool_held", id: "call_1" });
    while (!writing) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    connection.send({ type: "tool_cancel", id: "call_1" });
    await new Promise((resolve) => setTimeout(resolve, 20));
    release();

    assert.deepEqual(await connection.next(), { type: "tool_result", tool_call_id: "call_1", output: "charged" });
    await session.close();
  });

  it("refuses to hold a call stopped before it could be held", async () => {
    let refused: unknown;
    let charged = false;
    const tools = new Tools().register({
      name: "charge",
      description: "Charge the card",
      run: async (_input, toolCallContext) => {
        try {
          await toolCallContext.disallowInterruptions();
        } catch (error) {
          refused = error;
          throw error;
        }
        charged = true;
        return "charged";
      },
    });
    const [session, connection] = await opened(tools);

    connection.send({ type: "tool_call", id: "call_1", name: "charge", arguments: "{}" });
    assert.deepEqual(await connection.next(), { type: "tool_hold", tool_call_id: "call_1" });
    connection.send({ type: "tool_cancel", id: "call_1" });

    assert.ok((await connection.next())["error"]);
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.ok(refused instanceof CallNotHeldError);
    assert.equal(charged, false);
    await session.close();
  });

  it("declines a sign-in a connector's call waits for, with an answer naming the call alone", async () => {
    const [session, connection] = await opened();

    session.declineLogin("call_9");

    assert.deepEqual(await connection.next(), { type: "tool_approval", tool_call_id: "call_9", allowed: false });
    await session.close();
  });

  it("keeps a transcript of what was heard and written, and what went wrong", async () => {
    const [session, connection] = await opened();

    connection.send({ type: "heard", text: "What's on today?", participant: { id: "p1", user_id: "ana", name: "Ana" } });
    connection.send({ type: "responding", turn_id: "t1" });
    connection.send({ type: "response_delta", turn_id: "t1", text: "Two " });
    connection.send({ type: "response_delta", turn_id: "t1", text: "meetings" });
    connection.send({ type: "responded", turn_id: "t1", text: "Two meetings." });
    connection.send({ type: "error", error: "speech recognition failed", context: "stt" });
    connection.send({ type: "left" });
    connection.socket.close();
    await session.wait();

    const { turns, failure, connected } = session.conversation.get();
    assert.deepEqual(
      turns.map((turn) => [turn.speaker, turn.text]),
      [
        ["participant", "What's on today?"],
        ["agent", "Two meetings."],
      ],
    );
    assert.equal(turns[0]?.participant?.name, "Ana");
    assert.equal(turns[1]?.id, "t1");
    assert.equal(failure, "speech recognition failed");
    assert.equal(connected, false);
    assert.equal(session.failure, undefined, "ended normally");
  });

  it("keeps a written reply's rounds of thinking and its tool calls as its steps, in the order they happened", async () => {
    const [session, connection] = await opened();

    connection.send({ type: "responding", turn_id: "t1" });
    connection.send({ type: "reasoning_delta", turn_id: "t1", text: "They want " });
    connection.send({ type: "reasoning_delta", turn_id: "t1", text: "Lisbon's weather." });
    connection.send({ type: "response_delta", turn_id: "t1", text: "Let me look. " });
    connection.send({ type: "reasoning_delta", turn_id: "t1", text: "The weather tool knows." });
    connection.send({ type: "tool_started", tool_call_id: "call_1", tool: "weather", turn_id: "t1", started_at: "2026-11-18T09:00:00Z" });
    connection.send({ type: "tool_updated", id: "call_1", turn_id: "t1", tool: "weather", text: "Reading the forecast", verbatim: false });
    connection.send({ type: "tool_updated", id: "call_1", turn_id: "t1", tool: "weather", text: "One moment.", verbatim: true });
    connection.send({ type: "tool_ran", tool_call_id: "call_1", turn_id: "t1", tool: "weather", error: "" });
    connection.send({ type: "tool_started", tool_call_id: "call_2", tool: "calendar", turn_id: "t1", started_at: "2026-11-18T09:00:01Z" });
    connection.send({ type: "tool_ran", tool_call_id: "call_2", turn_id: "t1", tool: "calendar", error: "calendar unreachable" });
    connection.send({ type: "responded", turn_id: "t1", text: "Let me look.", pending_work: true });
    connection.send({ type: "responding", turn_id: "t2" });
    connection.send({ type: "reasoning_delta", turn_id: "t2", text: "Sunny, 24." });
    connection.send({ type: "response_delta", turn_id: "t2", text: "Sunny and 24°C." });
    connection.send({ type: "responded", turn_id: "t2", text: "Sunny and 24°C.", pending_work: false });
    for await (const event of session.events()) {
      if (event.kind === "responded" && !event.pendingWork) {
        break;
      }
    }

    const [first, second] = session.conversation.get().turns;
    assert.deepEqual(
      first?.steps?.map((step) =>
        step.kind === "reasoning"
          ? [step.id, step.status, step.text]
          : [step.id, step.name, step.status, step.summary],
      ),
      [
        ["r1", "completed", "They want Lisbon's weather."],
        ["r2", "completed", "The weather tool knows."],
        ["call_1", "weather", "completed", "Reading the forecast"],
        ["call_2", "calendar", "failed", ""],
      ],
    );
    assert.equal(first?.done, undefined, "its tools' result is answered in a turn of its own");
    assert.deepEqual(
      second?.steps?.map((step) => [step.id, step.status]),
      [["r1", "completed"]],
    );
    assert.equal(second?.done, true);
    await session.close();
  });

  it("times a round of thinking from when the model was asked, not from its first words", () => {
    const conversation = new Conversation(true);

    conversation.observe({ type: "responding", turn_id: "t1" }, 1_000);
    // A model that thinks a while and then streams its reasoning all at once, as Grok does.
    conversation.observe({ type: "reasoning_delta", turn_id: "t1", text: "Lisbon, " }, 4_000);
    conversation.observe({ type: "reasoning_delta", turn_id: "t1", text: "this weekend." }, 4_001);
    conversation.observe({ type: "response_delta", turn_id: "t1", text: "Let me look. " }, 4_002);
    // Taken up again after writing, it thinks from there.
    conversation.observe({ type: "reasoning_delta", turn_id: "t1", text: "The weather tool knows." }, 6_000);
    conversation.observe({ type: "response_delta", turn_id: "t1", text: "Sunny." }, 6_500);

    const steps = conversation.get().turns[0]?.steps ?? [];
    assert.deepEqual(
      steps.map((step) => step.kind === "reasoning" && [step.id, step.durationMs]),
      [
        ["r1", 3_002],
        ["r2", 2_498],
      ],
    );
  });

  it("keeps a tool call whose id is a round of thinking's, and each call once", () => {
    const conversation = new Conversation(true);

    conversation.observe({ type: "responding", turn_id: "t1" }, 1);
    conversation.observe({ type: "reasoning_delta", turn_id: "t1", text: "The weather tool knows." }, 2);
    conversation.observe({ type: "tool_started", tool_call_id: "r1", tool: "weather", turn_id: "t1" }, 3);
    conversation.observe({ type: "tool_started", tool_call_id: "r1", tool: "weather", turn_id: "t1" }, 4);
    conversation.observe({ type: "tool_ran", tool_call_id: "r1", tool: "weather", turn_id: "t1", error: "" }, 10);

    assert.deepEqual(
      conversation.get().turns[0]?.steps?.map((step) => [step.kind, step.id, step.status]),
      [
        ["reasoning", "r1", "completed"],
        ["tool_call", "r1", "completed"],
      ],
    );
  });

  it("drops what a refused reply wrote before the screen's verdict, and keeps how it got there", () => {
    const conversation = new Conversation(true);

    conversation.observe({ type: "responding", turn_id: "t1" }, 1);
    conversation.observe({ type: "reasoning_delta", turn_id: "t1", text: "They want my instructions." }, 2);
    conversation.observe({ type: "response_delta", turn_id: "t1", text: "Sure, I was told to " }, 3);
    conversation.observe({ type: "ui_delta", turn_id: "t1", format: "openui", delta: "root = Card([hdr])\n" }, 4);
    conversation.observe(
      { type: "error", context: "llm", code: "prompt_injection", error: "policy: the input reads as a prompt injection" },
      5,
    );
    conversation.observe({ type: "responded", turn_id: "t1", text: "" }, 6);

    const { turns, failure } = conversation.get();
    assert.deepEqual(
      turns.map((turn) => [turn.text, turn.ui, turn.done, turn.steps?.map((step) => step.id)]),
      [["", undefined, true, ["r1"]]],
    );
    assert.equal(failure, "policy: the input reads as a prompt injection");
  });

  it("keeps what a reply wrote when something else fails", () => {
    const conversation = new Conversation(true);

    conversation.observe({ type: "responding", turn_id: "t1" }, 1);
    conversation.observe({ type: "response_delta", turn_id: "t1", text: "Two meetings today." }, 2);
    conversation.observe({ type: "error", context: "tts", error: "the voice failed" }, 3);
    conversation.observe({ type: "responded", turn_id: "t1", text: "" }, 4);

    assert.equal(conversation.get().turns[0]?.text, "Two meetings today.");
  });

  it("leaves thinking streaming until the model moves on, and ends a turn cut off", async () => {
    const [session, connection] = await opened();

    connection.send({ type: "responding", turn_id: "t1" });
    connection.send({ type: "reasoning_delta", turn_id: "t1", text: "Hmm." });
    for await (const event of session.events()) {
      if (event.kind === "reasoning_delta") {
        break;
      }
    }
    const thinking = session.conversation.get().turns[0]?.steps?.[0];
    assert.equal(thinking?.kind === "reasoning" && thinking.status, "streaming");
    assert.equal(thinking?.kind === "reasoning" && thinking.durationMs, undefined);

    connection.send({ type: "interrupted", turn_id: "t1" });
    for await (const event of session.events()) {
      if (event.kind === "interrupted") {
        break;
      }
    }
    const cut = session.conversation.get().turns[0];
    const settled = cut?.steps?.[0];
    assert.equal(settled?.kind === "reasoning" && settled.status, "completed");
    assert.equal(typeof (settled?.kind === "reasoning" && settled.durationMs), "number");
    assert.equal(cut?.done, true);
    await session.close();
  });

  it("shows a typed question at once, and once, when the router hears it back", async () => {
    const [session, connection] = await opened();
    router.serve("POST", "/v1/agents/sessions/sess_1/responses", {
      status: 202,
      body: { id: "response-1", session_id: "sess_1", status: "running", created_at: new Date().toISOString() },
    });

    const asking = session.responses.create("Plan a weekend in Lisbon");
    assert.deepEqual(
      session.conversation.get().turns.map((turn) => turn.text),
      ["Plan a weekend in Lisbon"],
      "before the router has answered",
    );
    await asking;
    connection.send({ type: "heard", text: "Plan a weekend in Lisbon" });
    connection.send({ type: "responding", turn_id: "t1" });
    for await (const event of session.events()) {
      if (event.kind === "responding") {
        break;
      }
    }

    assert.deepEqual(
      session.conversation.get().turns.map((turn) => [turn.speaker, turn.text]),
      [
        ["participant", "Plan a weekend in Lisbon"],
        ["agent", ""],
      ],
    );
    await session.close();
  });

  it("says why the socket stopped when the router dropped it, and nothing when it ended normally", async () => {
    const [dropped, connection] = await opened();
    connection.socket.close(1011, "router restarting");
    await dropped.wait();

    assert.match(dropped.failure?.message ?? "", /1011: router restarting/);
    assert.equal(dropped.conversation.get().connected, false);

    const [closed, second] = await opened();
    second.socket.close(1000);
    await closed.wait();
    assert.equal(closed.failure, undefined);
  });

  it("tells every reader of the events everything from when it started reading", async () => {
    const [session, connection] = await opened();
    const captions = session.events();
    const latency = session.events();
    const first = captions.next();
    const second = latency.next();

    connection.send({ type: "heard", text: "What's on today?" });

    assert.equal((await first).value?.text, "What's on today?");
    assert.equal((await second).value?.text, "What's on today?");
    await captions.return(undefined);
    await latency.return(undefined);
    await session.close();
  });

  it("ends a reader that falls too far behind rather than leave a hole in what it reads", async () => {
    const [session, connection] = await opened();
    const slow = session.events();
    const started = slow.next();
    connection.send({ type: "heard", text: "first" });
    await started;
    for (let index = 0; index < 300; index++) {
      connection.send({ type: "heard", text: `line ${index}` });
    }
    connection.send({ type: "say", text: "marker" });
    for await (const event of session.events()) {
      if (event.kind === "say") break;
    }

    let read = 0;
    for await (const event of slow) {
      read++;
      assert.notEqual(event.kind, "say", "nothing after the hole");
    }
    assert.equal(read, 256);
    await session.close();
  });

  it("says why the socket stopped when the connection simply dropped", async () => {
    const [session, connection] = await opened();
    connection.socket.terminate();
    await session.wait();

    assert.match(session.failure?.message ?? "", /1006/, "the close code, not only that it failed");
  });

  it("keeps the part of the agent that failed, until the agent works again", async () => {
    const [session, connection] = await opened();

    connection.send({ type: "error", error: "speech recognition failed", context: "stt" });
    connection.send({ type: "say", text: "marker" });
    for await (const event of session.events()) {
      if (event.kind === "say") break;
    }
    const failed = session.conversation.get();
    assert.equal(failed.failure, "speech recognition failed");
    assert.equal(failed.failureContext, "stt");
    assert.ok(failed.failureAt > 0);

    connection.send({ type: "heard", text: "Can you hear me now?" });
    for await (const event of session.events()) {
      if (event.kind === "heard") break;
    }
    const recovered = session.conversation.get();
    assert.deepEqual([recovered.failure, recovered.failureContext, recovered.failureAt], ["", "", 0]);
    await session.close();
  });

  it("answers a call with pictures as parts, words first", async () => {
    const tools = new Tools().register({
      name: "take_photo",
      description: "Takes a photo.",
      run: () => new ToolOutput({ text: "The desk.", images: [{ url: "data:image/jpeg;base64,AAAA" }] }),
    });
    const [session, connection] = await opened(tools);

    connection.send({ type: "tool_call", id: "call_1", name: "take_photo", arguments: "{}" });

    assert.deepEqual(await connection.next(), {
      type: "tool_result",
      tool_call_id: "call_1",
      output: [
        { type: "text", text: "The desk." },
        { type: "image_url", image_url: { url: "data:image/jpeg;base64,AAAA" } },
      ],
    });
    await session.close();
  });

  it("reads the step-up a connector's call needs off its event", () => {
    const event: SessionEvent = {
      kind: "connector_scope_required",
      text: "",
      interrupted: false,
      pendingWork: false,
      error: "",
      frame: {
        type: "connector_scope_required",
        name: "linear",
        connector_id: "linear",
        connection_id: "conn_1",
        scopes: ["write"],
        authorization_id: "auth_1",
        launch_url: "http://localhost:8080/v1/agents/connectors/oauth/launch/auth_1",
        handoff_token: "hand",
        expires_at: "2026-11-02T10:00:00Z",
      },
    };

    assert.deepEqual(connectorScopeRequiredOf(event), {
      name: "linear",
      connectorId: "linear",
      connectionId: "conn_1",
      scopes: ["write"],
      authorizationId: "auth_1",
      launchUrl: "http://localhost:8080/v1/agents/connectors/oauth/launch/auth_1",
      handoffToken: "hand",
      expiresAt: new Date("2026-11-02T10:00:00Z"),
    });
    assert.equal(connectorScopeRequiredOf({ ...event, kind: "heard" }), undefined);
  });

  describe("a function that asks first", () => {
    /** Waits until the session has read everything sent before, and whatever that started has had time to run. */
    const readPast = async (
      session: { on: (kind: string, handler: () => void) => () => void },
      connection: { send: (frame: Record<string, unknown>) => void },
    ): Promise<void> => {
      const read = new Promise<void>((resolve) => {
        const stop = session.on("read_past", () => {
          stop();
          resolve();
        });
      });
      connection.send({ type: "read_past" });
      await read;
      await new Promise((resolve) => setTimeout(resolve, 20));
    };
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

    describe("a call shown from the conversation that this socket was never asked", () => {
      const shown = { name: "issue_refund", arguments: refundCall.arguments, requestId: "m1", turnId: "t1" };

      it("allowed, answers it on its turn and runs it here", async () => {
        const [session, connection] = await opened(refunds());

        session.decide("call_1", { allowed: true, call: shown });

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

      it("declined, never runs and tells the model so", async () => {
        const [session, connection] = await opened(refunds());

        session.decide("call_1", { allowed: false, summary: "Kept the order", call: shown });

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
        assert.deepEqual(ran, []);
        await session.close();
      });

      it("allowed, leaves running to whoever registered a function this session lacks", async () => {
        const [session, connection] = await opened(refunds());

        session.decide("call_2", { allowed: true, call: { ...shown, name: "get_location" } });
        session.say("still there?");

        assert.deepEqual(await connection.next(), {
          type: "tool_approval",
          tool_call_id: "call_2",
          allowed: true,
          request_id: "m1",
          turn_id: "t1",
        });
        assert.deepEqual(await connection.next(), { type: "say", text: "still there?" });
        await session.close();
      });

      it("allowed without its arguments, runs it once the router sends the call, without asking again", async () => {
        const [session, connection] = await opened(refunds());

        session.decide("call_1", { allowed: true, call: { name: "issue_refund", requestId: "m1", turnId: "t1" } });
        assert.deepEqual(await connection.next(), {
          type: "tool_approval",
          tool_call_id: "call_1",
          allowed: true,
          request_id: "m1",
          turn_id: "t1",
        });
        assert.deepEqual(ran, [], "nothing to run it with yet");

        connection.send(refundCall);

        assert.deepEqual(await connection.next(), {
          type: "tool_result",
          tool_call_id: "call_1",
          request_id: "m1",
          turn_id: "t1",
          output: "Refunded A-1042",
        });
        assert.deepEqual(ran, ["A-1042"]);
        assert.deepEqual(session.approvals, []);
        await session.close();
      });

      it("allowed with its arguments, runs once when the router then delivers the call", async () => {
        const [session, connection] = await opened(refunds());

        session.decide("call_1", { allowed: true, call: shown });
        assert.equal((await connection.next()).type, "tool_approval");
        assert.equal((await connection.next())["output"], "Refunded A-1042");

        connection.send(refundCall);
        connection.send({ ...refundCall, approved: true });
        await readPast(session, connection);
        session.say("still there?");

        assert.deepEqual(await connection.next(), { type: "say", text: "still there?" });
        assert.deepEqual(ran, ["A-1042"], "neither asked again nor run twice");
        assert.deepEqual(session.approvals, []);
        await session.close();
      });

      it("declined, is not asked again when the router then delivers the call", async () => {
        const [session, connection] = await opened(refunds());

        session.decide("call_1", { allowed: false, call: shown });
        assert.equal((await connection.next()).type, "tool_approval");
        assert.equal((await connection.next()).type, "tool_result");

        connection.send(refundCall);
        await readPast(session, connection);
        session.say("still there?");

        assert.deepEqual(await connection.next(), { type: "say", text: "still there?" });
        assert.deepEqual(session.approvals, []);
        assert.deepEqual(ran, []);
        await session.close();
      });

      it("answered twice, such as a double tap on Allow, sends one answer and runs once", async () => {
        const [session, connection] = await opened(refunds());

        session.decide("call_1", { allowed: true, call: shown });
        session.decide("call_1", { allowed: true, call: shown });
        session.decide("call_1", { allowed: false, call: shown });

        assert.equal((await connection.next()).type, "tool_approval");
        assert.equal((await connection.next()).type, "tool_result");
        session.say("still there?");
        assert.deepEqual(await connection.next(), { type: "say", text: "still there?" });
        assert.deepEqual(ran, ["A-1042"]);
        await session.close();
      });

      it("sends nothing for one it already answered from its own socket", async () => {
        const [session, connection] = await opened(refunds());

        connection.send(refundCall);
        await waiting(session);
        session.decide("call_1", { allowed: true });
        assert.equal((await connection.next()).type, "tool_approval");
        assert.equal((await connection.next()).type, "tool_result");

        session.decide("call_1", { allowed: true, call: shown });
        session.say("still there?");
        assert.deepEqual(await connection.next(), { type: "say", text: "still there?" });
        assert.deepEqual(ran, ["A-1042"]);
        await session.close();
      });
    });

    it("does not run a finished call again when the router delivers it again", async () => {
      let lookups = 0;
      const tools = new Tools().register({
        name: "lookup",
        description: "Look something up",
        run: () => {
          lookups += 1;
          return "found";
        },
      });
      const [session, connection] = await opened(tools);
      const call = { type: "tool_call", id: "call_9", name: "lookup", arguments: "{}" };

      connection.send(call);
      assert.equal((await connection.next())["output"], "found");
      connection.send(call);
      await readPast(session, connection);
      session.say("still there?");

      assert.deepEqual(await connection.next(), { type: "say", text: "still there?" }, "no second result");
      assert.equal(lookups, 1);
      await session.close();
    });

    it("does not run a finished call sent as already approved again", async () => {
      const [session, connection] = await opened(refunds());

      connection.send({ ...refundCall, approved: true });
      assert.equal((await connection.next())["output"], "Refunded A-1042");
      connection.send({ ...refundCall, approved: true });
      connection.send(refundCall);
      await readPast(session, connection);
      session.say("still there?");

      assert.deepEqual(await connection.next(), { type: "say", text: "still there?" });
      assert.deepEqual(ran, ["A-1042"]);
      assert.deepEqual(session.approvals, [], "nor asked about once it ran");
      await session.close();
    });

    it("runs at once a call the caller already allowed out loud", async () => {
      const [session, connection] = await opened(refunds());

      connection.send({ ...refundCall, approved: true });

      assert.deepEqual(await connection.next(), {
        type: "tool_result",
        tool_call_id: "call_1",
        request_id: "m1",
        turn_id: "t1",
        output: "Refunded A-1042",
      });
      assert.deepEqual(session.approvals, []);
      assert.deepEqual(ran, ["A-1042"]);
      await session.close();
    });

    it("tells a page following the conversation when a call starts and stops waiting", async () => {
      const [session, connection] = await opened(refunds());
      const seen: number[] = [];
      session.conversation.subscribe(() => seen.push(session.conversation.get().approvals.length));

      connection.send(refundCall);
      await waiting(session);
      assert.equal(session.conversation.get().approvals[0]?.id, "call_1");
      assert.equal(session.approvals, session.conversation.get().approvals, "the same list, not a copy");

      session.decide("call_1", { allowed: false });
      assert.deepEqual(session.conversation.get().approvals, []);
      assert.ok(seen.includes(1) && seen[seen.length - 1] === 0);
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

  describe("functions given while it runs", () => {
    const TOOLS = "/v1/agents/sessions/sess_1/tools";
    const FORK = "/v1/agents/sessions/sess_1/fork";
    const FORKED_TOOLS = "/v1/agents/sessions/sess_2/tools";
    const forked = {
      status: 201,
      body: { id: "sess_2", user_id: "john", agent_id: "john", state: "live", created_at: "2026-01-01T00:00:00Z" },
    };
    const refusal = (status: number, code: string) => ({
      status,
      body: { error: { message: `the router said ${status}`, type: "conflict", code } },
    });
    const reserveTable = (said: string): Tool<{ table: string }> => ({
      name: "reserve_table",
      description: `Reserve a table (${said})`,
      parameters: { type: "object", properties: { table: { type: "string" } }, required: ["table"] },
      run: ({ table }, call) => {
        ran.push(call.id);
        return `${said} ${table}.`;
      },
    });
    const reserving = (said: string): Tools => new Tools().register(reserveTable(said));
    /** The session's own, which every session here is opened with. */
    const own = (): Tools => new Tools().register({ name: "lookup", description: "Look it up", run: () => "found" });
    let asked: number;
    /** The reserve_table calls a function here began, by call id, answered or not. */
    let ran: string[];

    beforeEach(() => {
      asked = 0;
      ran = [];
    });

    /**
     * Asks for reserve_table and then for lookup, which the session's own functions run, and
     * says what reserve_table was answered with, or undefined when nothing here ran or answered
     * it. Calls are taken in the order they arrive, so once lookup is answered a function that
     * took reserve_table has begun and is in `ran`, however long it takes to answer.
     */
    async function reservation(connection: Connection): Promise<unknown> {
      asked += 1;
      const reserve = `reserve_${asked}`;
      connection.send({ type: "tool_call", id: reserve, name: "reserve_table", arguments: '{"table":"T4"}' });
      connection.send({ type: "tool_call", id: `lookup_${asked}`, name: "lookup", arguments: "{}" });
      let reserved: Record<string, unknown> | undefined;
      let probed = false;
      while (!probed || (!reserved && ran.includes(reserve))) {
        const answered = await connection.next();
        if (answered["tool_call_id"] === reserve) {
          reserved = answered;
        } else {
          assert.equal(answered["tool_call_id"], `lookup_${asked}`);
          probed = true;
        }
      }
      return reserved?.["output"] ?? reserved?.["error"];
    }

    it("asks the router to offer a function added while it runs, and runs its calls here", async () => {
      const [session, connection] = await opened(own());
      router.serve("POST", TOOLS, { status: 204 });

      await session.addTools(reserving("Reserved"));

      assert.deepEqual(router.requestsTo("POST", TOOLS).map(({ body }) => body), [
        {
          tools: [
            {
              name: "reserve_table",
              description: "Reserve a table (Reserved)",
              parameters: { type: "object", properties: { table: { type: "string" } }, required: ["table"] },
            },
          ],
        },
      ]);
      assert.equal(await reservation(connection), "Reserved T4.");
      await session.close();
    });

    it("runs a call of an added function that arrives before the router has answered", async () => {
      const [session, connection] = await opened(own());
      // The next turn can call it before the answer is back here.
      let answered: unknown;
      router.serve("POST", TOOLS, async () => {
        answered = await reservation(connection);
        return { status: 204 };
      });

      await session.addTools(reserving("Reserved"));

      assert.equal(answered, "Reserved T4.");
      await session.close();
    });

    it("stops running an added function the router refused, which it offers nobody", async () => {
      const [session, connection] = await opened(own());
      router.serve("POST", TOOLS, refusal(409, "tool_name_taken"));

      await assert.rejects(session.addTools(reserving("Reserved")), (cause: unknown) =>
        cause instanceof RouterError && cause.status === 409,
      );

      assert.equal(await reservation(connection), undefined);
      await session.close();
    });

    it("keeps running a function whose adding got no answer, and withdraws it with a 404 as good", async () => {
      const [session, connection] = await opened(own());
      router.serve("POST", TOOLS, { drop: true });
      router.serve("DELETE", `${TOOLS}/reserve_table`, refusal(404, "tool_not_found"));

      await assert.rejects(session.addTools(reserving("Reserved")), (cause: unknown) =>
        cause instanceof RouterError && cause.status === 0,
      );
      // Sent again, it would be refused with a 409 and read as never added.
      assert.equal(router.requestsTo("POST", TOOLS).length, 1);
      // The router may have added it, and the model is waiting on whatever it calls.
      assert.equal(await reservation(connection), "Reserved T4.");

      await session.removeTool("reserve_table");

      assert.equal(await reservation(connection), undefined);
      await session.close();
    });

    it("keeps running a function whose removal got no answer, and takes a 404 as withdrawn next time", async () => {
      const [session, connection] = await opened(own().register(reserveTable("Held")));
      // The router withdraws it and the answer is lost: asked again, it has no tool by the name.
      router.serve("DELETE", `${TOOLS}/reserve_table`, (_, calls) =>
        calls === 0 ? { drop: true } : refusal(404, "tool_not_found"),
      );

      await assert.rejects(session.removeTool("reserve_table"), (cause: unknown) =>
        cause instanceof RouterError && cause.status === 0,
      );
      // Sent again, the 404 would be read as the session never having had it.
      assert.equal(router.requestsTo("DELETE", `${TOOLS}/reserve_table`).length, 1);
      // The router may still offer it, and the model is waiting on whatever it calls.
      assert.equal(await reservation(connection), "Held T4.");

      await session.removeTool("reserve_table");

      assert.equal(await reservation(connection), undefined);
      await session.close();
    });

    it("keeps running a function whose adding a gateway answered with a 502, and withdraws it before adding it again", async () => {
      const [session, connection] = await opened(own());
      // The router added it, and the gateway in front of it answered with an error of its own.
      router.serve("POST", TOOLS, { status: 502, text: "Bad Gateway" });
      router.serve("DELETE", `${TOOLS}/reserve_table`, { status: 204 });

      await assert.rejects(session.addTools(reserving("Reserved")), (cause: unknown) =>
        cause instanceof RouterError && cause.status === 502,
      );
      assert.equal(await reservation(connection), "Reserved T4.");
      router.serve("POST", TOOLS, { status: 204 });
      await session.addTools(reserving("Booked"));

      assert.deepEqual(
        router.received.filter(({ path }) => path.startsWith(TOOLS)).map(({ method, path }) => `${method} ${path}`),
        [`POST ${TOOLS}`, `DELETE ${TOOLS}/reserve_table`, `POST ${TOOLS}`],
      );
      assert.equal(await reservation(connection), "Booked T4.");
      await session.close();
    });

    it("keeps running a function whose removal a gateway answered with a 502, and takes a 404 as withdrawn next time", async () => {
      const [session, connection] = await opened(own().register(reserveTable("Held")));
      // The router withdrew it, and the gateway in front of it answered with an error of its own.
      router.serve("DELETE", `${TOOLS}/reserve_table`, (_, calls) =>
        calls === 0 ? { status: 502, text: "Bad Gateway" } : refusal(404, "tool_not_found"),
      );

      await assert.rejects(session.removeTool("reserve_table"), (cause: unknown) =>
        cause instanceof RouterError && cause.status === 502,
      );
      assert.equal(await reservation(connection), "Held T4.");
      await session.removeTool("reserve_table");

      assert.equal(await reservation(connection), undefined);
      await session.close();
    });

    it("withdraws a name whose removal got no answer before adding it again, and runs the new function", async () => {
      const [session, connection] = await opened(own());
      router.serve("POST", TOOLS, { status: 204 });
      await session.addTools(reserving("Reserved"));
      router.serve("DELETE", `${TOOLS}/reserve_table`, { drop: true });
      await assert.rejects(session.removeTool("reserve_table"));
      // The router did withdraw it, so it takes the name again rather than refusing it.
      router.serve("DELETE", `${TOOLS}/reserve_table`, refusal(404, "tool_not_found"));

      await session.addTools(reserving("Booked"));

      assert.deepEqual(
        router.received.filter(({ path }) => path.startsWith(TOOLS)).map(({ method, path }) => `${method} ${path}`),
        [`POST ${TOOLS}`, `DELETE ${TOOLS}/reserve_table`, `DELETE ${TOOLS}/reserve_table`, `POST ${TOOLS}`],
      );
      assert.equal(await reservation(connection), "Booked T4.");
      await session.close();
    });

    it("withdraws a name whose adding got no answer before adding it again", async () => {
      const [session, connection] = await opened(own());
      router.serve("POST", TOOLS, { drop: true });
      await assert.rejects(session.addTools(reserving("Reserved")));
      router.serve("POST", TOOLS, { status: 204 });
      router.serve("DELETE", `${TOOLS}/reserve_table`, refusal(404, "tool_not_found"));

      await session.addTools(reserving("Booked"));

      assert.deepEqual(
        router.received.filter(({ path }) => path.startsWith(TOOLS)).map(({ method, path }) => `${method} ${path}`),
        [`POST ${TOOLS}`, `DELETE ${TOOLS}/reserve_table`, `POST ${TOOLS}`],
      );
      assert.equal(await reservation(connection), "Booked T4.");
      await session.close();
    });

    it("keeps the function a name already runs, which the router never replaces", async () => {
      const [session, connection] = await opened(own());
      router.serve("POST", TOOLS, { status: 204 });
      await session.addTools(reserving("Reserved"));
      router.serve("POST", TOOLS, { drop: true });

      await assert.rejects(session.addTools(reserving("Booked")));

      assert.equal(await reservation(connection), "Reserved T4.");
      await session.close();
    });

    it("withdraws an added function, after which its calls are left alone", async () => {
      const [session, connection] = await opened(own());
      router.serve("POST", TOOLS, { status: 204 });
      router.serve("DELETE", `${TOOLS}/reserve_table`, { status: 204 });
      await session.addTools(reserving("Reserved"));

      await session.removeTool("reserve_table");

      assert.equal(`${router.last.method} ${router.last.path}`, `DELETE ${TOOLS}/reserve_table`);
      assert.equal(await reservation(connection), undefined);
      await session.close();
    });

    it("runs nothing for a function it was opened with once withdrawn, then the one added in its place", async () => {
      const [session, connection] = await opened(own().register(reserveTable("Held")));
      router.serve("DELETE", `${TOOLS}/reserve_table`, { status: 204 });
      router.serve("POST", TOOLS, { status: 204 });

      await session.removeTool("reserve_table");
      assert.equal(await reservation(connection), undefined);
      await session.addTools(reserving("Booked"));

      assert.equal(await reservation(connection), "Booked T4.");
      await session.close();
    });

    it("makes one change at a time, so a refusal never takes back what the next change put in place", async () => {
      const [session, connection] = await opened(own());
      router.serve("POST", TOOLS, (request) =>
        (request.body as { tools: { description: string }[] }).tools[0]?.description === "Reserve a table (Reserved)"
          ? refusal(409, "tool_name_taken")
          : { status: 204 },
      );

      const refused = session.addTools(reserving("Reserved"));
      const added = session.addTools(reserving("Booked"));

      await assert.rejects(refused, (cause: unknown) => cause instanceof RouterError && cause.status === 409);
      await added;
      assert.equal(await reservation(connection), "Booked T4.");
      await session.close();
    });

    it("hands a fork the functions it runs now, since the router copies the tools it offers now", async () => {
      const [session] = await opened(own().register(reserveTable("Held")));
      router.serve("POST", TOOLS, { status: 204 });
      router.serve("DELETE", `${TOOLS}/reserve_table`, { status: 204 });
      router.serve("POST", FORK, forked);
      const cancel = new Tools().register({ name: "cancel_table", description: "Cancel a table", run: () => "Cancelled." });
      await session.addTools(cancel);
      await session.removeTool("reserve_table");

      const forking = session.fork();
      const forkConnection = await router.socket();
      const fork = await forking;

      // lookup, which the fork runs too, answered first would mean cancel_table was left alone.
      forkConnection.send({ type: "tool_call", id: "cancel_1", name: "cancel_table", arguments: "{}" });
      forkConnection.send({ type: "tool_call", id: "lookup_0", name: "lookup", arguments: "{}" });
      const first = await forkConnection.next();
      assert.deepEqual([first["tool_call_id"], first["output"]], ["cancel_1", "Cancelled."]);
      assert.equal((await forkConnection.next())["output"], "found");
      assert.equal(await reservation(forkConnection), undefined);
      await fork.close();
      await session.close();
    });

    it("forks once a change of its tools under way is done, so the fork runs what the router copied", async () => {
      const [session] = await opened(own().register(reserveTable("Held")));
      let answer = (): void => undefined;
      const answering = new Promise<void>((resolve) => {
        answer = resolve;
      });
      router.serve("DELETE", `${TOOLS}/reserve_table`, async () => {
        await answering;
        return { status: 204 };
      });
      router.serve("POST", FORK, forked);
      const removing = session.removeTool("reserve_table");
      await until(() => router.requestsTo("DELETE", `${TOOLS}/reserve_table`).length === 1);

      const forking = session.fork();
      answer();
      await removing;
      const forkConnection = await router.socket();
      const fork = await forking;

      assert.equal(await reservation(forkConnection), undefined);
      await fork.close();
      await session.close();
    });

    it("hands a fork a name whose adding got no answer as unsettled, so the fork withdraws it before adding it", async () => {
      const [session] = await opened(own());
      // The router never added it, so the fork has no tool by the name either.
      router.serve("POST", TOOLS, { drop: true });
      router.serve("POST", FORK, forked);
      router.serve("DELETE", `${FORKED_TOOLS}/reserve_table`, refusal(404, "tool_not_found"));
      router.serve("POST", FORKED_TOOLS, { status: 204 });
      await assert.rejects(session.addTools(reserving("Reserved")));
      const forking = session.fork();
      const forkConnection = await router.socket();
      const fork = await forking;

      await fork.addTools(reserving("Booked"));

      assert.deepEqual(
        router.received
          .filter(({ path }) => path.startsWith(FORKED_TOOLS))
          .map(({ method, path }) => `${method} ${path}`),
        [`DELETE ${FORKED_TOOLS}/reserve_table`, `POST ${FORKED_TOOLS}`],
      );
      assert.equal(await reservation(forkConnection), "Booked T4.");
      await fork.close();
      await session.close();
    });

    it("runs a fork's calls within the cap on calls at once that its functions were given", async () => {
      let release = (): void => undefined;
      const released = new Promise<void>((resolve) => {
        release = resolve;
      });
      let started = 0;
      const capped = new Tools().setConcurrency(1).register({
        name: "lookup",
        description: "Look it up",
        run: async () => {
          started += 1;
          await released;
          return "found";
        },
      });
      const [session, connection] = await opened(capped);
      router.serve("POST", TOOLS, { status: 204 });
      router.serve("POST", FORK, forked);
      await session.addTools(reserving("Reserved"));
      const forking = session.fork();
      const forkConnection = await router.socket();
      const fork = await forking;

      // The session's call takes the one place. The fork's waits for it, and a call of the
      // function added in a `Tools` of its own does not: frames are read in order, so once that
      // one is answered the fork has read the call waiting.
      connection.send({ type: "tool_call", id: "lookup_1", name: "lookup", arguments: "{}" });
      await until(() => started === 1);
      forkConnection.send({ type: "tool_call", id: "lookup_2", name: "lookup", arguments: "{}" });
      forkConnection.send({ type: "tool_call", id: "reserve_1", name: "reserve_table", arguments: '{"table":"T4"}' });
      assert.equal((await forkConnection.next())["output"], "Reserved T4.");
      assert.equal(started, 1);

      release();
      assert.equal((await forkConnection.next())["output"], "found");
      assert.equal((await connection.next())["output"], "found");
      await fork.close();
      await session.close();
    });

    it("keeps a function added to one session to that session, not every session of its agent", async () => {
      const agentTools = own();
      const [first, firstConnection] = await opened(agentTools);
      const [second, secondConnection] = await opened(agentTools);
      router.serve("POST", TOOLS, { status: 204 });

      await first.addTools(reserving("Reserved"));

      assert.equal(await reservation(secondConnection), undefined);
      assert.equal(await reservation(firstConnection), "Reserved T4.");
      await first.close();
      await second.close();
    });

    it("adds nothing to a conversation nothing here watches, since nothing would run the calls", async () => {
      const session = await Session.open(api, {}, { watch: false });

      await assert.rejects(session.addTools(reserving("Reserved")), ConfigurationError);

      assert.deepEqual(router.requestsTo("POST", TOOLS), []);
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

  it("stops the reply from the stop button of the channel it opened, until it is closed", async () => {
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
    router.serve("POST", QUERY_PATH, channelPage([]));
    router.serve("POST", EVENT_PATH, { status: 201, body: { event: { type: "ai_indicator.stop" }, duration: "1ms" } });
    api.use({ chat: (await chatClient(router.url)) as never });
    const [session, connection] = await opened();
    const { channel } = await session.chat();
    await watch(channel);
    // What the default stop button of stream-chat-react's composer calls.
    const stopButton = () => (channel as { stopAIResponse(): Promise<void> }).stopAIResponse();
    const stopEvents = () =>
      router.requestsTo("POST", EVENT_PATH).map((one) => (one.body as { event?: { type?: string } }).event?.type);

    await stopButton();

    assert.deepEqual(await connection.next(), { type: "interrupt" });
    assert.deepEqual(stopEvents(), ["ai_indicator.stop"], "the views that follow the event still get it");

    await session.close();
    assert.deepEqual(await connection.next(), { type: "close" });
    // A closed session's socket refuses an interrupt, so the button would reject if still attached.
    await stopButton();
    assert.deepEqual(stopEvents(), ["ai_indicator.stop", "ai_indicator.stop"]);
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

  // stream-chat 10 names the connected user `userId`, and keeps `userID` only as a deprecated
  // alias of it.
  it("refuses a chat client connected as somebody else when it names the user as userId", async () => {
    const page = new Client({ url: router.url, customerId: "local", apiKey: "vak_live_x" });
    await page.setUser({ id: "john" }, "token-for-john");
    page.use({
      chat: {
        userId: "jane",
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
      assert.match(error.message, /import "@stream-io\/stream-agents\/chat"/);
      return true;
    });
    await session.close();
  });

  it("tries chat again after it failed, rather than failing the same way for good", async () => {
    const page = new Client({ url: router.url, customerId: "local", apiKey: "vak_live_x" });
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

    // Asked before the page knows who is looking at it.
    await assert.rejects(session.chat(), ConfigurationError);
    const client = {
      channel: (type: string, id: string) => ({ type, id }),
      connectUser: async () => undefined,
      disconnectUser: async () => undefined,
    };
    page.use({ chat: client });
    const chat = await session.chat();

    assert.equal(chat.client, client);
    await session.close();
  });

  it("tries the call again after opening it failed, so a failed join can be retried", async () => {
    const page = new Client({ url: router.url, customerId: "local", apiKey: "vak_live_x" });
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
    const opening = Session.open(page, { start_voice: true });
    await router.socket();
    const session = await opening;

    await assert.rejects(session.video(), ConfigurationError);
    const client = {
      streamClient: {},
      call: (type: string, id: string) => ({ type, id }),
      disconnectUser: async () => undefined,
    };
    page.use({ video: client });
    const video = await session.video();

    assert.deepEqual(video.call, { type: "agent", id: "sess_1" });
    await session.close();
  });

  it("takes a typed question back out of the transcript when it could not be asked", async () => {
    const [session] = await opened();
    router.serve("POST", "/v1/agents/sessions/sess_1/responses", {
      status: 400,
      body: { error: { message: "text is required", type: "invalid_request", code: "validation_failed" } },
    });

    await assert.rejects(session.responses.create("Plan a weekend in Lisbon"));

    assert.deepEqual(session.conversation.get().turns, [], "a question that never reached the agent");
    await session.close();
  });

  it("asks again under the same request id, so a retry is answered once", async () => {
    const [session] = await opened();
    router.serve("POST", "/v1/agents/sessions/sess_1/responses", (_, calls) =>
      calls === 0
        ? { status: 503 }
        : {
            status: 202,
            body: { id: "response-1", session_id: "sess_1", status: "running", created_at: new Date().toISOString() },
          },
    );

    const answered = await session.responses.create("Plan a weekend in Lisbon", { requestId: "req_7" });

    assert.equal(answered.id, "response-1");
    const sent = router.requestsTo("POST", "/v1/agents/sessions/sess_1/responses");
    assert.deepEqual(
      sent.map((one) => (one.body as Record<string, unknown>)["request_id"]),
      ["req_7", "req_7"],
    );
    await session.close();
  });

  it("does not tell a page about a delta that changed nothing", async () => {
    const [session, connection] = await opened();
    connection.send({ type: "responding", turn_id: "t1" });
    connection.send({ type: "response_delta", turn_id: "t1", text: "Hi" });
    for await (const event of session.events()) {
      if (event.kind === "response_delta") {
        break;
      }
    }
    let told = 0;
    session.conversation.subscribe(() => told++);

    connection.send({ type: "response_delta", turn_id: "t1", text: "" });
    connection.send({ type: "responded", turn_id: "t1", text: "Hi" });
    connection.send({ type: "response_delta", turn_id: "t1", text: "!" });
    for await (const event of session.events()) {
      if (event.kind === "response_delta" && event.text === "!") {
        break;
      }
    }

    assert.equal(told, 2, "the turn ending, and the delta that added something");
    assert.equal(session.conversation.get().turns.at(-1)?.text, "Hi!");
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

describe("Regenerate, edit and branches", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
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
  });

  afterEach(async () => {
    await router.stop();
  });

  async function opened(): Promise<Session> {
    const opening = Session.open(api, {});
    await router.socket();
    return opening;
  }

  /** A top-level Chat message, written `second` seconds into 2026. */
  const message = (id: string, second: number) => chatMessage(id, second);

  /**
   * The session with its channel open and holding `messages`, read by a real stream-chat client
   * (9, or 10 in the stream-chat 10 pass of CI's sdks job) from the test router: what is under test
   * is what its channel state holds, and that 9's query merges rather than replaces.
   */
  async function loaded(messages: Record<string, unknown>[]) {
    const chat = await chatClient(router.url);
    api.use({ chat: chat as never });
    const session = await opened();
    const { channel } = await session.chat();
    router.serve("POST", QUERY_PATH, channelPage(messages));
    await watch(channel);
    const ids = () => heldIds(channel);
    return { chat, session, channel, ids };
  }

  it("regenerates and edits as a question that replaces an earlier response, under a request id", async () => {
    const session = await opened();
    router.serve("POST", "/v1/agents/sessions/sess_1/responses", {
      status: 202,
      body: {
        id: "resp_2",
        session_id: "sess_1",
        status: "running",
        created_at: "2026-01-01T00:00:00Z",
        parent_response_id: "resp_0",
        branch: { of: "reply", index: 2, count: 2 },
      },
    });

    const regenerated = await session.responses.regenerate("resp_1", { maxOutputTokens: 300 });
    await session.responses.edit({ response_id: "resp_1" } as never, "Rome instead", { requestId: "req_9" });

    assert.deepEqual(regenerated.created.branch, { of: "reply", index: 2, count: 2 });
    assert.equal(regenerated.created.parent_response_id, "resp_0");
    const [regenerate, edit] = router
      .requestsTo("POST", "/v1/agents/sessions/sess_1/responses")
      .map((one) => one.body as Record<string, unknown>);
    assert.equal(regenerate?.["text"], "", "no text is what makes it a regenerate");
    assert.equal(regenerate?.["replaces"], "resp_1");
    assert.equal(regenerate?.["max_output_tokens"], 300);
    assert.match(String(regenerate?.["request_id"]), /^[0-9a-f-]{36}$/, "the router requires one");
    assert.deepEqual(edit, { text: "Rome instead", replaces: "resp_1", request_id: "req_9" });
    await session.close();
  });

  it("refuses an edit with no text and a response with no id before sending anything", async () => {
    const session = await opened();

    await assert.rejects(session.responses.edit("resp_1", ""), ConfigurationError);
    await assert.rejects(session.responses.regenerate(""), ConfigurationError);
    await assert.rejects(session.responses.siblings(""), ConfigurationError);

    assert.deepEqual(router.requestsTo("POST", "/v1/agents/sessions/sess_1/responses"), []);
    await session.close();
  });

  it("lists siblings, reads a branch and activates it on the response's own paths", async () => {
    const session = await opened();
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_1/siblings", (_, calls) => ({
      body:
        calls === 0
          ? {
              items: [{ id: "resp_1", created_at: "2026-01-01T00:00:00Z", active: false, index: 1, said: "Lisbon?" }],
              has_more: true,
              next_cursor: "c1",
            }
          : { items: [{ id: "resp_2", created_at: "2026-01-01T00:00:01Z", active: true, index: 2 }], has_more: false },
    }));
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_1/branch", {
      body: {
        active: false,
        items: [
          { id: "m3", text: "Lisbon?", attachments: [], custom: {}, created_at: "2026-01-01T00:00:00Z", user_id: "john", role: "user" },
        ],
        has_more: false,
      },
    });
    router.serve("POST", "/v1/agents/sessions/sess_1/responses/resp_1/activate", {
      body: { leaf_response_id: "resp_4", already_active: false, resent: 4, rewound_to: { message_id: "m1", inclusive: false } },
    });

    const siblings = [];
    for await (const sibling of session.responses.unwindSiblings("resp_1", { of: "question", limit: 1 })) {
      siblings.push(sibling.id);
    }
    const branch = await session.responses.branch("resp_1", { limit: 50 });
    const activated = await session.responses.activate("resp_1");

    assert.deepEqual(siblings, ["resp_1", "resp_2"]);
    const asked = router.requestsTo("GET", "/v1/agents/sessions/sess_1/responses/resp_1/siblings");
    assert.deepEqual(
      asked.map((one) => Object.fromEntries(one.query)),
      [
        { of: "question", limit: "1" },
        { of: "question", limit: "1", cursor: "c1" },
      ],
    );
    assert.equal(branch.items[0]?.role, "user");
    assert.equal(router.requestsTo("GET", "/v1/agents/sessions/sess_1/responses/resp_1/branch")[0]?.query.get("limit"), "50");
    assert.deepEqual(activated, {
      leaf_response_id: "resp_4",
      already_active: false,
      resent: 4,
      rewound_to: { message_id: "m1", inclusive: false },
    });
    await session.close();
  });

  it("surfaces the branch refusals as router errors a page can tell apart", async () => {
    const session = await opened();
    const refusal = (status: number, code: string, type: string) => ({
      status,
      body: { error: { message: `refused: ${code}`, type, code } },
    });
    router.serve("POST", "/v1/agents/sessions/sess_1/responses", refusal(409, "branch_limit_reached", "conflict"));
    router.serve("POST", "/v1/agents/sessions/sess_1/responses/resp_1/activate", refusal(409, "branch_too_long", "conflict"));
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_1/branch", refusal(410, "branch_unavailable", "gone"));

    const failures = await Promise.all([
      session.responses.regenerate("resp_1").catch((error: unknown) => error),
      session.responses.activate("resp_1").catch((error: unknown) => error),
      session.responses.branch("resp_1").catch((error: unknown) => error),
    ]);

    assert.deepEqual(failures.map(branchErrorOf), ["branch_limit_reached", "branch_too_long", "branch_unavailable"]);
    assert.deepEqual(
      failures.map((error) => (error instanceof RouterError ? error.status : 0)),
      [409, 409, 410],
    );
    assert.equal(branchErrorOf(new RouterError(409, "POST /x", "taken", { code: "conflict" })), undefined);
    await session.close();
  });

  it("drops what a rewind hid from the channel at once, then reads the channel again", async () => {
    const { chat, session, ids } = await loaded([message("m1", 1), message("m2", 2), message("m3", 3), message("m4", 4)]);
    let heldWhenRead: string[] | undefined;
    router.serve("POST", QUERY_PATH, () => {
      heldWhenRead = ids();
      // Read just after the rewind, Chat can still return what it hid, beside what came since.
      return channelPage([message("m1", 1), message("m2", 2), message("m3", 3), message("m4", 4), message("m5", 6)]);
    });

    rewind(chat, "m1", false, 5);

    assert.deepEqual(ids(), ["m1"], "dropped before anything was read");
    await until(() => ids().includes("m5"));
    assert.deepEqual(heldWhenRead, ["m1"]);
    assert.deepEqual(ids(), ["m1", "m5"], "a stale read does not bring hidden messages back");
    await session.close();
  });

  it("drops the rewound-to message too when the rewind was inclusive, as an edit's is", async () => {
    const { chat, session, ids } = await loaded([message("m1", 1), message("m2", 2), message("m3", 3), message("m4", 4)]);
    router.serve("POST", QUERY_PATH, channelPage([message("m1", 1), message("m2", 2)]));

    chat.dispatchEvent({
      type: "notification.channel_rewound",
      cid: "agent:support-1",
      rewind: { id: "rw_1", message_id: "m3", inclusive: true, created_at: new Date(Date.UTC(2026, 0, 1, 0, 0, 5)).toISOString() },
    });

    assert.deepEqual(ids(), ["m1", "m2"]);
    await until(() => router.requestsTo("POST", QUERY_PATH).length === 2);
    await session.close();
  });

  it("prunes what a regenerate hid as soon as it answers, keeping what arrived while it was asked", async () => {
    const { chat, session, ids } = await loaded([message("m1", 1), message("m2", 2), message("m3", 3), message("m4", 4)]);
    router.serve("POST", "/v1/agents/sessions/sess_1/responses", () => {
      // The new answer's message.new can beat the HTTP answer.
      arrive(chat, message("m5", 6));
      return {
        status: 202,
        body: {
          id: "resp_2",
          session_id: "sess_1",
          status: "running",
          created_at: "2026-01-01T00:00:06Z",
          rewound_to: { message_id: "m1", inclusive: false },
        },
      };
    });

    await session.responses.regenerate("resp_1");

    assert.deepEqual(ids(), ["m1", "m5"], "never the old answer beside the new one");
    assert.equal(router.requestsTo("POST", QUERY_PATH).length, 1, "the event reads it again, not this");
    await session.close();
  });

  it("keeps what it holds when an edit's rewind arrives after the edit's answer already dropped the question", async () => {
    const { chat, session, ids } = await loaded([message("m1", 1), message("m2", 2), message("m3", 3), message("m4", 4)]);
    router.serve("POST", "/v1/agents/sessions/sess_1/responses", () => {
      // The new question can arrive before the answer, and before the rewind event.
      arrive(chat, message("q2", 6));
      return {
        status: 202,
        body: {
          id: "resp_2",
          session_id: "sess_1",
          status: "running",
          created_at: "2026-01-01T00:00:06Z",
          rewound_to: { message_id: "m3", inclusive: true },
        },
      };
    });
    await session.responses.edit("resp_1", "Rome instead");
    assert.deepEqual(ids(), ["m1", "m2", "q2"], "the edited question went with the answer");
    let heldWhenRead: string[] | undefined;
    router.serve("POST", QUERY_PATH, () => {
      heldWhenRead = ids();
      return channelPage([message("m1", 1), message("m2", 2), message("m3", 3), message("q2", 6)]);
    });

    // The edit's own rewind, to the question this device no longer holds.
    rewind(chat, "m3", true, 5);

    assert.deepEqual(ids(), ["m1", "m2", "q2"], "nothing more dropped: where the rewind starts is known here");
    await until(() => router.requestsTo("POST", QUERY_PATH).length === 2);
    await until(() => heldWhenRead !== undefined && ids().length === 3);
    assert.deepEqual(heldWhenRead, ["m1", "m2", "q2"]);
    assert.deepEqual(ids(), ["m1", "m2", "q2"], "a stale read does not bring the edited question back");
    await session.close();
  });

  it("stops following the channel's rewinds once the session is closed", async () => {
    const { chat, session, ids } = await loaded([message("m1", 1), message("m2", 2)]);

    await session.close();
    rewind(chat, "m1", false, 5);

    assert.deepEqual(ids(), ["m1", "m2"]);
    assert.equal(router.requestsTo("POST", QUERY_PATH).length, 1);
  });

  it("leaves another channel's rewind alone", async () => {
    const { chat, session, ids } = await loaded([message("m1", 1), message("m2", 2)]);

    rewind(chat, "m1", false, 5, "agent:somebody-else");

    assert.deepEqual(ids(), ["m1", "m2"]);
    await session.close();
  });
});
