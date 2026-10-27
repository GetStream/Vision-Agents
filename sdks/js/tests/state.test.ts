import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Client, ConversationState, Session, type ConversationSnapshot } from "../src/index.js";
import { TestRouter, type Connection } from "./router.js";

describe("ConversationState", () => {
  /** The states one frame after another leaves, starting idle. */
  function folded(...frames: Record<string, unknown>[]): ConversationSnapshot[] {
    const state = new ConversationState();
    return frames.map((frame) => {
      state.observe(frame);
      return state.get();
    });
  }

  it("starts idle", () => {
    assert.deepEqual(new ConversationState().get(), { kind: "idle", skills: [] });
  });

  it("follows somebody talking, the model answering, and the answer ending", () => {
    const kinds = folded(
      { type: "hearing" },
      { type: "heard" },
      { type: "responding" },
      { type: "response_delta" },
      { type: "responded" },
    ).map((each) => each.kind);
    // `heard` changes nothing: what was said is in, and the model has not started on it yet.
    assert.deepEqual(kinds, ["listening", "listening", "responding", "responding", "idle"]);
  });

  it("goes idle when the reply is cut off", () => {
    assert.equal(folded({ type: "responding" }, { type: "interrupted" })[1]?.kind, "idle");
  });

  it("names the skills working, and answers once the last is done", () => {
    const states = folded(
      { type: "delegated", skill: "vision" },
      { type: "delegated", skill: "search" },
      { type: "task_settled", skill: "vision" },
      { type: "task_cancelled", skill: "search" },
    );
    assert.deepEqual(states, [
      { kind: "working", skills: ["vision"] },
      { kind: "working", skills: ["vision", "search"] },
      { kind: "working", skills: ["search"] },
      { kind: "responding", skills: [] },
    ]);
  });

  it("ends when the agent leaves, and stays ended", () => {
    const kinds = folded({ type: "left" }, { type: "hearing" }).map((each) => each.kind);
    assert.deepEqual(kinds, ["ended", "ended"]);
  });

  it("leaves the state alone for frames with no bearing on it", () => {
    const states = folded({ type: "hearing" }, { type: "spoke" }, { type: "turn" }, { type: "something_new" });
    assert.equal(states[0], states[3], "the same snapshot, not an equal one");
  });

  it("tells listeners once per change, and hands back bound functions", () => {
    const state = new ConversationState();
    const { get, subscribe } = state;
    let told = 0;
    const stop = subscribe(() => told++);

    state.observe({ type: "hearing" });
    state.observe({ type: "hearing" });
    state.observe({ type: "responding" });
    assert.equal(told, 2);
    assert.equal(get().kind, "responding");

    stop();
    state.observe({ type: "responded" });
    assert.equal(told, 2);
  });
});

describe("session.state", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
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
  });

  afterEach(async () => {
    await router.stop();
  });

  async function opened(): Promise<[Session, Connection]> {
    const opening = Session.open(api, { start_voice: true });
    const connection = await router.socket();
    return [await opening, connection];
  }

  it("follows the socket without taking anything from events()", async () => {
    const [session, connection] = await opened();
    const seen: string[] = [];
    session.state.subscribe(() => seen.push(session.state.get().kind));

    const frames = [{ type: "hearing" }, { type: "responding" }, { type: "responded" }];
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
    assert.deepEqual(read, ["hearing", "responding", "responded"], "every event is still the caller's");
    assert.deepEqual(seen, ["listening", "responding", "idle"]);
    await session.close();
  });

  it("ends when the session is closed", async () => {
    const [session] = await opened();
    await session.close();
    assert.equal(session.state.get().kind, "ended");
  });

  it("ends when the socket drops", async () => {
    const [session, connection] = await opened();
    connection.socket.close();
    await session.wait();
    assert.equal(session.state.get().kind, "ended");
  });
});
