import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { createElement } from "react";
import { renderToString } from "react-dom/server";

import { Client, Session, ToolCalls, toolCallTitle, type SessionEvent } from "../src/index.js";
import { useToolCall, useToolCalls, useTurnToolCalls } from "../src/react.js";
import { TestRouter, type Connection } from "./router.js";

const started = {
  type: "tool_call.started",
  tool_call_id: "call-1",
  tool: "book_flight",
  turn_id: "turn-1",
  display_title: "Booking your flight",
  status: "waiting",
  started_at: "2026-11-18T12:00:00Z",
  duration_ms: 0,
};

describe("ToolCalls", () => {
  it("keeps the last state of each call, in the order they started", () => {
    const calls = new ToolCalls();
    calls.observe(started);
    calls.observe({ ...started, tool_call_id: "call-2", tool: "search_flights", status: "in_progress" });
    calls.observe({
      ...started,
      type: "tool_call.finished",
      status: "success",
      duration_ms: 1200,
      approved_at: "2026-11-18T12:00:00.400Z",
      cursor: "s.4",
    });

    assert.deepEqual([...calls.get().keys()], ["call-1", "call-2"]);
    assert.deepEqual(calls.get().get("call-1"), {
      type: "tool_call.finished",
      tool_call_id: "call-1",
      tool: "book_flight",
      turn_id: "turn-1",
      display_title: "Booking your flight",
      status: "success",
      started_at: "2026-11-18T12:00:00Z",
      duration_ms: 1200,
      approved_at: "2026-11-18T12:00:00.400Z",
    });
    assert.equal(calls.get().get("call-2")?.status, "in_progress");
  });

  it("gives one turn's calls in the order they started, the same array until one of them changes", () => {
    const calls = new ToolCalls();
    calls.observe({ ...started, tool_call_id: "a" });
    calls.observe({ ...started, tool_call_id: "other", turn_id: "turn-2" });
    calls.observe({ ...started, tool_call_id: "b" });

    const first = calls.forTurn("turn-1");
    assert.deepEqual(
      first.map((call) => call.tool_call_id),
      ["a", "b"],
    );
    assert.deepEqual(calls.forTurn("turn-3"), []);

    calls.observe({ ...started, tool_call_id: "other", turn_id: "turn-2", status: "success" });
    assert.equal(calls.forTurn("turn-1"), first, "another turn's call changed, so the same array");

    calls.observe({ ...started, tool_call_id: "a", status: "success" });
    const changed = calls.forTurn("turn-1");
    assert.notEqual(changed, first);
    assert.deepEqual(
      changed.map((call) => [call.tool_call_id, call.status]),
      [
        ["a", "success"],
        ["b", "waiting"],
      ],
    );
  });

  it("titles a call by its tool's display title, or its name without one", () => {
    const calls = new ToolCalls();
    calls.observe(started);
    calls.observe({ ...started, tool_call_id: "call-2", tool: "list_issues", display_title: "" });

    assert.equal(toolCallTitle(calls.get().get("call-1")!), "Booking your flight");
    assert.equal(toolCallTitle(calls.get().get("call-2")!), "list_issues");
  });

  it("titles a running or waiting call by its last update, and a finished one by its display title", () => {
    const titled = (frame: Record<string, unknown>) => {
      const calls = new ToolCalls();
      calls.observe({ ...started, ...frame });
      return toolCallTitle(calls.get().get("call-1")!);
    };
    const update = { type: "tool_call.progress", status: "in_progress", progress: "Comparing 40 flights" };

    assert.equal(titled(update), "Comparing 40 flights");
    assert.equal(titled({ progress: "Book the 9:40 to Lisbon?" }), "Book the 9:40 to Lisbon?");
    assert.equal(titled({ ...update, type: "tool_call.finished", status: "success" }), "Booking your flight");
    assert.equal(titled({ ...update, progress: "" }), "Booking your flight");
  });

  it("keeps a call's custom data, and none for a frame without it", () => {
    const calls = new ToolCalls();
    calls.observe({ ...started, status: "success", type: "tool_call.finished", custom: { cheapest: 129 } });
    calls.observe({ ...started, tool_call_id: "call-2", custom: ["not", "an", "object"] });

    assert.deepEqual(calls.get().get("call-1")?.custom, { cheapest: 129 });
    assert.equal(calls.get().get("call-2")?.custom, undefined);
  });

  it("leaves the map alone for frames about no call's state", () => {
    const calls = new ToolCalls();
    calls.observe(started);
    const before = calls.get();

    calls.observe({ type: "tool_started", tool_call_id: "call-9", tool: "x" });
    calls.observe({ type: "tool_call.progress", status: "in_progress" });

    assert.equal(calls.get(), before, "the same map, not an equal one");
  });
});

describe("session.toolCalls and session.on", () => {
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

  /** Reads events until one of kind arrives. */
  async function until(session: Session, kind: string): Promise<void> {
    for await (const event of session.events()) {
      if (event.kind === kind) {
        return;
      }
    }
  }

  it("keeps each call's latest state, and a handler hears its kind until it is turned off", async () => {
    const [session, connection] = await opened();
    const heard: string[] = [];
    let finished = 0;
    const off = session.on("tool_call.progress", (event: SessionEvent) => {
      heard.push(String(event.frame["progress"]));
      off();
    });
    session.on("tool_call.finished", () => finished++);

    connection.send({ ...started, status: "in_progress" });
    connection.send({ ...started, type: "tool_call.progress", status: "in_progress", progress: "Searching for flights" });
    connection.send({ ...started, type: "tool_call.progress", status: "in_progress", progress: "Comparing prices" });
    connection.send({ ...started, type: "tool_call.finished", status: "failed", error: "the airline is down" });
    await until(session, "tool_call.finished");

    assert.deepEqual(heard, ["Searching for flights"]);
    assert.equal(finished, 1);
    assert.equal(session.toolCall("call-1")?.status, "failed");
    assert.equal(session.toolCall("call-1")?.error, "the airline is down");
    assert.equal(session.toolCall("call-1")?.progress, undefined, "the finished frame carries the whole state");
    await session.close();
  });

  it("renders the calls through the React hooks", async () => {
    const [session, connection] = await opened();
    connection.send({ ...started, type: "tool_call.progress", status: "in_progress", progress: "Booking" });
    await until(session, "tool_call.progress");

    connection.send({ ...started, tool_call_id: "call-2", turn_id: "turn-2", status: "in_progress" });
    await until(session, "tool_call.started");

    function Calls() {
      const calls = useToolCalls(session);
      const call = useToolCall(session, "call-1");
      const turn = useTurnToolCalls(session, "turn-1");
      return createElement(
        "p",
        null,
        `${calls.size} ${call?.status} ${call?.progress} ${turn.map(toolCallTitle).join(",")}`,
      );
    }

    assert.equal(renderToString(createElement(Calls)), "<p>2 in_progress Booking Booking</p>");
    await session.close();
  });
});
