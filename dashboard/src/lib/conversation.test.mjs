import assert from "node:assert/strict";
import { test } from "node:test";
import {
  active,
  activityLabel,
  elapsed,
  localRouter,
  mergeMessages,
} from "./conversation.ts";

const start = "2026-09-10T10:00:00Z";
const finished = "2026-09-10T10:00:05Z";
const message = {
  id: "assistant",
  role: "assistant",
  text: "",
  state: "thinking",
  response_started_at: start,
  state_started_at: start,
  saved: false,
};

test("all backend activity states have live timers and readable labels", () => {
  for (const state of ["thinking", "queued", "tools", "writing"])
    assert.equal(active(state), true);
  for (const state of ["completed", "failed", "cancelled", "interrupted"])
    assert.equal(active(state), false);
  assert.equal(activityLabel("queued"), "Waiting for research");
  assert.equal(activityLabel("tools"), "Running tools");
  assert.equal(activityLabel("writing"), "Writing answer");
});
test("timestamps reconstruct timers; completion freezes duration and execution excludes queueing", () => {
  assert.equal(
    elapsed(start, undefined, undefined, Date.parse(start) + 3200),
    "3.2s",
  );
  assert.equal(
    elapsed(start, finished, 5000, Date.parse(start) + 999999),
    "5.0s",
  );
  assert.equal(
    elapsed("2026-09-10T10:00:02Z", finished, undefined, Date.now()),
    "3.0s",
  );
  assert.equal(
    elapsed(start, undefined, undefined, Date.parse(start) - 100),
    "0.0s",
  );
  assert.equal(elapsed("", undefined, undefined, Date.now()), "—");
});
test("snapshot merges retain one response across tools and place the question first", () => {
  const user = { ...message, id: "question", role: "user", state: "completed" };
  const tool = {
    type: "tool_calling",
    tool_call_id: "tool-one",
    name: "search_docs",
    title: "Search documentation",
    status: "running",
    phase: "running",
    started_at: start,
  };
  const running = { ...message, state: "tools", attachments: [tool] };
  const completed = {
    ...running,
    state: "completed",
    text: "Answer",
    saved: true,
    finished_at: finished,
    attachments: [
      {
        ...tool,
        status: "completed",
        finished_at: finished,
        duration_ms: 5000,
      },
    ],
  };
  const merged = mergeMessages(mergeMessages([message, user], [running]), [
    completed,
  ]);
  assert.deepEqual(
    merged.map((message) => message.id),
    ["question", "assistant"],
  );
  assert.equal(merged[1].attachments?.[0].started_at, start);
  assert.equal(merged[1].attachments?.[0].duration_ms, 5000);
  assert.equal(merged[1].text, "Answer");
  assert.equal(merged[1].saved, true);
});
test("local backend override does not redirect conversation data to remote hosts", () => {
  assert.equal(
    localRouter(undefined, "https://configured.example"),
    "https://configured.example",
  );
  assert.equal(
    localRouter("http://127.0.0.1:8098/", ""),
    "http://127.0.0.1:8098",
  );
  for (const value of [
    "https://other.example",
    "http://user:pass@localhost",
    "http://localhost/private",
    "javascript:alert(1)",
  ])
    assert.throws(() => localRouter(value, ""));
});
