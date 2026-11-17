import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { CameraFrames, Client, NoFrames, Session, Tools } from "../src/index.js";
import { TestRouter, type Connection } from "./router.js";

describe("CameraFrames", () => {
  const at = 1_800_000_000_000;

  it("picks the newest frames taken by the moment asked about, oldest first", () => {
    const frames = new CameraFrames();
    frames.record("a", at - 1_000);
    frames.record("b", at - 500);
    frames.record("c", at);

    assert.deepEqual(
      frames.select(at, 2).map((each) => each.jpeg),
      ["b", "c"],
    );
  });

  it("never hands a frame from after the question, past the clock skew it allows", () => {
    const frames = new CameraFrames();
    frames.record("before", at - 100);
    frames.record("within skew", at + CameraFrames.skew);
    frames.record("after", at + CameraFrames.skew + 1);

    assert.deepEqual(
      frames.select(at, 8).map((each) => each.jpeg),
      ["before", "within skew"],
    );
  });

  it("says nothing recent enough was shown, rather than handing an old frame", () => {
    const frames = new CameraFrames();
    frames.record("stale", at - CameraFrames.freshFor - 1);
    assert.throws(() => frames.select(at, 1), NoFrames);
    assert.throws(() => new CameraFrames().select(at, 1), NoFrames);
  });

  it("keeps the newest eight, and hands at least one and at most eight", () => {
    const frames = new CameraFrames();
    for (let index = 1; index <= 10; index++) {
      frames.record(`f${index}`, at - 1_000 + index * 10);
    }
    assert.deepEqual(
      frames.select(at, 100).map((each) => each.id),
      [3, 4, 5, 6, 7, 8, 9, 10],
    );
    assert.equal(frames.select(at, 0).length, 1);
    assert.equal(frames.select(at, Number.NaN).length, 1);
  });

  it("forgets every frame when cleared, and counts on", () => {
    const frames = new CameraFrames();
    frames.record("a", at);
    frames.clear();
    assert.throws(() => frames.select(at, 1), NoFrames);
    frames.record("b", at);
    assert.equal(frames.select(at, 1)[0]?.id, 2);
  });
});

describe("get_video_frames", () => {
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

  async function opened(tools?: Tools): Promise<[Session, Connection]> {
    const opening = Session.open(api, { start_voice: true }, tools ? { tools } : {});
    const connection = await router.socket();
    return [await opening, connection];
  }

  const asked = (atMs: number, limit: number) => ({
    type: "tool_call",
    id: "call_1",
    name: "get_video_frames",
    arguments: JSON.stringify({ at_ms: atMs, limit }),
    request_id: "task_1",
    turn_id: "turn_1",
  });

  it("answers with the frames the camera showed, and what they are", async () => {
    const [session, connection] = await opened();
    const now = Date.now();
    session.cameraFrames = new CameraFrames();
    session.cameraFrames.record("AAAA", now - 600);
    session.cameraFrames.record("BBBB", now - 100);

    connection.send(asked(now, 2));

    assert.deepEqual(await connection.next(), {
      type: "tool_result",
      tool_call_id: "call_1",
      request_id: "task_1",
      turn_id: "turn_1",
      // An array of parts, which is the only way the router takes pictures in a tool's output.
      output: [
        {
          type: "text",
          text: JSON.stringify({
            clock: "device_unix_ms",
            frames: [
              { captured_at_ms: now - 600, frame_id: "camera-1" },
              { captured_at_ms: now - 100, frame_id: "camera-2" },
            ],
            source: "camera",
          }),
        },
        { type: "image_url", image_url: { url: "data:image/jpeg;base64,AAAA" } },
        { type: "image_url", image_url: { url: "data:image/jpeg;base64,BBBB" } },
      ],
    });
    await session.close();
  });

  it("says nothing recent was shown as the answer, not as an error", async () => {
    const [session, connection] = await opened();
    session.cameraFrames = new CameraFrames();

    connection.send(asked(Date.now(), 1));

    const answer = await connection.next();
    assert.equal(answer["output"], new NoFrames().message);
    assert.equal(answer["error"], undefined);
    await session.close();
  });

  it("leaves the request to the watcher the router asked, camera on or not", async () => {
    const tools = new Tools().register({
      name: "ping",
      description: "Answers pong",
      run: () => "pong",
    });
    const [session, connection] = await opened(tools);
    session.cameraFrames = new CameraFrames();
    session.cameraFrames.record("AAAA", Date.now());

    connection.send({ ...asked(Date.now(), 1), run: false });
    connection.send({ type: "tool_call", id: "call_2", name: "ping", arguments: "{}" });

    assert.equal((await connection.next())["tool_call_id"], "call_2");
    await session.close();
  });

  it("leaves the request to whoever owns it when this device has no camera on", async () => {
    const tools = new Tools().register({
      name: "ping",
      description: "Answers pong",
      run: () => "pong",
    });
    const [session, connection] = await opened(tools);

    connection.send(asked(Date.now(), 1));
    connection.send({ type: "tool_call", id: "call_2", name: "ping", arguments: "{}" });

    // The first thing sent up answers the second call: the first got no reply at all.
    const answer = await connection.next();
    assert.equal(answer["tool_call_id"], "call_2");
    await session.close();
  });
});
