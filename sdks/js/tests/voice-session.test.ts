import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Client, ConfigurationError, Session } from "../src/index.js";
import { VoiceSession, type VoiceStatus } from "../src/video.js";
import { inPage } from "./device.js";
import { TestRouter, type Connection } from "./router.js";

/** An observable with a current value, the shape the Video SDK's call state has. */
class Subject<T> {
  private readonly listeners = new Set<(value: T) => void>();
  constructor(private value: T) {}
  subscribe(listener: (value: T) => void): { unsubscribe(): void } {
    this.listeners.add(listener);
    listener(this.value);
    return { unsubscribe: () => this.listeners.delete(listener) };
  }
  next(value: T): void {
    this.value = value;
    this.listeners.forEach((listener) => listener(value));
  }
  get observed(): number {
    return this.listeners.size;
  }
}

/** A Stream call that records what was done to it. */
function fakeCall(type: string, id: string, log: string[], fail: { join?: Error } = {}) {
  return {
    type,
    id,
    camera: {
      selectDirection: async (direction: string) => void log.push(`camera.${direction}`),
      enable: async () => void log.push("camera.on"),
      disable: async () => void log.push("camera.off"),
      state: { mediaStream$: new Subject<unknown>(undefined) },
    },
    microphone: {
      enable: async () => void log.push("mic.on"),
      disable: async () => void log.push("mic.off"),
    },
    join: async (options: unknown) => {
      log.push(`join ${JSON.stringify(options)}`);
      if (fail.join) {
        throw fail.join;
      }
    },
    leave: async () => void log.push("leave"),
    state: {
      remoteParticipants$: new Subject<{ isSpeaking: boolean }[]>([]),
      localParticipant$: new Subject<{ isSpeaking: boolean } | undefined>(undefined),
    },
  };
}

type FakeCall = ReturnType<typeof fakeCall>;

describe("VoiceSession", () => {
  let router: TestRouter;
  let api: Client;
  let log: string[];
  let calls: FakeCall[];
  let failJoin: Error | undefined;

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
    router.serve("POST", "/v1/agents/sessions/sess_1/stop", { status: 204 });
    log = [];
    calls = [];
    failJoin = undefined;
    api.use({
      video: {
        streamClient: { userID: "john" },
        call: (type: string, id: string) => {
          const call = fakeCall(type, id, log, failJoin ? { join: failJoin } : {});
          calls.push(call);
          return call;
        },
        disconnectUser: async () => undefined,
      },
    });
  });

  afterEach(async () => {
    await router.stop();
  });

  async function started(): Promise<[VoiceSession, Connection]> {
    const starting = VoiceSession.start(api.agent("john"));
    const connection = await router.socket();
    return [await starting, connection];
  }

  it("starts a session with voice on, for the agent named", async () => {
    const [voice] = await started();
    const created = router.received.find((each) => each.path === "/v1/agents/sessions");
    assert.deepEqual(created?.body, { agent: "john", start_voice: true });
    assert.equal(voice.get().status, "connecting");
    await voice.end();
  });

  it("starts a session from a page in the device's own zone", async () => {
    const [voice] = await inPage("Europe/Lisbon", started);
    const created = router.received.find((each) => each.path === "/v1/agents/sessions");
    assert.deepEqual(created?.body, { agent: "john", start_voice: true, timezone: "Europe/Lisbon" });
    await voice.end();
  });

  it("joins with the microphone on and the camera off, creating the call", async () => {
    const [voice] = await started();
    await voice.join();

    assert.deepEqual(log, ["camera.off", "mic.on", 'join {"create":true}']);
    assert.equal(voice.call, calls[0]);
    assert.deepEqual(
      { muted: voice.get().muted, cameraEnabled: voice.get().cameraEnabled, status: voice.get().status },
      { muted: false, cameraEnabled: false, status: "idle" },
    );
    await voice.end();
  });

  it("picks the back lens before the camera is turned on, so nothing else is published", async () => {
    const [voice] = await started();
    await voice.join({ camera: true });

    assert.deepEqual(log.slice(0, 3), ["camera.back", "camera.on", "mic.on"]);
    assert.equal(voice.get().cameraEnabled, true);
    assert.ok(voice.session.cameraFrames, "the agent can ask what the camera showed");
    await voice.end();
    assert.equal(voice.session.cameraFrames, undefined);
  });

  it("joins once however many times it is asked", async () => {
    const [voice] = await started();
    await Promise.all([voice.join(), voice.join()]);
    await voice.join();
    assert.equal(log.filter((each) => each.startsWith("join")).length, 1);
    await voice.end();
  });

  it("rejects a failed join, lets go of the devices, and joins a fresh call when asked again", async () => {
    failJoin = new Error("Permission denied");
    const [voice] = await started();

    await assert.rejects(voice.join(), /Permission denied/);
    assert.deepEqual(log.slice(-2).sort(), ["camera.off", "mic.off"]);
    assert.equal(voice.get().status, "connecting");

    failJoin = undefined;
    // The client hands back fresh calls from here on; the failed one is never reused.
    calls.splice(0);
    await voice.join();
    assert.equal(calls.length, 1);
    assert.equal(voice.call, calls[0]);
    assert.deepEqual([calls[0]?.type, calls[0]?.id], ["agent", "sess_1"]);
    await voice.end();
  });

  it("mutes and unmutes the microphone", async () => {
    const [voice] = await started();
    await voice.setMuted(true);
    assert.equal(voice.get().muted, false, "nothing to mute before joining");

    await voice.join();
    await voice.setMuted(true);
    assert.equal(voice.get().muted, true);
    await voice.setMuted(false);
    assert.equal(voice.get().muted, false);
    assert.deepEqual(log.slice(-2), ["mic.off", "mic.on"]);
    await voice.end();
  });

  it("turns the camera on with the back lens, and forgets what it showed when off", async () => {
    const [voice] = await started();
    await voice.join();
    await voice.setCameraEnabled(true);
    assert.ok(voice.session.cameraFrames);
    voice.session.cameraFrames.record("AAAA", Date.now());

    await voice.setCameraEnabled(false);
    assert.equal(voice.session.cameraFrames, undefined);
    assert.equal(voice.get().cameraEnabled, false);
    assert.deepEqual(log.slice(-3), ["camera.back", "camera.on", "camera.off"]);
    await voice.end();
  });

  it("closes the session it started when it leaves", async () => {
    const [voice] = await started();
    await voice.join();
    await voice.leave();

    assert.equal(log.at(-1), "leave");
    assert.equal(voice.session.live, false);
    assert.equal(voice.call, undefined);
    assert.equal(voice.get().status, "idle", "a call that is over is not still connecting");
  });

  it("connects again when it joins after leaving", async () => {
    const opening = Session.open(api, { agent: "john", start_voice: true });
    await router.socket();
    const session = await opening;
    const voice = VoiceSession.of(session);
    await voice.join();
    await voice.leave();

    const seen: VoiceStatus[] = [];
    voice.subscribe(() => seen.push(voice.get().status));
    await voice.join();
    assert.deepEqual(seen, ["connecting", "idle"]);
    await voice.end();
  });

  it("leaves a session it was handed running, unless it ends it", async () => {
    const opening = Session.open(api, { agent: "john", start_voice: true });
    await router.socket();
    const session = await opening;

    const visitor = VoiceSession.of(session);
    await visitor.join();
    await visitor.leave();
    assert.equal(session.live, true);

    const host = VoiceSession.of(session);
    await host.join();
    await host.end();
    assert.equal(session.live, false);
  });

  it("refuses a session held in writing", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_2",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });
    const opening = Session.open(api, { agent: "john" });
    await router.socket();
    const session = await opening;

    assert.throws(() => VoiceSession.of(session), ConfigurationError);
    await session.close();
  });

  it("shows what the agent is doing, with speaking taken from the call", async () => {
    const [voice, connection] = await started();
    const seen: VoiceStatus[] = [];
    voice.subscribe(() => seen.push(voice.get().status));
    await voice.join();
    const call = calls[0]!;

    // Fed through the socket, in the order a turn goes.
    async function after(frame: Record<string, unknown>): Promise<void> {
      connection.send(frame);
      for await (const event of voice.session.events()) {
        if (event.kind === frame["type"]) {
          return;
        }
      }
    }

    call.state.localParticipant$.next({ isSpeaking: true });
    await after({ type: "hearing" });
    call.state.localParticipant$.next({ isSpeaking: false });
    await after({ type: "heard" });
    await after({ type: "responding" });
    await after({ type: "delegated", skill: "vision" });
    await after({ type: "task_settled", skill: "vision" });
    call.state.remoteParticipants$.next([{ isSpeaking: true }]);
    await after({ type: "responded" });
    call.state.remoteParticipants$.next([{ isSpeaking: false }]);

    assert.deepEqual(seen, ["idle", "listening", "thinking", "speaking", "idle"]);
    await voice.end();
  });

  describe("a reply written before it is spoken", () => {
    const reply = "Octopuses have three hearts.";

    /** Sends a frame down the socket, and waits until the session has read it. */
    async function fed(voice: VoiceSession, connection: Connection, frame: Record<string, unknown>) {
      connection.send(frame);
      for await (const event of voice.session.events()) {
        if (event.kind === frame["type"]) {
          return;
        }
      }
    }

    /** A turn written in full, and its voice starting, without the call hearing it yet. */
    async function written(voice: VoiceSession, connection: Connection): Promise<void> {
      await fed(voice, connection, { type: "responding", turn_id: "t1" });
      await fed(voice, connection, { type: "response_delta", turn_id: "t1", text: reply });
      await fed(voice, connection, { type: "responded", turn_id: "t1", text: reply });
      assert.equal(voice.get().status, "thinking", "written, and nothing said yet");
      await fed(voice, connection, {
        type: "speaking",
        turn_id: "t1",
        synthesis_id: "t1",
        text: reply,
        at: "2026-10-27T12:00:00Z",
      });
    }

    it("thinks until the call hears it, then speaks", async () => {
      const [voice, connection] = await started();
      await voice.join();
      const call = calls[0]!;
      const seen: VoiceStatus[] = [];
      voice.subscribe(() => {
        if (seen.at(-1) !== voice.get().status) {
          seen.push(voice.get().status);
        }
      });

      await written(voice, connection);
      assert.equal(voice.get().status, "thinking", "the session says it started; the call has not heard it");
      call.state.remoteParticipants$.next([{ isSpeaking: true }]);
      call.state.remoteParticipants$.next([{ isSpeaking: false }]);

      assert.deepEqual(seen, ["thinking", "speaking", "idle"]);
      await voice.end();
    });

    it("stops thinking when the call never hears it", async () => {
      const [voice, connection] = await started();
      await voice.join();

      await written(voice, connection);
      assert.equal(voice.get().status, "thinking");
      await new Promise((resolve) => setTimeout(resolve, 1_600));

      assert.equal(voice.get().status, "idle");
      await voice.end();
    });
  });

  it("stops following the call once it has left", async () => {
    const [voice] = await started();
    await voice.join();
    const call = calls[0]!;
    await voice.end();

    assert.equal(call.state.remoteParticipants$.observed, 0);
    assert.equal(call.state.localParticipant$.observed, 0);
    let told = 0;
    voice.subscribe(() => told++);
    call.state.remoteParticipants$.next([{ isSpeaking: true }]);
    assert.equal(told, 0);
  });
});
