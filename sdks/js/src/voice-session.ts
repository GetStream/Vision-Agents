import type { Call, StreamVideoClient, StreamVideoParticipant } from "@stream-io/video-client";

import { ConfigurationError } from "./errors.js";
import { CameraFrames } from "./frames.js";
import type { AgentHandle } from "./handle.js";
import type { Session } from "./session.js";
import type { CreateSessionOptions } from "./sessions.js";

/**
 * How long an utterance the session says has started may go unheard on the call before the
 * agent is taken to be done thinking anyway. The call reports a speaker a few hundred
 * milliseconds after the audio starts, and not at all for audio too quiet to count.
 */
const VOICE_DETECTION_MS = 1_500;

/**
 * What the agent is doing, as a voice visualization shows it: the states `VoiceGlow` and
 * `VoiceWave` in `stream-chat-react/ai-components` take.
 */
export type VoiceStatus = "connecting" | "idle" | "listening" | "thinking" | "speaking";

/** Where a voice session is. The same object until something in it changes. */
export interface VoiceSnapshot {
  /** The call this device is on, once it has joined. */
  readonly call?: Call;
  /** The `StreamVideoClient` the call was joined on, for `<StreamVideo client>`. */
  readonly client?: StreamVideoClient;
  /** Whether this device's microphone is off. */
  readonly muted: boolean;
  /** Whether this device's camera is on. */
  readonly cameraEnabled: boolean;
  /**
   * What the agent is doing. Connecting until joined; speaking while somebody else on the
   * call is heard, which is the agent; listening while this device is heard or the agent is
   * transcribing; thinking while it answers, its skills work, or its reply waits for its
   * voice; idle otherwise, and once this device has left.
   */
  readonly status: VoiceStatus;
}

/**
 * A spoken conversation: the agent on a call, and this device on the same call.
 *
 * Two things happen, in this order:
 *
 * 1. The router starts a session with voice on, which puts the agent on the session's own
 *    call, `agent:<session id>`.
 * 2. Stream's Video SDK joins that call as the user `setUser` named, with `join()`.
 *
 * The transcript comes over the session socket and into its Chat channel rather than out of
 * the call, so `session` is the same `Session` a text conversation uses: `session.chat()`
 * opens its transcript.
 *
 * Playing the agent is the page's: `<ParticipantsAudio />` from `@stream-io/video-react-sdk`
 * in React, or `call.bindAudioElement` for each remote participant without one. Start it from
 * a click, since a browser plays audio and asks for the microphone only after one.
 *
 * `get` and `subscribe` are bound, so a page can hand them over as they are:
 *
 * ```ts
 * const { call, muted, status } = useSyncExternalStore(voice.subscribe, voice.get);
 * ```
 */
export class VoiceSession {
  /** The conversation: the transcript, and what the agent is doing. */
  readonly session: Session;

  /**
   * True when `start` created the session, false when it was handed over with `of`. Leaving
   * closes only a session this device owns.
   */
  private readonly owned: boolean;
  /** What the camera showed lately, while it is on, for the agent to look at. */
  private readonly frames = new CameraFrames();
  private readonly listeners = new Set<() => void>();
  private snapshot: VoiceSnapshot = Object.freeze({
    muted: false,
    cameraEnabled: false,
    status: "connecting" as const,
  });
  private joining: Promise<void> | undefined;
  /** The call the last flow ended on, so a later one is joined on a fresh instance of it. */
  private previous: Call | undefined;
  /** Whether this device has left the call, which shows as idle rather than connecting. */
  private left = false;
  private remoteSpeaking = false;
  private localSpeaking = false;
  /** Set while an utterance has started and the call has not heard it yet. */
  private starting: ReturnType<typeof setTimeout> | undefined;
  private readonly stops: (() => void)[] = [];
  private sampler: FrameSampler | undefined;

  private constructor(session: Session, owned: boolean) {
    this.session = session;
    this.owned = owned;
  }

  /**
   * Starts an agent on a new session's own call, `agent:<session id>`, and prepares to join
   * it. It resolves once the agent is there.
   *
   * ```ts
   * const voice = await VoiceSession.start(api.agent("my-agent"));
   * await voice.join();
   * ```
   */
  static async start(agent: AgentHandle, options: CreateSessionOptions = {}): Promise<VoiceSession> {
    const session = await agent.sessions.create({ ...options, start_voice: true });
    return new VoiceSession(session, true);
  }

  /**
   * Joins the call of a session that already has voice on: one switched over with
   * `session.voice.start()`, or carried on with `sessions.resume(id)`. Leaving it leaves the
   * session running; `end` closes it.
   */
  static of(session: Session): VoiceSession {
    if (!session.voice.started) {
      throw new ConfigurationError(
        "this conversation is held in writing, so there is no call to join: start voice with " +
          "session.voice.start() first",
      );
    }
    return new VoiceSession(session, false);
  }

  /** Where the voice session is now. */
  readonly get = (): VoiceSnapshot => this.snapshot;

  /** Calls `listener` whenever the snapshot changes. It returns what stops it. */
  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  /** The call this device is on, once it has joined. */
  get call(): Call | undefined {
    return this.snapshot.call;
  }

  /**
   * Joins the call from this device, with the microphone on. The camera is off unless
   * `camera` is true, and then starts on the back lens where there is one, since what an agent
   * is shown is whatever the caller points at. Both are set before joining, so nothing else
   * is ever published.
   *
   * It joins on the `StreamVideoClient` handed over with `client.use`, or else on one built for
   * the client's key and user, which every session shares. Joining twice joins once: a second
   * call while the first is under way waits for it. A failed join rejects, and can be tried
   * again.
   */
  join(options: { camera?: boolean } = {}): Promise<void> {
    this.joining ??= this.connect(options.camera ?? false).catch((cause: unknown) => {
      this.joining = undefined;
      throw cause;
    });
    return this.joining;
  }

  /** Turns this device's microphone off or on. The agent stays on the call either way. */
  async setMuted(muted: boolean): Promise<void> {
    const call = this.snapshot.call;
    if (!call) {
      return;
    }
    await (muted ? call.microphone.disable() : call.microphone.enable());
    this.change({ muted });
  }

  /**
   * Turns this device's camera on or off. Turning it on picks the back lens again.
   *
   * While it is on, the agent can look through it: its vision skill asks for what the camera
   * showed when a question was put, and `session` answers with those frames.
   */
  async setCameraEnabled(enabled: boolean): Promise<void> {
    const call = this.snapshot.call;
    if (!call) {
      return;
    }
    if (enabled) {
      await call.camera.selectDirection("back", { enableCamera: false });
      await call.camera.enable();
      this.look(call);
    } else {
      await call.camera.disable();
      this.stopLooking();
    }
    this.change({ cameraEnabled: enabled });
  }

  /**
   * Leaves the call. Closes the session too when `start` created it; a session handed over
   * with `of` is left running, for whoever else is on it.
   */
  leave(): Promise<void> {
    return this.hangUp(this.owned);
  }

  /** Hangs up: leaves the call and ends the session, whoever started it. */
  end(): Promise<void> {
    return this.hangUp(true);
  }

  private async connect(camera: boolean): Promise<void> {
    // Joining again after leaving is connecting again.
    this.left = false;
    this.change({});
    const video = await this.session.video();
    const client = video.client as unknown as StreamVideoClient;
    // One Call is one join: after a failed join or a leave, the next one is a fresh instance.
    const call = this.previous
      ? client.call(this.previous.type, this.previous.id)
      : (video.call as Call);
    this.previous = call;

    try {
      if (camera) {
        // Does nothing where the browser cannot tell lenses apart, such as on a laptop.
        await call.camera.selectDirection("back", { enableCamera: false });
        await call.camera.enable();
      } else {
        await call.camera.disable();
      }
      await call.microphone.enable();
      // Created rather than only joined: which of this device and the agent arrives first is
      // a race, and the agent's own join creates it the same way.
      await call.join({ create: true });
    } catch (cause) {
      // Not left holding the microphone or the camera for a call this device never got on.
      await Promise.allSettled([call.microphone.disable(), call.camera.disable()]);
      throw cause;
    }

    this.follow(call);
    if (camera) {
      this.look(call);
    }
    this.change({ call, client, muted: false, cameraEnabled: camera });
  }

  /** Follows who is speaking on the call and what the conversation is doing, for `status`. */
  private follow(call: Call): void {
    const remote = call.state.remoteParticipants$.subscribe((participants) => {
      this.remoteSpeaking = participants.some(speaking);
      if (this.remoteSpeaking) {
        this.heardOnCall();
      }
      this.change({});
    });
    const local = call.state.localParticipant$.subscribe((participant) => {
      this.localSpeaking = participant ? speaking(participant) : false;
      this.change({});
    });
    let utterance = this.session.speech.get()?.synthesisId;
    const speech = this.session.speech.subscribe(() => {
      const started = this.session.speech.get()?.synthesisId;
      if (started !== undefined && started !== utterance) {
        utterance = started;
        this.voiceStarting();
      }
      this.change({});
    });
    this.stops.push(
      () => remote.unsubscribe(),
      () => local.unsubscribe(),
      this.session.state.subscribe(() => this.change({})),
      speech,
      () => this.heardOnCall(),
    );
  }

  /**
   * An utterance has started, by the session's word, and the call has not heard it yet: the
   * call says who is speaking a moment after the audio starts. Still thinking until it does,
   * or until `VOICE_DETECTION_MS` says it is not going to.
   */
  private voiceStarting(): void {
    if (this.remoteSpeaking) {
      return;
    }
    clearTimeout(this.starting);
    this.starting = setTimeout(() => {
      this.starting = undefined;
      this.change({});
    }, VOICE_DETECTION_MS);
  }

  /** The agent is heard on the call, or this device left it: nothing is starting any more. */
  private heardOnCall(): void {
    clearTimeout(this.starting);
    this.starting = undefined;
  }

  private async hangUp(close: boolean): Promise<void> {
    await this.joining?.catch(() => undefined);
    const call = this.snapshot.call;
    this.stopLooking();
    for (const stop of this.stops.splice(0)) {
      stop();
    }
    this.joining = undefined;
    this.left = true;
    this.remoteSpeaking = false;
    this.localSpeaking = false;
    this.snapshot = Object.freeze({ muted: false, cameraEnabled: false, status: "idle" });
    this.notify();
    try {
      await call?.leave();
    } finally {
      if (close) {
        await this.session.close();
      }
    }
  }

  /** Samples the call's camera for the agent, and lets `session` answer with what it showed. */
  private look(call: Call): void {
    this.sampler?.stop();
    this.sampler = FrameSampler.start(call, this.frames);
    this.session.cameraFrames = this.frames;
  }

  private stopLooking(): void {
    this.sampler?.stop();
    this.sampler = undefined;
    this.frames.clear();
    this.session.cameraFrames = undefined;
  }

  private status(joined: boolean): VoiceStatus {
    if (!joined) {
      return this.left ? "idle" : "connecting";
    }
    // Speaking comes from the call: a reply is written before it is spoken, and is done
    // before the agent stops talking. Until it is heard, the agent is still thinking.
    if (this.remoteSpeaking) {
      return "speaking";
    }
    if (this.localSpeaking) {
      return "listening";
    }
    switch (this.session.state.get().kind) {
      case "listening":
        return "listening";
      case "responding":
      case "working":
        return "thinking";
      default:
        return this.session.speech.awaiting || this.starting !== undefined ? "thinking" : "idle";
    }
  }

  /** Applies `patch` and works `status` out again, telling listeners only if anything moved. */
  private change(patch: Partial<Omit<VoiceSnapshot, "status">>): void {
    const merged = { ...this.snapshot, ...patch };
    const next: VoiceSnapshot = { ...merged, status: this.status(merged.call !== undefined) };
    const current = this.snapshot;
    if (
      next.call === current.call &&
      next.client === current.client &&
      next.muted === current.muted &&
      next.cameraEnabled === current.cameraEnabled &&
      next.status === current.status
    ) {
      return;
    }
    this.snapshot = Object.freeze(next);
    this.notify();
  }

  private notify(): void {
    for (const listener of [...this.listeners]) {
      try {
        listener();
      } catch (cause) {
        // A call's state is read on this call, and one listener's mistake must not stop it.
        queueMicrotask(() => {
          throw cause;
        });
      }
    }
  }
}

function speaking(participant: StreamVideoParticipant): boolean {
  return participant.isSpeaking === true;
}

/**
 * Keeps a frame of the call's own camera twice a second, scaled down and encoded as JPEG,
 * for the agent to look at. Draws the camera's stream into a video element nobody sees, and
 * that onto a canvas. Does nothing where there is no DOM to draw with.
 */
class FrameSampler {
  private timer: ReturnType<typeof setInterval> | undefined;
  private readonly video: HTMLVideoElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly stream: { unsubscribe(): void };
  private busy = false;
  private stopped = false;

  private constructor(call: Call, private readonly frames: CameraFrames) {
    this.video = document.createElement("video");
    this.video.muted = true;
    this.video.playsInline = true;
    this.canvas = document.createElement("canvas");
    this.stream = call.camera.state.mediaStream$.subscribe((stream) => {
      this.video.srcObject = stream ?? null;
      if (stream) {
        this.video.play().catch(() => undefined);
      }
    });
    this.timer = setInterval(() => void this.capture(), CameraFrames.every);
  }

  static start(call: Call, frames: CameraFrames): FrameSampler | undefined {
    return typeof document === "undefined" ? undefined : new FrameSampler(call, frames);
  }

  stop(): void {
    this.stopped = true;
    clearInterval(this.timer);
    this.timer = undefined;
    this.stream.unsubscribe();
    this.video.srcObject = null;
  }

  private async capture(): Promise<void> {
    const { videoWidth: width, videoHeight: height } = this.video;
    if (this.busy || width === 0 || height === 0) {
      return;
    }
    this.busy = true;
    const capturedAt = Date.now();
    try {
      const scale = Math.min(1, CameraFrames.longestSide / Math.max(width, height));
      this.canvas.width = Math.round(width * scale);
      this.canvas.height = Math.round(height * scale);
      this.canvas.getContext("2d")?.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
      const blob = await new Promise<Blob | null>((resolve) =>
        this.canvas.toBlob(resolve, "image/jpeg", CameraFrames.quality),
      );
      // A camera turned off while this was encoding showed nothing the agent should see.
      if (blob && !this.stopped) {
        this.frames.record(base64(new Uint8Array(await blob.arrayBuffer())), capturedAt);
      }
    } finally {
      this.busy = false;
    }
  }
}

function base64(bytes: Uint8Array): string {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  }
  return btoa(binary);
}
