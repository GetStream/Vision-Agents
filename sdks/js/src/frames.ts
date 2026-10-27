/**
 * What this device's camera showed lately, kept so the agent can look when it asks.
 *
 * The router asks with `get_video_frames` when its vision skill needs to see, naming the
 * moment it took the question and how many frames it wants. A `VoiceSession` with the camera
 * on samples the call's own camera twice a second and keeps the newest eight here: about four
 * seconds. Nothing here touches a camera, so it is the same in a browser and in a test.
 */
export class CameraFrames {
  /** The tool the router asks for frames with. Answered by the session, never offered to the model. */
  static readonly toolName = "get_video_frames";
  /** How often a frame is kept, in milliseconds. */
  static readonly every = 500;
  /** How many are kept. */
  static readonly keep = 8;
  /** How old the newest may be at the moment asked about, as the router's own workers allow. */
  static readonly freshFor = 5_000;
  /**
   * How far this device's clock may run ahead of the router's: a frame taken just after the
   * question by this clock may be just before it by the router's.
   */
  static readonly skew = 2_000;
  /** The longest side a frame is kept at, in pixels. */
  static readonly longestSide = 1024;
  /** JPEG quality a frame is encoded at. */
  static readonly quality = 0.7;

  private kept: CameraFrame[] = [];
  private counter = 0;

  /** Keeps one frame, a base64 JPEG, taken at `capturedAt` (Unix milliseconds). */
  record(jpeg: string, capturedAt: number = Date.now()): void {
    this.counter += 1;
    this.kept.push({ id: this.counter, capturedAt, jpeg });
    if (this.kept.length > CameraFrames.keep) {
      this.kept.splice(0, this.kept.length - CameraFrames.keep);
    }
  }

  /** Forgets every frame, for a camera turned off: what it showed before is not what it shows. */
  clear(): void {
    this.kept = [];
  }

  /**
   * The newest `limit` frames taken at or before `at` (Unix milliseconds), oldest first, as the
   * router's own workers pick them: never a newer frame in place of one the question was asked
   * about. Throws `NoFrames` when the camera has shown nothing recent enough.
   */
  select(at: number, limit: number): CameraFrame[] {
    const before = this.kept.filter((frame) => frame.capturedAt <= at + CameraFrames.skew);
    const newest = before.at(-1);
    if (!newest || at - newest.capturedAt > CameraFrames.freshFor) {
      throw new NoFrames();
    }
    const count = Math.max(1, Math.min(Number.isFinite(limit) ? Math.floor(limit) : 1, CameraFrames.keep));
    return before.slice(-count);
  }
}

/** One frame the camera showed. */
export interface CameraFrame {
  /** Counts up from 1 for the life of the store, so the agent can tell frames apart. */
  readonly id: number;
  /** When it was taken, in Unix milliseconds. */
  readonly capturedAt: number;
  /** The frame as a base64 JPEG, without a `data:` prefix. */
  readonly jpeg: string;
}

/** The camera has shown nothing recent enough. */
export class NoFrames extends Error {
  constructor() {
    super(
      "The camera has shown nothing in the last few seconds; ask them to point it at what they mean again.",
    );
    this.name = "NoFrames";
  }
}
