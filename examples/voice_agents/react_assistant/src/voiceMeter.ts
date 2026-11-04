/**
 * Whether one side of the call is talking, and how loud, from its audio level as it flows.
 *
 * The call's own speaking flags change a few times a second, too coarse to move with the words.
 * The level is read on each frame of the glow instead, from `useAudioLevel` in Stream's AI
 * components, so the glow swells on each syllable and settles in each pause. This adds what the
 * glow needs on top: when a voice counts as talking, and when it has stopped. A port of the gate
 * in the iOS and Android apps' `VoiceMeter`.
 */
export class VoiceMeter {
  /** How long a voice counts as talking after it was last heard, so pauses read as pauses. */
  private static readonly HOLD_MS = 600;

  private lastAudible = Number.NEGATIVE_INFINITY;
  private stretchStart = Number.NEGATIVE_INFINITY;

  /**
   * @param audible The level, from 0 to 1, below which the voice counts as silent.
   * @param onset How long the voice has to be heard, in milliseconds, before it counts as talking.
   */
  constructor(
    private readonly audible: number,
    private readonly onset = 0,
  ) {}

  /** The voice's `level` at `now`, from 0 to 1, or undefined when it is not talking. */
  read(level: number | null, now: number): number | undefined {
    if (level === null) {
      return undefined;
    }
    if (level > this.audible) {
      if (now - this.lastAudible >= VoiceMeter.HOLD_MS) {
        this.stretchStart = now;
      }
      this.lastAudible = now;
    }
    if (now - this.lastAudible >= VoiceMeter.HOLD_MS || this.lastAudible - this.stretchStart < this.onset) {
      return undefined;
    }
    return level;
  }
}
