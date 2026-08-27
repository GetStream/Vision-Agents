import { clock, effect, frameLoop, init, surface } from "vgpu";
import type { FrameLoopHandle, Gpu } from "vgpu";

import orbShader from "./voiceOrb.wgsl";

/** Mood is what the orb should look like: a colour and how hard it is working. */
export type Mood = {
  tint: readonly [number, number, number];
  energy: number;
  presence: number;
};

/**
 * How fast the orb catches up with a new mood, per second. Turns arrive in steps —
 * listening, thinking, speaking — and easing between them is what stops the panel
 * from flickering on a call with short exchanges.
 */
const chase = { tint: 5, energy: 6, presence: 2.5 };

function ease(from: number, to: number, rate: number, seconds: number): number {
  return from + (to - from) * (1 - Math.exp(-rate * seconds));
}

/**
 * startVoiceOrb renders the orb on `canvas` until the returned function is called.
 *
 * The mood is pulled once per frame rather than pushed, so the conversation can
 * change as often as it likes without touching the GPU or restarting the loop.
 */
export function startVoiceOrb(
  canvas: HTMLCanvasElement,
  mood: () => Mood,
  onFailure: (error: unknown) => void,
): () => void {
  let stopped = false;
  let loop: FrameLoopHandle | undefined;
  let gpu: Gpu | undefined;

  void (async () => {
    try {
      gpu = await init();
    } catch (error) {
      onFailure(error);
      return;
    }
    if (stopped) {
      gpu.dispose();
      return;
    }

    const screen = surface(gpu, canvas, { dpr: [1, 2] });
    const opening = mood();
    const shown = {
      tint: [...opening.tint] as [number, number, number],
      energy: opening.energy,
      presence: opening.presence,
    };
    const orb = effect(gpu, orbShader, {
      label: "voice-orb",
      set: {
        params: {
          tint: shown.tint,
          time: 0,
          energy: shown.energy,
          presence: shown.presence,
          texel: screen.texelSize,
        },
      },
    });
    screen.onResize(() => orb.set({ params: { texel: screen.texelSize } }));

    const time = clock(gpu);
    loop = frameLoop(gpu, (frame) => {
      const target = mood();
      const step = Math.min(time.deltaTime, 0.1);
      for (let channel = 0; channel < 3; channel++) {
        shown.tint[channel] = ease(
          shown.tint[channel],
          target.tint[channel],
          chase.tint,
          step,
        );
      }
      shown.energy = ease(shown.energy, target.energy, chase.energy, step);
      shown.presence = ease(shown.presence, target.presence, chase.presence, step);

      orb.set({
        params: {
          tint: shown.tint,
          time: time.time,
          energy: shown.energy,
          presence: shown.presence,
        },
      });
      frame.pass(screen, orb);
    });
  })();

  return () => {
    stopped = true;
    loop?.stop();
    gpu?.dispose();
  };
}
