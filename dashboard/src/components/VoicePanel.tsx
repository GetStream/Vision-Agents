"use client";

import { useEffect, useRef, useState } from "react";

import type { Voice, VoiceState } from "@/lib/useSession";
import { startVoiceOrb, type Mood } from "./voiceOrb";

/**
 * How each state looks. The colours are the ones the rest of the dashboard already
 * uses for the same ideas: blue for the caller, amber for work in progress, green
 * for the agent.
 */
const moods: Record<VoiceState, Mood & { label: string }> = {
  idle: { label: "quiet", tint: [0.36, 0.4, 0.52], energy: 0, presence: 0.22 },
  listening: {
    label: "listening",
    tint: [0.24, 0.72, 0.98],
    energy: 0.3,
    presence: 0.85,
  },
  thinking: {
    label: "thinking",
    tint: [0.98, 0.68, 0.26],
    energy: 0.2,
    presence: 0.6,
  },
  speaking: {
    label: "speaking",
    tint: [0.28, 0.88, 0.6],
    energy: 0.65,
    presence: 1,
  },
};

/** A state nothing follows within this long has been abandoned, so the orb settles. */
const stale = 12_000;

/** How many words in one utterance count as full effort. */
const loud = 24;

function stateOf(voice: Voice): VoiceState {
  if (voice.state !== "idle" && Date.now() - voice.at > stale) {
    return "idle";
  }
  return voice.state;
}

function moodOf(voice: Voice): Mood {
  const mood = moods[stateOf(voice)];
  if (mood === moods.idle) {
    return mood;
  }
  // A long sentence should look like more than a one-word answer, which is as
  // close to loudness as a page with no audio can honestly get.
  const said = Math.min(voice.words / loud, 1);
  return { ...mood, energy: mood.energy * (0.55 + said * 0.45) };
}

export function VoicePanel({
  voice,
  className = "",
}: {
  voice: Voice;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const latest = useRef<Voice>(voice);
  const [drawn, setDrawn] = useState(true);
  const [, settle] = useState(0);

  useEffect(() => {
    latest.current = voice;
  }, [voice]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    return startVoiceOrb(canvas, () => moodOf(latest.current), () =>
      setDrawn(false),
    );
  }, []);

  // The orb reads the clock every frame, but the caption below it only changes when
  // React renders, so a state nothing followed needs a nudge to stop reading "thinking".
  useEffect(() => {
    if (voice.state === "idle") {
      return;
    }
    const left = Math.max(stale - (Date.now() - voice.at), 0);
    const timer = setTimeout(() => settle((n) => n + 1), left + 100);
    return () => clearTimeout(timer);
  }, [voice]);

  const label = moods[stateOf(voice)].label;

  return (
    <section
      className={`relative overflow-hidden rounded-xl border border-line bg-[#0a0a0d] ${className}`}
    >
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`The call is ${label}`}
        className="block h-40 w-full"
      />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between px-4 py-3 text-xs">
        <span className="font-medium uppercase tracking-wide text-white/80">
          {label}
        </span>
        <span className="text-white/40">
          {drawn ? "who holds the floor" : "this browser has no WebGPU"}
        </span>
      </div>
    </section>
  );
}
