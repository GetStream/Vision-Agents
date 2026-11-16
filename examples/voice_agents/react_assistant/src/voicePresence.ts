import type { AgentState, Turn } from "@stream-io/stream-agents";

/** What the glow shows during a call. */
export type Phase =
  | { readonly kind: "ready" }
  | { readonly kind: "listening" }
  | { readonly kind: "thinking" }
  | { readonly kind: "working"; readonly skills: readonly string[] }
  | { readonly kind: "speaking" };

/** The least time a phase is shown before the glow moves on. */
const MIN_PHASE_MS = 400;
/** How long a new phase has to hold before it is shown. */
const SETTLE_MS = 150;
/** How long the line has to stay quiet before the glow goes back to ready. */
const READY_SETTLE_MS = 800;
/** How long a written reply waits for its voice before the glow stops thinking. */
const AWAIT_VOICE_MS = 10_000;

const READY: Phase = { kind: "ready" };

/**
 * Who is talking and what the agent is doing, a phase at a time, as the Android app works it out.
 *
 * The agent speaking comes from its audio, since a reply is written before it is spoken. The
 * person speaking comes from the call, which marks them speaking from the audio it sends after
 * echo cancellation. From the moment the person has been heard until the agent's voice starts,
 * the agent is thinking, including the gap after its reply is written and before it is spoken.
 * A phase is held for at least `MIN_PHASE_MS`, so events arriving close together do not make the
 * glow flicker.
 */
export class Presence {
  phase: Phase = READY;

  private since: number;
  private wanted: Phase = READY;
  private wantedSince: number;
  private spokenTurn: string | undefined;
  private writtenTurn: string | undefined;
  private writtenAt: number;

  constructor(now: number) {
    this.since = now;
    this.wantedSince = now;
    this.writtenAt = now;
  }

  /**
   * Moves on to `now`. `turn` is the latest turn said on the call, if any: what was written
   * before it is left out, so only what is said on the call moves the glow.
   */
  step(input: {
    now: number;
    agentAudible: boolean;
    personSpeaking: boolean;
    state: AgentState;
    skills: readonly string[];
    turn: Turn | undefined;
  }): Phase {
    const { now, agentAudible, personSpeaking, state, skills, turn } = input;
    if (turn?.speaker === "agent" && turn.id !== this.writtenTurn) {
      this.writtenTurn = turn.id;
      this.writtenAt = now;
    }
    if (agentAudible && turn?.speaker === "agent") {
      this.spokenTurn = turn.id;
    }

    let target: Phase;
    if (agentAudible) {
      target = { kind: "speaking" };
    } else if (personSpeaking || state === "listening") {
      target = { kind: "listening" };
    } else if (state === "working") {
      target = { kind: "working", skills };
    } else if (state === "responding") {
      target = { kind: "thinking" };
    } else if (turn && turn.speaker !== "agent") {
      // Heard, and nothing written back yet.
      target = { kind: "thinking" };
    } else if (turn && turn.id !== this.spokenTurn && now - this.writtenAt < AWAIT_VOICE_MS) {
      // Written, and not spoken yet: the voice is still on its way.
      target = { kind: "thinking" };
    } else {
      target = READY;
    }

    if (!samePhase(target, this.wanted)) {
      this.wanted = target;
      this.wantedSince = now;
    }
    const settle = this.wanted.kind === "ready" ? READY_SETTLE_MS : SETTLE_MS;
    if (
      !samePhase(this.wanted, this.phase) &&
      now - this.since >= MIN_PHASE_MS &&
      now - this.wantedSince >= settle
    ) {
      this.phase = this.wanted;
      this.since = now;
    }
    return this.phase;
  }
}

function samePhase(a: Phase, b: Phase): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "working" && b.kind === "working") {
    return a.skills.join("\n") === b.skills.join("\n");
  }
  return true;
}

/** What the glow is showing, in words. */
export function phaseLabel(phase: Phase): string {
  switch (phase.kind) {
    case "ready":
      return "Ready";
    case "listening":
      return "Listening";
    case "thinking":
      return "Thinking";
    case "working":
      return workingLabel(phase.skills);
    case "speaking":
      return "Speaking";
  }
}

/** "Working on trip planning and recall", for skills `trip_planning` and `recall`. */
export function workingLabel(skills: readonly string[]): string {
  return "Working on " + skills.map((skill) => skill.replaceAll("_", " ")).join(" and ");
}

/** A failure as the person needs to hear it, by the part of the agent that failed. */
export function troubleText(context: string): string {
  switch (context) {
    case "stt":
      return "Juno can't hear you right now: speech recognition isn't working.";
    case "llm":
    case "sts":
      return "Juno can't answer right now: its model isn't working.";
    case "tts":
      return "Juno can't speak right now: its voice isn't working.";
    default:
      return "Something went wrong on Juno's side.";
  }
}
