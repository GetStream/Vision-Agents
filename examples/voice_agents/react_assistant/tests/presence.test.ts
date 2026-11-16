import type { Turn } from "@stream-io/stream-agents";
import { describe, expect, it } from "vitest";

import { Presence, phaseLabel, workingLabel } from "../src/voicePresence";

const idle = { agentAudible: false, personSpeaking: false, state: "idle" as const, skills: [], turn: undefined };

describe("Presence", () => {
  it("speaks while the agent is heard, after the phase has settled", () => {
    const presence = new Presence(0);

    expect(presence.step({ ...idle, now: 100, agentAudible: true }).kind).toBe("ready");
    expect(presence.step({ ...idle, now: 500, agentAudible: true }).kind).toBe("speaking");
  });

  it("thinks from the moment the person was heard until the agent's voice starts", () => {
    const presence = new Presence(0);
    const heard: Turn = { id: "p1", speaker: "participant", text: "Hi", at: 0 };

    presence.step({ ...idle, now: 400, turn: heard });
    expect(presence.step({ ...idle, now: 600, turn: heard }).kind).toBe("thinking");
  });

  it("holds a phase long enough that the glow does not flicker", () => {
    const presence = new Presence(0);
    presence.step({ ...idle, now: 400, personSpeaking: true });
    presence.step({ ...idle, now: 600, personSpeaking: true });

    expect(presence.step({ ...idle, now: 700, agentAudible: true }).kind).toBe("listening");
  });

  it("says what skills are working on, in words", () => {
    expect(workingLabel(["trip_planning", "recall"])).toBe("Working on trip planning and recall");
    expect(phaseLabel({ kind: "ready" })).toBe("Ready");
  });
});
