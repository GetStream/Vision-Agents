import type { Turn } from "@stream-io/calypso";
import { describe, expect, it } from "vitest";

import { Captions, sentenceIn } from "../src/captions";

const agent = (text: string, id = "t1"): Turn => ({ id, speaker: "agent", text, at: 0 });
const person = (text: string): Turn => ({ id: "p1", speaker: "participant", text, at: 0 });

describe("sentenceIn", () => {
  it("shows the sentence being said, from after the last one that ended", () => {
    expect(sentenceIn("Hi there. How are you?", 22)).toBe("How are you?");
  });

  it("holds back a word until all of it has been said", () => {
    expect(sentenceIn("Two meetings today", 8)).toBe("Two");
  });

  it("says nothing before the first word is out", () => {
    expect(sentenceIn("Lisbon", 3)).toBe("");
  });
});

describe("Captions", () => {
  it("reveals the agent's reply at the pace of its voice, not as it was written", () => {
    const captions = new Captions();
    const turn = agent("You have two meetings today and a dentist at five.");

    captions.update({ turn, agentIsAudible: true, agentIsAnswering: true, now: 1000, elapsed: 1000 });
    expect(captions.caption?.text).toBe("You have two");
    expect(captions.isVisible).toBe(true);
  });

  it("shows the rest once the agent has stopped for good, so a reply never ends half captioned", () => {
    const captions = new Captions();
    const turn = agent("Sure. Done.");

    captions.update({ turn, agentIsAudible: true, agentIsAnswering: true, now: 100, elapsed: 100 });
    captions.update({ turn, agentIsAudible: false, agentIsAnswering: false, now: 2000, elapsed: 1900 });
    expect(captions.caption?.text).toBe("Done.");
  });

  it("shows what the person said once transcribed, in their own style, and lets it go", () => {
    const captions = new Captions();

    captions.update({ turn: person("What's this?"), agentIsAudible: false, agentIsAnswering: false, now: 0, elapsed: 0 });
    expect(captions.caption).toEqual({ text: "What's this?", isPerson: true });
    expect(captions.isVisible).toBe(true);

    captions.update({ turn: person("What's this?"), agentIsAudible: false, agentIsAnswering: false, now: 3000, elapsed: 3000 });
    expect(captions.isVisible).toBe(false);
  });
});
