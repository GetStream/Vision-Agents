import { describe, expect, it } from "vitest";

import { VoiceMeter } from "../src/voiceMeter";
import { troubleText } from "../src/voicePresence";

describe("VoiceMeter", () => {
  it("counts a voice as talking through the pauses between words, then lets it go", () => {
    const meter = new VoiceMeter(0.04);

    expect(meter.read(0.5, 0)).toBe(0.5);
    expect(meter.read(0.01, 300)).toBe(0.01);
    expect(meter.read(0.01, 700)).toBeUndefined();
  });

  it("ignores room noise that never holds long enough to be somebody talking", () => {
    const meter = new VoiceMeter(0.1, 120);

    expect(meter.read(0.3, 0)).toBeUndefined();
    expect(meter.read(0.3, 150)).toBe(0.3);
  });

  it("says nothing while there is no audio to read", () => {
    expect(new VoiceMeter(0.04).read(null, 0)).toBeUndefined();
  });
});

describe("troubleText", () => {
  it("names the part of Juno that failed, in words the person needs", () => {
    expect(troubleText("stt")).toContain("can't hear you");
    expect(troubleText("tts")).toContain("can't speak");
    expect(troubleText("sts")).toBe(troubleText("llm"));
    expect(troubleText("")).toBe("Something went wrong on Juno's side.");
  });
});
