import { describe, expect, it } from "vitest";

import { isoDay, parseDay } from "../src/dates";
import { fitWithin } from "../src/photos";
import { daySections, dayTitle } from "../src/sections";

describe("fitWithin", () => {
  it("shrinks a photo so its longest side fits, keeping its shape", () => {
    expect(fitWithin(4032, 3024, 1280)).toEqual({ width: 1280, height: 960 });
    expect(fitWithin(3024, 4032, 1280)).toEqual({ width: 960, height: 1280 });
  });

  it("never enlarges a small one", () => {
    expect(fitWithin(640, 480, 1280)).toEqual({ width: 640, height: 480 });
  });
});

describe("days", () => {
  it("reads a day the model names, and refuses one that is not a real date", () => {
    expect(isoDay(parseDay("2026-10-16")!)).toBe("2026-10-16");
    expect(parseDay("2026-02-30")).toBeUndefined();
    expect(parseDay("tomorrow")).toBeUndefined();
  });

  it("titles a conversation's day as Today, Yesterday, This week, or its month", () => {
    const now = new Date(2026, 10, 2, 15);
    expect(dayTitle(new Date(2026, 10, 2, 9), now)).toBe("Today");
    expect(dayTitle(new Date(2026, 10, 1, 23), now)).toBe("Yesterday");
    expect(dayTitle(new Date(2026, 9, 29), now)).toBe("This week");
    expect(dayTitle(new Date(2026, 8, 3), now)).toContain("2026");
  });

  it("groups conversations by day in the list's order, a day twice when the order says so", () => {
    const now = new Date(2026, 10, 2, 15);
    const items = [new Date(2026, 10, 2), new Date(2026, 10, 1), new Date(2026, 10, 2)];

    expect(daySections(items, (each) => each, now).map((day) => [day.title, day.items.length])).toEqual([
      ["Today", 1],
      ["Yesterday", 1],
      ["Today", 1],
    ]);
  });
});
