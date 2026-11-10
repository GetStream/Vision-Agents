import { afterEach, describe, expect, it, vi } from "vitest";
import { parseAIUIComponentDefinition } from "stream-chat-react/ai-components";

import described from "../../swift_assistant/app/SwiftAssistant/UIComponents.json";
import { calendarDays } from "../src/reminders";

describe("the trip components", () => {
  it("reads the components the iOS app describes, by the names the agent places them by", () => {
    const definitions = (described as unknown[]).map(parseAIUIComponentDefinition);
    expect(definitions.map((each) => each?.name)).toEqual(["place_card", "trip_map", "calendar_check"]);
    expect(definitions[2]?.parameters.map((each) => [each.name, each.type.format])).toEqual([
      ["start", "date"],
      ["end", "date"],
    ]);
  });
});

describe("calendarDays", () => {
  afterEach(() => vi.unstubAllGlobals());

  const keep = (reminders: { title: string; due: number }[]) =>
    vi.stubGlobal("localStorage", {
      getItem: () => JSON.stringify(reminders.map((each, index) => ({ id: String(index), ...each }))),
    });

  it("lists every day of a plan with what is on it, and the free ones as free", () => {
    keep([{ title: "Dentist", due: new Date(2026, 10, 14, 9, 30).getTime() }]);
    const days = calendarDays(new Date(2026, 10, 14), new Date(2026, 10, 15));
    expect(days.map(({ date }) => date.getDate())).toEqual([14, 15]);
    expect(days[0]?.events).toHaveLength(1);
    expect(days[0]?.events[0]).toContain("Dentist");
    expect(days[1]?.events).toEqual([]);
  });

  it("stops at two weeks, and reads a plan that ends before it starts as its first day", () => {
    keep([]);
    expect(calendarDays(new Date(2026, 10, 1), new Date(2026, 11, 31))).toHaveLength(14);
    expect(calendarDays(new Date(2026, 10, 14), new Date(2026, 10, 10))).toHaveLength(1);
  });
});
