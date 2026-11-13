import { afterEach, describe, expect, it, vi } from "vitest";
import { parseAIUIComponentDefinition } from "stream-chat-react/ai-components";

import described from "../../swift_assistant/app/SwiftAssistant/UIComponents.json";
import bookingDescribed from "../src/bookingComponents.json";
import { applyBrandProfile, BRAND_PROFILES } from "../src/brandTheme";
import { calendarDays } from "../src/reminders";

describe("the trip and hospitality booking components", () => {
  it("reads the portable trip and booking components by the names the agent places them by", () => {
    const definitions = [...(described as unknown[]), ...(bookingDescribed as unknown[])].map(
      parseAIUIComponentDefinition,
    );
    expect(definitions.map((each) => each?.name)).toEqual([
      "place_card",
      "trip_map",
      "calendar_check",
      "restaurant_card",
      "availability_gauge",
      "seat_selector",
    ]);
    expect(definitions[2]?.parameters.map((each) => [each.name, each.type.format])).toEqual([
      ["start", "date"],
      ["end", "date"],
    ]);
    expect(definitions[3]?.parameters.map((each) => each.name)).toEqual([
      "name",
      "cuisine",
      "rating",
      "price_tier",
    ]);
  });

  it("applies brand token overrides onto the document root for both custom and OpenUI views", () => {
    const props = new Map<string, string>();
    vi.stubGlobal("document", {
      documentElement: {
        dataset: {} as Record<string, string>,
        style: {
          setProperty: (k: string, v: string) => props.set(k, v),
        },
      },
    });
    applyBrandProfile("ember");
    const ember = BRAND_PROFILES.find((p) => p.id === "ember")!;
    expect(props.get("--str-chat__ai-accent")).toBe(ember.accent);
    expect(props.get("--brand-radius")).toBe(`${ember.radiusPx}px`);
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
