import { describe, expect, it } from "vitest";

import { isReservationOf, RESERVE_TABLE, reserveTable, type DiningTable, type Reservation } from "../src/reserveTable";

const tables: DiningTable[] = [
  { id: "T1", zone: "Terrace", seats: 2, available: false },
  { id: "T4", zone: "Terrace", seats: 4, available: true },
  { id: "B2", zone: "Bar", seats: 2, available: true },
];

/** Calls the seat selector's tool as the model would, with what it heard reserved. */
async function reserve(args: Record<string, string>) {
  const heard: Reservation[] = [];
  const tools = reserveTable(tables, "Alex", (made) => heard.push(made));
  const answer = tools.call(RESERVE_TABLE, JSON.stringify(args), new AbortController().signal);
  return { answer, heard };
}

describe("reserveTable", () => {
  it("asks first, showing the booking as the person would read it", () => {
    const tools = reserveTable(tables, "Alex", () => undefined);
    expect(tools.approval(RESERVE_TABLE)?.reasonArgument).toBe("summary");
  });

  it("books a free table on the card, under the person's name, with a confirmation code", async () => {
    const { answer, heard } = await reserve({ table_id: "T4", when: "tonight at 8pm", summary: "Terrace, for 4" });

    const text = String(await answer);
    expect(heard).toHaveLength(1);
    const [made] = heard;
    expect(made?.table.id).toBe("T4");
    expect(made?.code).toMatch(/^[0-9A-F]{6}$/);
    expect(made?.tables).toEqual(["T1", "T4", "B2"]);
    expect(text).toContain("Reserved table T4");
    expect(text).toContain("under Alex");
    expect(text).toContain(made!.code);
  });

  it("refuses a taken table and names the free ones", async () => {
    const { answer, heard } = await reserve({ table_id: "T1", when: "tonight at 8pm", summary: "Terrace, for 2" });

    await expect(answer).rejects.toThrow("T1 is not a free table on the seat selector. The free ones are T4, B2.");
    expect(heard).toEqual([]);
  });

  it("refuses a table no card offered", async () => {
    const { answer, heard } = await reserve({ table_id: "T9", when: "tonight", summary: "somewhere" });

    await expect(answer).rejects.toThrow("T9 is not a free table");
    expect(heard).toEqual([]);
  });

  it("refuses a booking that says no time", async () => {
    const { answer, heard } = await reserve({ table_id: "T4", when: "  ", summary: "Terrace, for 4" });

    await expect(answer).rejects.toThrow("Say when the table is for.");
    expect(heard).toEqual([]);
  });
});

describe("isReservationOf", () => {
  const reservation: Reservation = { table: tables[1]!, when: "tonight", code: "ABC123", tables: ["T1", "T4", "B2"] };

  it("matches a card showing the same tables, in any order", () => {
    expect(isReservationOf(reservation, [tables[2]!, tables[0]!, tables[1]!])).toBe(true);
  });

  it("does not match a card of other tables, or of only some of them", () => {
    expect(isReservationOf(reservation, [tables[0]!, tables[1]!])).toBe(false);
    expect(isReservationOf(reservation, [...tables.slice(0, 2), { id: "C1", zone: "Courtyard", seats: 6, available: true }])).toBe(false);
    expect(isReservationOf(undefined, tables)).toBe(false);
  });
});
