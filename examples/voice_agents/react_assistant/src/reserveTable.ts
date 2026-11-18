import { Tools } from "@stream-io/stream-agents";
import { z } from "zod";

/** One table a seat selector offers. */
export type DiningTable = { id: string; zone: string; seats: number; available: boolean };

/** A booking the seat selector's tool made, and the tables of the card it was made from. */
export type Reservation = {
  readonly table: DiningTable;
  readonly when: string;
  readonly code: string;
  readonly tables: readonly string[];
};

/** What the tool the seat selector brings is called. */
export const RESERVE_TABLE = "reserve_table";

/** Whether `reservation` was made from a card offering `tables`. */
export function isReservationOf(reservation: Reservation | undefined, tables: readonly DiningTable[]): boolean {
  if (!reservation || reservation.tables.length !== tables.length) return false;
  return tables.every((table) => reservation.tables.includes(table.id));
}

/** What a seat selector offers, told apart from another's: its tables and which are free. */
export function offerOf(tables: readonly DiningTable[]): string {
  return JSON.stringify(tables.map(({ id, zone, seats, available }) => [id, zone, seats, available]));
}

/**
 * Reserves one of the free tables a seat selector offers, under `guest`'s name, once the person
 * says it may. `onReserved` hears the reservation.
 *
 * The card brings it: the agent has no way to book a table until a card offering tables is on
 * the screen.
 */
export function reserveTable(
  tables: readonly DiningTable[],
  guest: string,
  onReserved: (reservation: Reservation) => void,
): Tools {
  return new Tools().register({
    name: RESERVE_TABLE,
    description:
      "Reserves a table from the seat selector on the person's screen. Call it when they chose a " +
      "table, such as by tapping one, with that table's id.",
    parameters: z.object({
      table_id: z.string().describe("the id of the table, as the seat selector shows it, such as T4"),
      when: z.string().describe("the day and time of the booking as they would read it, such as tomorrow at 8pm"),
      summary: z
        .string()
        .describe("the table, the party and the time as they would read it, such as Terrace, for 4, tomorrow at 8pm"),
    }),
    displayTitle: "Reserving your table",
    approval: {
      title: "Reserve this table?",
      message: "Juno books it under your name.",
      reasonArgument: "summary",
      allowTitle: "Reserve",
      declineTitle: "Not now",
    },
    run: ({ table_id, when }) => {
      const table = tables.find((each) => each.id === table_id && each.available);
      if (!table) {
        const free = tables.filter((each) => each.available).map((each) => each.id).join(", ");
        throw new Error(`${table_id} is not a free table on the seat selector. The free ones are ${free}.`);
      }
      if (!when.trim()) {
        throw new Error("Say when the table is for.");
      }
      const reservation: Reservation = {
        table,
        when,
        code: crypto.randomUUID().slice(0, 6).toUpperCase(),
        tables: tables.map((each) => each.id),
      };
      onReserved(reservation);
      return (
        `Reserved table ${table.id} (${table.zone}, for ${table.seats}) for ${when}, under ${guest}. ` +
        `The confirmation code is ${reservation.code}. The seat selector on their screen now shows ` +
        "this booking, so confirm it in a sentence and do not show or patch the seat selector again."
      );
    },
  });
}
