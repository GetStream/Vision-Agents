import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { z } from "zod";

import { ConfigurationError, Tools } from "../src/index.js";

const nothing = new AbortController().signal;

const bookFlight = z.object({
  flight_id: z.string().regex(/^[A-Z]{2}\d{1,4}$/).describe("e.g. LH400"),
  seats: z.number().int().min(1).max(9),
  cabin: z.enum(["economy", "business", "first"]).default("economy"),
});

describe("A tool described by a Zod schema", () => {
  it("is declared to the model as the JSON Schema the Zod schema describes", () => {
    const tools = new Tools().register({
      name: "book_flight",
      description: "Book a flight for the caller",
      parameters: bookFlight,
      run: () => "booked",
    });

    assert.deepEqual(tools.declared()[0]!.parameters, {
      type: "object",
      properties: {
        flight_id: { type: "string", pattern: "^[A-Z]{2}\\d{1,4}$", description: "e.g. LH400" },
        seats: { type: "integer", minimum: 1, maximum: 9 },
        cabin: { type: "string", enum: ["economy", "business", "first"], default: "economy" },
      },
      required: ["flight_id", "seats"],
    });
  });

  it("is handed the arguments as the schema parses them, typed from it", async () => {
    let seen: { flight_id: string; seats: number; cabin: "economy" | "business" | "first" } | undefined;
    const tools = new Tools().register({
      name: "book_flight",
      description: "Book a flight for the caller",
      parameters: bookFlight,
      run: (booking) => {
        seen = booking;
        return `booked ${booking.seats} on ${booking.flight_id}`;
      },
    });

    const output = await tools.call("book_flight", '{"flight_id":"LH400","seats":2}', nothing);

    assert.equal(output, "booked 2 on LH400");
    assert.deepEqual(seen, { flight_id: "LH400", seats: 2, cabin: "economy" });
  });

  it("refuses arguments the schema does not take, naming each, without running", async () => {
    let ran = false;
    const tools = new Tools().register({
      name: "book_flight",
      description: "Book a flight for the caller",
      parameters: bookFlight,
      run: () => {
        ran = true;
        return "booked";
      },
    });

    await assert.rejects(
      tools.call("book_flight", '{"flight_id":"lh400","seats":12}', nothing),
      (error: Error) =>
        error.message.startsWith("book_flight was asked for with arguments it cannot take:") &&
        error.message.includes("flight_id:") &&
        error.message.includes("seats:"),
    );
    assert.equal(ran, false);
  });

  it("has its approval checked against the schema's fields", () => {
    const tools = new Tools();

    assert.throws(
      () =>
        tools.register({
          name: "book_flight",
          description: "Book a flight for the caller",
          parameters: bookFlight,
          approval: { title: "Book {flight} for you?" },
          run: () => "booked",
        }),
      ConfigurationError,
    );
    tools.register({
      name: "book_flight",
      description: "Book a flight for the caller",
      parameters: bookFlight,
      approval: { title: "Book {flight_id} for you?" },
      run: () => "booked",
    });
  });

  it("is refused when it does not describe an object", () => {
    assert.throws(
      () =>
        new Tools().register({
          name: "shout",
          description: "Shout a word",
          parameters: z.string(),
          run: () => "",
        }),
      /shout's parameters describe string, and a tool takes an object/,
    );
  });

  it("is refused when the schema cannot say what it is as JSON Schema", () => {
    const opaque = {
      "~standard": { version: 1, vendor: "other", validate: (value: unknown) => ({ value }) },
    };

    assert.throws(
      () =>
        new Tools().register({
          name: "opaque",
          description: "Takes something",
          parameters: opaque as never,
          run: () => "",
        }),
      /opaque's parameters are a Standard Schema with no JSON Schema/,
    );
  });
});
