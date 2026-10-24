import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { CallNotHeldError, ConfigurationError, ToolCall, Tools, render } from "../src/index.js";

describe("Tools", () => {
  const nothing = new AbortController().signal;

  it("declares a function the way the session spec takes it", () => {
    const tools = new Tools().register({
      name: "get_weather",
      description: "Get the weather somewhere",
      parameters: { type: "object", properties: { city: { type: "string" } } },
      run: () => "sunny",
    });

    assert.deepEqual(tools.declared(), [
      {
        name: "get_weather",
        description: "Get the weather somewhere",
        parameters: { type: "object", properties: { city: { type: "string" } } },
      },
    ]);
  });

  it("declares who runs it, what a call is doing and what the person is asked", () => {
    const tools = new Tools().register({
      name: "issue_refund",
      description: "Refund an order",
      executor: "client",
      displayTitle: "Refunding your order",
      approval: { title: "Refund order A-1042?", reasonArgument: "reason", allowTitle: "Refund" },
      run: () => "refunded",
    });

    assert.deepEqual(tools.declared(), [
      {
        name: "issue_refund",
        description: "Refund an order",
        executor: "client",
        display_title: "Refunding your order",
        approval: { title: "Refund order A-1042?", reason_argument: "reason", allow_title: "Refund" },
      },
    ]);
  });

  it("leaves out parameters for a function that takes none", () => {
    const tools = new Tools().register({
      name: "now",
      description: "The time",
      run: () => "noon",
    });

    assert.deepEqual(tools.declared(), [{ name: "now", description: "The time" }]);
  });

  it("keeps the order they were registered in", () => {
    const tools = new Tools()
      .register({ name: "one", description: "First", run: () => "" })
      .register({ name: "two", description: "Second", run: () => "" });

    assert.deepEqual(
      tools.declared().map((tool) => tool.name),
      ["one", "two"],
    );
  });

  it("refuses a function with no description, since it is all the model chooses by", () => {
    assert.throws(
      () => new Tools().register({ name: "x", description: "", run: () => "" }),
      ConfigurationError,
    );
  });

  it("refuses the same name twice rather than letting one win quietly", () => {
    const tools = new Tools().register({ name: "x", description: "One", run: () => "" });

    assert.throws(
      () => tools.register({ name: "x", description: "Another", run: () => "" }),
      ConfigurationError,
    );
  });

  it("hands the model's arguments to the function", async () => {
    const tools = new Tools().register<{ city: string }>({
      name: "get_weather",
      description: "Get the weather somewhere",
      run: (input) => `it is sunny in ${input.city}`,
    });

    assert.equal(
      await tools.call("get_weather", JSON.stringify({ city: "Boulder" }), nothing),
      "it is sunny in Boulder",
    );
  });

  it("treats no arguments as none, since a model often sends nothing at all", async () => {
    const tools = new Tools().register({
      name: "now",
      description: "The time",
      run: (input) => JSON.stringify(input),
    });

    assert.equal(await tools.call("now", "", nothing), "{}");
  });

  it("reports arguments it cannot read rather than calling the function with them", async () => {
    let ran = false;
    const tools = new Tools().register({
      name: "x",
      description: "One",
      run: () => {
        ran = true;
        return "";
      },
    });

    await assert.rejects(() => tools.call("x", "{not json", nothing), ConfigurationError);
    assert.equal(ran, false);
  });

  it("reports a name nothing is registered as", async () => {
    await assert.rejects(() => new Tools().call("x", "{}", nothing), ConfigurationError);
  });

  it("waits for a function that returns a promise", async () => {
    const tools = new Tools().register({
      name: "slow",
      description: "Take a moment",
      run: async () => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return "done";
      },
    });

    assert.equal(await tools.call("slow", "{}", nothing), "done");
  });
});

describe("render", () => {
  it("uses a string as it is, since that is already words the model can read", () => {
    assert.equal(render("sunny"), "sunny");
  });

  it("turns everything else into JSON", () => {
    assert.equal(render({ sky: "clear" }), '{"sky":"clear"}');
    assert.equal(render([1, 2]), "[1,2]");
    assert.equal(render(3), "3");
  });

  it("renders a function that returned nothing as nothing", () => {
    assert.equal(render(undefined), "");
  });

  it("falls back to the value's own words when it cannot be JSON", () => {
    const looping: Record<string, unknown> = {};
    looping["self"] = looping;

    assert.equal(render(looping), "[object Object]");
  });

  it("declares a call's policies, and leaves the router's defaults unsaid", () => {
    const tools = new Tools()
      .register({
        name: "book_flight",
        description: "Book a flight",
        parameters: { type: "object", properties: { flight_id: { type: "string" } } },
        approval: { title: "Book flight {flight_id}?" },
        interruptible: true,
        timeoutMs: 5000,
        run: () => "booked",
      })
      .register({ name: "lookup", description: "Look it up", run: () => "found" });

    const [booking, lookup] = tools.declared();
    assert.equal(booking?.interruptible, true);
    assert.equal(booking?.timeout_ms, 5000);
    assert.deepEqual(booking?.approval, { title: "Book flight {flight_id}?" });
    for (const policy of ["interruptible", "timeout_ms", "approval"]) {
      assert.equal(policy in (lookup ?? {}), false, policy);
    }
  });

  it("refuses an approval that names an argument the function does not take", () => {
    assert.throws(
      () =>
        new Tools().register({
          name: "book_flight",
          description: "Book a flight",
          parameters: { type: "object", properties: { flight_id: { type: "string" } } },
          approval: { title: "Book {flight}?" },
          run: () => "booked",
        }),
      (error: unknown) => error instanceof ConfigurationError && /\{flight\}/.test(error.message),
    );
  });

  it("refuses a timeout that is not a duration", () => {
    assert.throws(
      () => new Tools().register({ name: "slow", description: "Slow", timeoutMs: -1, run: () => "" }),
      ConfigurationError,
    );
  });

  it("fails and aborts a call that takes longer than its timeout", async () => {
    let aborted: unknown;
    const tools = new Tools().register({
      name: "stuck",
      description: "Never finishes",
      timeoutMs: 20,
      run: (_input, signal) =>
        new Promise(() => {
          signal.addEventListener("abort", () => (aborted = signal.reason));
        }),
    });

    await assert.rejects(() => tools.call("stuck", "{}", new AbortController().signal), /stuck took longer than 20 ms/);
    assert.match(String(aborted), /took longer/);
  });

  it("hands a function the call it runs for, which does nothing without a line", async () => {
    let seen: ToolCall | undefined;
    const tools = new Tools().register({
      name: "whoami",
      description: "Who asked",
      run: async (_input, _signal, call) => {
        seen = call;
        call.update("checking");
        call.say("One moment.");
        return (await call.disallowInterruptions()).aborted ? "stopped" : "kept";
      },
    });

    assert.equal(await tools.call("whoami", "{}", new AbortController().signal), "kept");
    assert.equal(seen?.id, "");
  });

  it("reports how a call is going on its line, and a call stopped before it was held", async () => {
    const sent: [string, string, boolean][] = [];
    const line = (holds: boolean) => ({
      update: (id: string, text: string, verbatim: boolean) => void sent.push([id, text, verbatim]),
      hold: async () => holds,
    });
    const call = new ToolCall({ id: "call_1" }, line(true));

    call.update("checking");
    call.say("One moment.");
    const kept = await call.disallowInterruptions();
    const stopped = await new ToolCall({ id: "call_2" }, line(false)).disallowInterruptions();

    assert.deepEqual(sent, [
      ["call_1", "checking", false],
      ["call_1", "One moment.", true],
    ]);
    assert.equal(kept.aborted, false);
    assert.equal(stopped.aborted, true);
    assert.ok(stopped.reason instanceof CallNotHeldError);
  });
});
