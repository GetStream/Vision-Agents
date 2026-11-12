import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { CallNotHeldError, ConfigurationError, ToolCallContext, ToolOutput, Tools, render } from "../src/index.js";

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

  it("renders what a function shows as parts, and words alone as words", () => {
    assert.deepEqual(render(new ToolOutput({ images: [{ url: "https://example.com/a.png", detail: "low" }] })), [
      { type: "image_url", image_url: { url: "https://example.com/a.png", detail: "low" } },
    ]);
    assert.equal(render(new ToolOutput({ text: "nothing to see" })), "nothing to see");
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

  it("fails a call that takes longer than its timeout", async () => {
    const tools = new Tools().register({
      name: "stuck",
      description: "Never finishes",
      timeoutMs: 20,
      run: () => new Promise(() => undefined),
    });

    await assert.rejects(() => tools.call("stuck", "{}", new AbortController().signal), /stuck took longer than 20 ms/);
  });

  it("hands a function the call it runs for, which does nothing without a line", async () => {
    let seen: ToolCallContext | undefined;
    const tools = new Tools().register({
      name: "whoami",
      description: "Who asked",
      run: async (_input, toolCallContext) => {
        seen = toolCallContext;
        toolCallContext.update("checking");
        toolCallContext.say("One moment.");
        await toolCallContext.disallowInterruptions();
        return "kept";
      },
    });

    assert.equal(await tools.call("whoami", "{}", new AbortController().signal), "kept");
    assert.equal(seen?.id, "");
  });

  it("hands run the arguments, then the call it runs for", async () => {
    const tools = new Tools()
      .register<{ city: string }>({
        name: "simple",
        description: "Only the arguments",
        run: ({ city }) => city,
      })
      .register<{ city: string }>({
        name: "context",
        description: "The arguments, then the call",
        run: ({ city }, toolCallContext) => `${toolCallContext.id} ${city}`,
      });
    const call = new ToolCallContext({ id: "call-1" });

    assert.equal(await tools.call("simple", JSON.stringify({ city: "Rome" }), new AbortController().signal, call), "Rome");
    assert.equal(
      await tools.call("context", JSON.stringify({ city: "Rome" }), new AbortController().signal, call),
      "call-1 Rome",
    );
  });

  it("stops waiting on a call that is cancelled", async () => {
    const cancel = new AbortController();
    const tools = new Tools().register({
      name: "slow",
      description: "Never finishes",
      run: () => new Promise(() => undefined),
    });

    const answer = tools.call("slow", "{}", cancel.signal);
    cancel.abort();
    await assert.rejects(answer, /slow was cancelled/);
  });

  it("reports how a call is going on its line, and a call stopped before it was held", async () => {
    const sent: [string, string, boolean][] = [];
    const line = (holds: boolean) => ({
      update: (id: string, text: string, verbatim: boolean) => void sent.push([id, text, verbatim]),
      hold: async () => holds,
    });
    const toolCallContext = new ToolCallContext({ id: "call_1" }, line(true));

    toolCallContext.update("checking");
    toolCallContext.say("One moment.");
    await toolCallContext.disallowInterruptions();

    assert.deepEqual(sent, [
      ["call_1", "checking", false],
      ["call_1", "One moment.", true],
    ]);
    await assert.rejects(
      new ToolCallContext({ id: "call_2" }, line(false)).disallowInterruptions(),
      CallNotHeldError,
    );
  });
});

describe("Tools concurrency", () => {
  const nothing = new AbortController().signal;
  /** Lets every call waiting on a microtask, such as for a place, get as far as it can. */
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  const gate = () => {
    let open = (): void => undefined;
    const opened = new Promise<void>((resolve) => {
      open = resolve;
    });
    return { opened, open };
  };

  it("runs at most as many calls of a function at once as its concurrency", async () => {
    let running = 0;
    let most = 0;
    const tools = new Tools().register({
      name: "lookup",
      description: "Look it up",
      concurrency: 2,
      run: async () => {
        running += 1;
        most = Math.max(most, running);
        await new Promise((resolve) => setTimeout(resolve, 5));
        running -= 1;
        return "found";
      },
    });

    const answers = await Promise.all(Array.from({ length: 6 }, () => tools.call("lookup", "{}", nothing)));

    assert.deepEqual(answers, Array(6).fill("found"));
    assert.equal(most, 2);
  });

  it("runs at most 16 calls at once across all its functions", async () => {
    const { opened, open } = gate();
    let started = 0;
    const run = async () => {
      started += 1;
      await opened;
      return "done";
    };
    const tools = new Tools()
      .register({ name: "one", description: "First", run })
      .register({ name: "two", description: "Second", run });

    const answers = Promise.all(
      Array.from({ length: 20 }, (_, i) => tools.call(i % 2 ? "one" : "two", "{}", nothing)),
    );
    await settle();
    assert.equal(started, 16);

    open();
    assert.equal((await answers).length, 20);
    assert.equal(started, 20);
  });

  it("counts the wait for a place toward a call's timeout", { timeout: 2000 }, async () => {
    const { opened, open } = gate();
    let started = 0;
    const tools = new Tools().register({
      name: "book",
      description: "Book it",
      concurrency: 1,
      timeoutMs: 30,
      run: async () => {
        started += 1;
        await opened;
        return "booked";
      },
    });

    const first = tools.call("book", "{}", nothing);
    first.catch(() => undefined);
    await settle();
    await assert.rejects(tools.call("book", "{}", nothing), /book took longer than 30 ms/);
    assert.equal(started, 1);

    // The call that timed out waiting gave up its turn, so the place goes to the next call.
    open();
    await assert.rejects(first, /took longer/);
    assert.equal(await tools.call("book", "{}", nothing), "booked");
  });

  it("takes a cancelled call out of the queue without keeping a place", { timeout: 2000 }, async () => {
    const { opened, open } = gate();
    let started = 0;
    const tools = new Tools().register({
      name: "book",
      description: "Book it",
      concurrency: 1,
      run: async () => {
        started += 1;
        await opened;
        return "booked";
      },
    });

    const first = tools.call("book", "{}", nothing);
    await settle();
    const cancel = new AbortController();
    const waiting = tools.call("book", "{}", cancel.signal);
    await settle();
    cancel.abort();
    await assert.rejects(waiting, /book was cancelled/);

    open();
    assert.equal(await first, "booked");
    assert.equal(await tools.call("book", "{}", nothing), "booked");
    assert.equal(started, 2);
  });

  it("keeps the place of a cancelled call until its function returns", { timeout: 2000 }, async () => {
    const { opened, open } = gate();
    let running = 0;
    let most = 0;
    const tools = new Tools().register({
      name: "write",
      description: "Write it",
      concurrency: 1,
      run: async () => {
        running += 1;
        most = Math.max(most, running);
        await opened;
        running -= 1;
        return "written";
      },
    });

    const cancel = new AbortController();
    const first = tools.call("write", "{}", cancel.signal);
    await settle();
    cancel.abort();
    await assert.rejects(first, /write was cancelled/);
    const next = tools.call("write", "{}", nothing);
    await settle();
    assert.equal(running, 1);

    open();
    assert.equal(await next, "written");
    assert.equal(most, 1);
  });

  it("gives back the place of a call that failed", { timeout: 2000 }, async () => {
    let calls = 0;
    const tools = new Tools().register({
      name: "flaky",
      description: "Fails twice",
      concurrency: 1,
      run: async () => {
        calls += 1;
        if (calls === 1) {
          throw new Error("rejected");
        }
        return "ok";
      },
    });
    const throwing = new Tools().register({
      name: "throws",
      description: "Throws before it returns",
      concurrency: 1,
      run: () => {
        calls += 1;
        if (calls === 3) {
          throw new Error("thrown");
        }
        return "ok";
      },
    });

    await assert.rejects(tools.call("flaky", "{}", nothing), /rejected/);
    assert.equal(await tools.call("flaky", "{}", nothing), "ok");
    await assert.rejects(throwing.call("throws", "{}", nothing), /thrown/);
    assert.equal(await throwing.call("throws", "{}", nothing), "ok");
  });

  it("refuses a concurrency that is not a whole number of calls", () => {
    for (const concurrency of [0, -1, 1.5, Number.NaN]) {
      assert.throws(
        () => new Tools().register({ name: "x", description: "One", concurrency, run: () => "" }),
        ConfigurationError,
        String(concurrency),
      );
    }
  });

  it("keeps the concurrency to itself, since only this process enforces it", () => {
    const tools = new Tools().register({ name: "x", description: "One", concurrency: 2, run: () => "" });

    assert.deepEqual(tools.declared(), [{ name: "x", description: "One" }]);
  });
});
