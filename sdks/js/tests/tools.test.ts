import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ConfigurationError, Tools, render } from "../src/index.js";

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
});
