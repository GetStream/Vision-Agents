import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";

import { LiveReasoning } from "../src/index.js";

// The windows of thinking the router puts on a reply's live updates, as `reasoningWindow` in
// `acceleration/internal/conversation/reasoning.go` writes them.
describe("LiveReasoning", () => {
  let reasoning: LiveReasoning;

  function window(text: string, offset: number, id = "r1"): void {
    reasoning.read({
      id: "m1",
      text: "ignored",
      reasoning: { id, offset, text, length: offset + text.length },
    } as { reasoning?: unknown });
  }

  beforeEach(() => {
    reasoning = new LiveReasoning();
  });

  it("puts windows in order together into the whole thinking", () => {
    window("The order was ", 0);
    window("delivered unopened.", 14);

    assert.equal(reasoning.get("r1"), "The order was delivered unopened.");
  });

  it("adds only what is new from a repeat of recent thinking", () => {
    window("Checking the policy", 0);

    window("the policy, which allows 30 days", 9);
    window("policy", 13);

    assert.equal(reasoning.get("r1"), "Checking the policy, which allows 30 days");
  });

  it("leaves a paragraph break for a missed window rather than running on", () => {
    window("First thought.", 0);

    window("Much later.", 40);

    assert.equal(reasoning.get("r1"), "First thought.\n\nMuch later.");
  });

  it("starts somebody joining midway from the window they get", () => {
    window("the rest of it", 300, "r2");

    assert.equal(reasoning.get("r2"), "the rest of it");
    assert.equal(reasoning.get("r1"), undefined);
  });

  it("counts positions in code points, as the router does", () => {
    window("café ", 0);

    window(" 🌸 bloom", 4);

    assert.equal(reasoning.get("r1"), "café 🌸 bloom");
  });

  it("changes nothing for a message with no window", () => {
    reasoning.read({});

    assert.equal(reasoning.get("r1"), undefined);
  });
});
