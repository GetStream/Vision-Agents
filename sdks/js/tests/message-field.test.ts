import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { messageField } from "../src/message-field.js";

describe("messageField", () => {
  it("reads a field top level, as stream-chat 9 has it", () => {
    assert.equal(messageField({ reasoning: 1 }, "reasoning"), 1);
  });

  it("reads a field under custom, as stream-chat 10 has it", () => {
    assert.equal(messageField({ custom: { reasoning: 1 } }, "reasoning"), 1);
  });

  it("prefers the top level when both have it", () => {
    assert.equal(messageField({ turn_id: "a", custom: { turn_id: "b" } }, "turn_id"), "a");
  });

  // stream-chat 9 keeps custom fields top level, so an app's own field named `custom` can be
  // anything. A string's own properties are not the message's.
  it("ignores a custom that is not an object", () => {
    assert.equal(messageField({ custom: "abc" }, "length"), undefined);
    assert.equal(messageField({ custom: null }, "x"), undefined);
    assert.equal(messageField({ custom: 7 }, "x"), undefined);
  });

  it("reads custom when the top level is null", () => {
    assert.equal(messageField({ turn_id: null, custom: { turn_id: "b" } }, "turn_id"), "b");
  });

  it("ignores an array custom", () => {
    assert.equal(messageField({ custom: ["a", "b"] }, "length"), undefined);
  });

  it("reads only what the message holds, not what every object inherits", () => {
    assert.equal(messageField({}, "toString"), undefined);
    assert.equal(messageField({ custom: {} }, "constructor"), undefined);
  });

  it("is undefined when neither has it", () => {
    assert.equal(messageField({ id: "m1" }, "reasoning"), undefined);
  });
});
