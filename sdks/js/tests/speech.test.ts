import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it, mock } from "node:test";

import { DEFAULT_CHARS_PER_MS, SPEECH_GRACE_MS, Speech } from "../src/index.js";

// The frames are the session socket's, as `frameOf` in acceleration/internal/api/sessionws.go
// writes them.
describe("Speech", () => {
  let clock: number;
  let voice: boolean;
  let speech: Speech;

  function speaking(turnId: string, synthesisId: string, text: string): void {
    speech.observe({ type: "speaking", turn_id: turnId, synthesis_id: synthesisId, text, at: "2026-10-22T09:30:00Z" });
  }

  function spoke(turnId: string, synthesisId: string, characters: number, durationMs: number): void {
    speech.observe({
      type: "spoke",
      turn_id: turnId,
      synthesis_id: synthesisId,
      characters,
      audio_duration_ms: durationMs,
      time_to_first_byte_ms: 120,
    });
  }

  /** The reply's text beginning, which on a call comes before its voice. */
  function writing(turnId: string, text = "Hello"): void {
    speech.observe({ type: "response_delta", turn_id: turnId, text });
  }

  /** Moves the store's clock and its timers on together. */
  function wait(ms: number): void {
    clock += ms;
    mock.timers.tick(ms);
  }

  beforeEach(() => {
    mock.timers.enable({ apis: ["setTimeout"] });
    clock = 1000;
    voice = true;
    speech = new Speech({ clock: () => clock, voice: () => voice });
  });

  afterEach(() => {
    speech.close();
    mock.timers.reset();
  });

  it("knows nothing of a turn the voice has not reached", () => {
    assert.equal(speech.get(), undefined);
    assert.equal(speech.heard("turn-1"), undefined);
    assert.equal(speech.heard(undefined), undefined, "a message with no turn has no speech");
    assert.equal(speech.heard(42), undefined);
    assert.equal(speech.heard({ id: "m1", text: "Hi" }), undefined);
  });

  it("reads the turn off a Chat message, as stream-chat 9 and 10 each have it", () => {
    speaking("turn-1", "turn-1", "Hello");
    clock += 1000;

    assert.deepEqual(speech.heard({ id: "m1", turn_id: "turn-1" }), { chars: 15, done: false });
    assert.deepEqual(speech.heard({ id: "m1", custom: { turn_id: "turn-1" } }), { chars: 15, done: false });
  });

  it("estimates a reply being heard at about 15 characters a second until it is measured", () => {
    speaking("turn-1", "turn-1", "Hello");
    clock += 1000;

    assert.deepEqual(speech.heard("turn-1"), { chars: 15, done: false });
    assert.equal(speech.get()?.charsPerMs, DEFAULT_CHARS_PER_MS);
  });

  it("paces an utterance by its own length once spoke says it", () => {
    speaking("say-1", "say-1", "Hi, thanks for calling Acme. How can I help you today, Jim?");
    spoke("say-1", "say-1", 60, 4000);

    assert.deepEqual(speech.heard("say-1"), { chars: 0, done: false });
    assert.deepEqual(speech.heard("say-1", clock + 2000), { chars: 30, done: false });
    assert.deepEqual(speech.heard("say-1", clock + 4000), { chars: 60, done: true });
    assert.equal(speech.get()?.charsPerMs, 60 / 4000, "the measured rate replaces the guess");
  });

  it("slows down rather than jumping back when the estimate ran ahead", () => {
    speaking("turn-1", "turn-1", "So");
    clock += 1000; // 15 estimated
    spoke("turn-1", "turn-1", 20, 2000); // 10 really heard by now

    assert.equal(speech.heard("turn-1")?.chars, 15);
    assert.equal(speech.heard("turn-1", clock + 500)?.chars, 17);
    assert.deepEqual(speech.heard("turn-1", clock + 1000), { chars: 20, done: true }, "and ends with the audio");
  });

  it("adds up the sentences of a turn a voice says one at a time", () => {
    speaking("turn-1", "turn-1#0", "One two three.");
    spoke("turn-1", "turn-1#0", 14, 1000);
    clock += 1000;
    speaking("turn-1", "turn-1#1", "Four.");

    assert.deepEqual(speech.heard("turn-1"), { chars: 14, done: false });
    clock += 200;
    assert.equal(speech.heard("turn-1")?.chars, 16, "the second at the first one's rate");
    clock += 10_000;
    assert.equal(speech.heard("turn-1")?.chars, 19, "a sentence is no longer than its text");
    assert.equal(speech.heard("turn-1")?.done, false, "until it is measured, it may still be playing");
  });

  it("plays a sentence reported early after the one still playing", () => {
    // A short sentence finished being made while the one before it plays, and the backend
    // reports it heard then rather than when the track reaches it.
    speaking("turn-1", "turn-1#0", "One two three.");
    spoke("turn-1", "turn-1#0", 14, 1000);
    clock += 300;
    speaking("turn-1", "turn-1#1", "Four.");
    spoke("turn-1", "turn-1#1", 5, 500);

    assert.equal(speech.heard("turn-1")?.chars, 4, "still in the first");
    assert.equal(speech.get()?.startedAt, 2000, "the second starts as the first ends");
    assert.deepEqual(speech.heard("turn-1", 2250), { chars: 16, done: false });
    assert.deepEqual(speech.heard("turn-1", 2500), { chars: 19, done: true });
  });

  it("freezes a turn the caller talked over at what was heard", () => {
    speaking("turn-1", "turn-1", "Tomorrow will be sunny and warm.");
    spoke("turn-1", "turn-1", 32, 2000);
    clock += 500;
    speech.observe({ type: "interrupted", turn_id: "turn-1" });

    assert.deepEqual(speech.heard("turn-1"), { chars: 8, done: true });
    assert.deepEqual(speech.heard("turn-1", clock + 5000), { chars: 8, done: true });
    assert.equal(speech.get()?.interrupted, true);

    speaking("turn-1", "turn-1#9", "late");
    assert.equal(speech.get()?.synthesisId, "turn-1", "nothing more of it is heard");
  });

  it("starts the next turn when it is heard, not after one that was cut off", () => {
    speaking("turn-1", "turn-1", "Tomorrow will be sunny and warm.");
    spoke("turn-1", "turn-1", 32, 2000);
    clock += 500;
    speech.observe({ type: "interrupted", turn_id: "turn-1" });
    speaking("turn-2", "turn-2", "Sorry, go on.");

    assert.equal(speech.get()?.turnId, "turn-2");
    assert.equal(speech.get()?.startedAt, clock);
    assert.equal(speech.get()?.interrupted, false);
  });

  it("tells listeners when the voice moves on, and stops when asked", () => {
    let told = 0;
    const stop = speech.subscribe(() => told++);

    speaking("turn-1", "turn-1", "Hello");
    const first = speech.get();
    spoke("turn-1", "turn-1", 5, 400);
    assert.equal(told, 2);
    assert.notEqual(speech.get(), first, "a change is a new snapshot");
    assert.equal(speech.get(), speech.get(), "and the same one until the next");

    stop();
    speech.observe({ type: "interrupted", turn_id: "turn-1" });
    assert.equal(told, 2);
  });

  it("ignores what is not speech, and a spoke for an utterance it never heard", () => {
    let told = 0;
    speech.subscribe(() => told++);

    speech.observe({ type: "heard", text: "Hi" });
    spoke("turn-1", "turn-1", 2, 100);
    speech.observe({ type: "interrupted", turn_id: "turn-1" });

    assert.equal(told, 0);
    assert.equal(speech.heard("turn-1"), undefined);
  });

  it("remembers only the last 20 turns", () => {
    for (let index = 0; index < 25; index++) {
      speaking(`turn-${index}`, `turn-${index}`, "Hi.");
    }

    assert.equal(speech.heard("turn-4"), undefined);
    assert.notEqual(speech.heard("turn-5"), undefined);
    assert.notEqual(speech.heard("turn-24"), undefined);
  });

  it("holds a reply whose text came first at nothing heard, and paces it once its voice starts", () => {
    let told = 0;
    speech.subscribe(() => told++);

    writing("turn-1");
    writing("turn-1", " there.");
    assert.deepEqual(speech.heard("turn-1"), { chars: 0, done: false });
    assert.equal(told, 1, "once, as it starts being held");

    wait(800);
    assert.deepEqual(speech.heard("turn-1"), { chars: 0, done: false });
    speaking("turn-1", "turn-1", "Hello there.");
    assert.equal(told, 2);
    wait(1000);
    assert.deepEqual(speech.heard("turn-1"), { chars: 15, done: false }, "from when it was heard");

    wait(SPEECH_GRACE_MS);
    assert.equal(told, 2, "a reply being heard is not let go of");
  });

  it("holds a line said as written, and one answered with words", () => {
    speech.observe({ type: "said", turn_id: "say-1", text: "Hi there." });
    speech.observe({ type: "responded", turn_id: "back-1", text: "Are you still there?" });

    assert.deepEqual(speech.heard("say-1"), { chars: 0, done: false });
    assert.deepEqual(speech.heard("back-1"), { chars: 0, done: false });
  });

  it("lets a reply go when its voice has not started in time, and says so", () => {
    let told = 0;
    writing("turn-1");
    speech.subscribe(() => told++);

    assert.equal(speech.heard("turn-1", clock + SPEECH_GRACE_MS), undefined, "however late the timer is");
    wait(SPEECH_GRACE_MS - 1);
    assert.deepEqual(speech.heard("turn-1"), { chars: 0, done: false });
    wait(1);
    assert.equal(speech.heard("turn-1"), undefined, "a router with no speaking shows text as it is");
    assert.equal(told, 1);

    writing("turn-1", " more");
    assert.equal(speech.heard("turn-1"), undefined, "more of it is not held again");
    assert.equal(told, 1);

    speaking("turn-1", "turn-1", "Hello more");
    wait(1000);
    assert.equal(speech.heard("turn-1"), undefined, "a page showed it in full, so a slow voice does not take it back");
    speech.observe({ type: "interrupted", turn_id: "turn-1", synthesis_id: "turn-1", heard_ms: 300 });
    assert.equal(speech.heard("turn-1"), undefined);
  });

  it("queues a later turn behind one it let go of that is still playing", () => {
    writing("turn-1");
    wait(SPEECH_GRACE_MS);
    speaking("turn-1", "turn-1", "Hello more");
    spoke("turn-1", "turn-1", 10, 2000);
    speaking("turn-2", "turn-2", "Next.");

    assert.equal(speech.get()?.startedAt, clock + 2000);
  });

  it("starts an utterance reported before the track reached it when the speech ahead has played", () => {
    speech.observe({ type: "speaking", turn_id: "turn-1", synthesis_id: "turn-1#1", text: "Four.", delay_ms: 600 });

    assert.equal(speech.get()?.startedAt, clock + 600);
    assert.deepEqual(speech.heard("turn-1", clock + 500), { chars: 0, done: false });
    assert.equal(speech.heard("turn-1", clock + 800)?.chars, 3);
  });

  it("cuts a turn at what the router says was heard, not at its own estimate", () => {
    speaking("turn-1", "turn-1#0", "One two three.");
    spoke("turn-1", "turn-1#0", 14, 1000);
    wait(1000);
    speaking("turn-1", "turn-1#1", "Four five six seven.");
    wait(1500); // estimated well into the second sentence

    speech.observe({ type: "interrupted", turn_id: "turn-1", synthesis_id: "turn-1#1", heard_ms: 200 });

    // The first in full, and 200 ms of the second at the measured 14 a second.
    assert.deepEqual(speech.heard("turn-1"), { chars: 16, done: true });
  });

  it("cuts at its own estimate when the router does not say what was heard", () => {
    speaking("turn-1", "turn-1", "Tomorrow will be sunny and warm.");
    wait(400);

    speech.observe({ type: "interrupted", turn_id: "turn-1" });

    assert.deepEqual(speech.heard("turn-1"), { chars: 6, done: true });
  });

  it("tells listeners when any turn is cut, not only the one heard last", () => {
    speaking("turn-1", "turn-1", "First.");
    speaking("turn-2", "turn-2", "Second.");
    let told = 0;
    speech.subscribe(() => told++);

    speech.observe({ type: "interrupted", turn_id: "turn-1" });

    assert.equal(told, 1);
    assert.equal(speech.heard("turn-1")?.done, true);
  });

  it("counts characters as code points, as the socket does", () => {
    speaking("turn-1", "turn-1#0", "Hi 👋🏽 café");
    wait(60_000);

    assert.equal("Hi 👋🏽 café".length, 12);
    assert.equal(speech.heard("turn-1")?.chars, 10, "a sentence is as long as its code points");
  });

  it("holds a message of the agent's still being written that the socket has said nothing of", async () => {
    let told = 0;
    speech.subscribe(() => told++);
    const live = { id: "m1", text: "Hello", turn_id: "turn-1", generating: true, ai_generated: true };

    assert.deepEqual(speech.heard(live), { chars: 0, done: false }, "Chat can be faster than the socket");
    assert.equal(told, 0, "not from inside heard");
    await Promise.resolve();
    assert.equal(told, 1);

    const v10 = { id: "m2", text: "Hi", custom: { turn_id: "turn-2", generating: true, source: "agent" } };
    assert.deepEqual(speech.heard(v10), { chars: 0, done: false });
    wait(SPEECH_GRACE_MS);
    assert.equal(speech.heard(live), undefined, "on the same grace");
  });

  it("does not hold a message that is finished, a person's, or in writing", () => {
    assert.equal(speech.heard({ id: "m1", text: "Hi", turn_id: "turn-1", ai_generated: true }), undefined);
    assert.equal(speech.heard({ id: "m2", text: "Hi", turn_id: "turn-2", generating: true }), undefined);
    voice = false;
    assert.equal(speech.heard({ id: "m3", text: "Hi", turn_id: "turn-3", generating: true, ai_generated: true }), undefined);
  });

  it("holds nothing once the agent has left the call, though it had been heard", () => {
    voice = false;
    speaking("turn-1", "turn-1", "Bye.");
    writing("turn-2");
    let told = 0;
    speech.subscribe(() => told++);

    speech.observe({ type: "left", at: "2026-10-22T09:30:00Z" });

    assert.equal(speech.heard("turn-2"), undefined, "what was waiting is let go of");
    assert.equal(told, 1);
    voice = true;
    writing("turn-3");
    assert.equal(speech.heard("turn-3"), undefined, "a reply in writing after is shown as it is");

    speech.observe({ type: "joined", at: "2026-10-22T09:31:00Z" });
    writing("turn-4");
    assert.deepEqual(speech.heard("turn-4"), { chars: 0, done: false });
  });

  it("holds nothing once the voice is stopped", () => {
    writing("turn-1");

    speech.voiceChanged(false);
    writing("turn-2");

    assert.equal(speech.heard("turn-1"), undefined);
    assert.equal(speech.heard("turn-2"), undefined);
  });

  it("freezes a reply cut off before any of it was heard at nothing, done", () => {
    let told = 0;
    writing("turn-1");
    speech.subscribe(() => told++);

    speech.observe({ type: "interrupted", turn_id: "turn-1" });

    assert.deepEqual(speech.heard("turn-1"), { chars: 0, done: true });
    assert.equal(told, 1);
    wait(SPEECH_GRACE_MS);
    assert.deepEqual(speech.heard("turn-1"), { chars: 0, done: true });
    assert.equal(told, 1, "its timer went with it");
  });

  it("does not freeze a reply it had already let go of", () => {
    writing("turn-1");
    wait(SPEECH_GRACE_MS);

    speech.observe({ type: "interrupted", turn_id: "turn-1" });

    assert.equal(speech.heard("turn-1"), undefined);
  });

  it("holds nothing for a conversation in writing", () => {
    voice = false;
    let told = 0;
    speech.subscribe(() => told++);

    writing("turn-1");
    speech.observe({ type: "responded", turn_id: "turn-1", text: "Hello" });

    assert.equal(speech.heard("turn-1"), undefined);
    assert.equal(told, 0);
  });

  it("holds replies once it has heard the voice, though it was not told the agent is on the call", () => {
    voice = false;
    speaking("turn-1", "turn-1", "Hi.");

    writing("turn-2");

    assert.deepEqual(speech.heard("turn-2"), { chars: 0, done: false });
  });

  it("lets go of a turn that says nothing, once that is known", () => {
    let told = 0;
    speech.subscribe(() => told++);

    writing("turn-1");
    speech.observe({ type: "responded", turn_id: "turn-1", text: "", pending_work: true });
    writing("turn-2");
    speech.observe({ type: "turn", turn_id: "turn-2", audio_out_ms: 0, interrupted: false });
    writing("turn-3");
    speech.observe({ type: "turn", turn_id: "turn-3", audio_out_ms: 900, interrupted: false });

    assert.equal(speech.heard("turn-1"), undefined, "an empty reply");
    assert.equal(speech.heard("turn-2"), undefined, "a turn that published no audio");
    assert.deepEqual(speech.heard("turn-3"), { chars: 0, done: false }, "audio went out, so its speaking is coming");
    assert.equal(told, 5, "held three times, let go twice");
  });
});
