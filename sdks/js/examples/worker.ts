/**
 * A worker that answers whatever the router sends it: phone calls, and messages written to
 * an agent that is not running.
 *
 * ```bash
 * npm run example build/examples/worker.js
 * ```
 *
 * Nothing here has to be reachable from the internet. The worker connects out to the
 * router and the router pushes work down that connection as it arrives. Stop it with
 * ctrl-c and the calls it is already holding are allowed to finish.
 */
import { Agent, Dispatch } from "@stream-io/vision-agents";

const dispatch = new Dispatch({ capacity: 4 });

const john = () =>
  new Agent({
    name: "John",
    instructions: "You answer the phone for a bike shop. Be brief and be warm.",
    pipeline: { llm: "llm-fast", stt: "en-low-latency", tts: "sonic_36" },
  });

dispatch.onCall(async (call) => {
  console.log(`answering ${call.callerNumber || "a caller"} on ${call.calledNumber}`);

  const session = await john().answer(call);
  await session.wait();

  console.log(`${call.callId} ended`);
});

dispatch.onMessage(async (message) => {
  console.log(`${message.userName || message.userId} wrote: ${message.text}`);

  const session = await dispatch.sessionFor(message, john);
  session.respond(message.text);
});

const stop = new AbortController();
process.on("SIGINT", () => stop.abort());
process.on("SIGTERM", () => stop.abort());

console.log("waiting for calls");
await dispatch.run(stop.signal);
console.log("stopped");
