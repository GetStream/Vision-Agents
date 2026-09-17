/**
 * An agent on a call you can join from a browser.
 *
 * ```bash
 * npm run example build/examples/voice.js
 * ```
 *
 * Needs a router to talk to — `go run ./cmd/router` in `acceleration/` — and
 * `STREAM_API_KEY` and `STREAM_API_SECRET` for the call itself.
 */
import { Agent } from "@stream-io/vision-agents";

const agent = new Agent({
  name: "John",
  instructions: "You are a friendly assistant. Keep your replies to a sentence or two.",
  pipeline: {
    llm: "llm-fast",
    stt: "en-low-latency",
    tts: "sonic_36",
    greeting: "Hey, what can I help you with?",
  },
});

agent.tools.register<{ city: string }>({
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: {
    type: "object",
    properties: { city: { type: "string", description: "A city name" } },
    required: ["city"],
  },
  run: ({ city }) => ({ city, sky: "clear", celsius: 21 }),
});

const session = await agent.join();
console.log(`open this to talk to the agent:\n${await agent.monitorURL(session)}\n`);

for await (const event of session.events()) {
  switch (event.kind) {
    case "heard":
      console.log(`caller: ${event.text}`);
      break;
    case "responded":
      console.log(`agent:  ${event.text}`);
      break;
    case "error":
      console.error(event.text);
      break;
    default:
      break;
  }
}
