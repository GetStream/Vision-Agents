# @stream-io/vision-agents

The JavaScript client for the Stream acceleration backend. One package for both ends: a
Node process that runs agents, and a browser that holds a conversation with one.

Nothing here does inference or touches media. The backend joins the call, hears the caller,
answers and speaks. What arrives here are the events saying so, and what stays here is
function calling — because the functions are here.

```bash
npm install @stream-io/vision-agents
```

Node 22 or newer, or any browser. No runtime dependencies: `fetch`, `WebSocket` and Web
Crypto are all taken from the runtime.

## The three ways to say who is calling

Which one a deployment uses is a property of the deployment rather than a choice.

| Credential | Who it is | Reaches |
| --- | --- | --- |
| `customerId` | A router with nothing in front of it, which is a laptop | everything |
| `apiKey` + `apiSecret` | A process you run | everything |
| `apiKey` + `token` | A browser | the conversations that user owns |

```ts
import { Client } from "@stream-io/vision-agents";

// On your own server.
const api = new Client({ apiKey: process.env.STREAM_API_KEY, apiSecret: process.env.STREAM_API_SECRET });

// In a browser, with a token your backend minted for this user.
const api = new Client({ apiKey: "vak_live_…", token: () => fetch("/api/token").then((r) => r.text()) });
```

`url` falls back to `STREAM_ACCELERATION_URL`, then `http://localhost:8080`. `customerId`,
`apiKey` and `apiSecret` fall back to `STREAM_ACCELERATION_CUSTOMER_ID`, `STREAM_API_KEY`
and `STREAM_API_SECRET`.

A hosted deployment is reached through Stream's authenticating proxy, which wants the
credential spelled its own way. Pass `authenticate: true`, or set
`STREAM_ACCELERATION_AUTHENTICATE`:

```ts
const api = new Client({
  url: "https://accelerate.gcp.stream-io-api.com",
  apiKey: process.env.STREAM_API_KEY,
  apiSecret: process.env.STREAM_API_SECRET,
  authenticate: true,
});
```

It is a switch rather than an extra header or two because the two spellings contradict each
other: a router reached directly needs `Stream-Auth-Type: server` before it admits a
backend, and that is the one thing the proxy refuses. Opt in rather than let the presence of
a credential decide, because a Stream key and secret are in the environment for plenty of
reasons that have nothing to do with this router.

A secret belongs on a server. A browser holding one could rewrite every agent in the app,
which is why the browser passes a token instead — and why the operations that configure an
agent answer a browser with a 403 rather than trusting it.

## Every endpoint, typed from the spec

`Client` has one method per HTTP method rather than one per endpoint. The paths, the
parameters, the body and the answer all come from `acceleration/api/openapi.yaml`, so the
spec is the API surface: all 93 operations are reachable, and a new one needs nothing
written here to be callable.

```ts
const configs = await api.get("/v1/agents/configs");
const calls = await api.get("/v1/agents/calls", { query: { limit: 20 } });
const answer = await api.post("/v1/search", { body: { query: "what changed in v3" } });
await api.delete("/v1/agents/sessions/{id}", { path: { id: "sess_1" } });
```

TypeScript will not let you `get` a path that only answers POST, misname a query
parameter, or forget a required body. A failure raises `RouterError`, carrying the status,
the operation and what the router said went wrong.

## A conversation somebody comes back to

A text conversation is kept in Stream Chat, so what was said outlives the session that
heard it. `conversation` returns the session holding one and opens one only if none is.

```ts
import { conversation } from "@stream-io/vision-agents";

const session = await conversation(api, { id, conversationId: stored, config_id: "cfg_1" });
// Keep session.conversation_id. It is the channel, and the way back to what was said.
```

`id` is your own name for the conversation, stable across the sessions that hold it, and
what a running one is found by — so two people's conversations must not share it. The
channel is the backend's to name: a first open takes none, and every later one takes the
`conversation_id` the first was given. It cannot be named up front, because naming one is a
resume, and a resume reads the channel without creating it — so a name nothing has been
held in yet is refused. Come back as the same `id` too; the backend checks a conversation
is reopened by whoever held it.

The one caller this cannot find a session for is an anonymous one going by no name, which
the backend tells about no sessions at all. Hold onto the session id and read it back with
`getSession` instead.

## An agent

```ts
import { Agent } from "@stream-io/vision-agents";

const agent = new Agent({
  name: "John",
  instructions: "You are a friendly assistant. Keep replies short.",
  pipeline: { llm: "llm-fast", stt: "en-low-latency", tts: "sonic_36" },
});

agent.tools.register<{ city: string }>({
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
  run: async ({ city }) => fetchWeather(city),
});

const session = await agent.join();
console.log(await agent.monitorURL(session));

for await (const event of session.events()) {
  console.log(event.kind, event.text);
}
```

`join` creates a Stream call and has the backend join it. `chat` holds the same
conversation in writing instead — the same instructions, the same skills, the same
knowledge, with nothing transcribed and nothing spoken.

The model asks for a tool over the session socket, this process runs it, and the answer
goes back the same way. A tool that throws is reported to the model, because the model is
mid-sentence waiting for it and can only say something useful if it is told.

## Agent dispatch

A caller reached a Stream call over SIP, or somebody wrote in a channel, and the router
found out by webhook. The agent, though, runs in your process. So the worker connects out
and waits, and the router pushes work down the connection as it arrives — nothing you run
has to be publicly reachable.

```ts
import { Agent, Dispatch } from "@stream-io/vision-agents";

const dispatch = new Dispatch({ capacity: 4 });

dispatch.onCall(async (call) => {
  const agent = new Agent({ name: "John", pipeline: { config: "support" } });
  const session = await agent.answer(call);
  await session.wait();
});

dispatch.onMessage(async (message) => {
  const session = await dispatch.sessionFor(message, () => new Agent({ name: "John" }));
  session.respond(message.text);
});

await dispatch.run();
```

`capacity` is a promise about what this process can answer: the router passes over a full
worker rather than queueing behind it. Several workers can wait at once, and the work is
shared between them.

A message only arrives here when no agent is running on its channel — one written to an
agent that is already running is answered by the router from that session, because that
agent is the one that knows what has been said. `sessionFor` keeps one conversation per
channel for the same reason.

Server side only. A worker is offered other people's callers, so anything that can open
that socket could answer for the whole app.

Placing a call outward is the other direction of the same thing:

```ts
const session = await agent.startCall("+15551234567", "+15557654321");
```

The agent is told it is navigating, so recordings are let finish and menus are answered
rather than talked over.

## An agent written down as a directory

```
agents/jean/
  instructions.md
  guardrail.md
  skills/think.md
  knowledge/pricing.md
  knowledge/urls.yaml
```

```ts
import { Agent } from "@stream-io/vision-agents";
import { loadFolder } from "@stream-io/vision-agents/node";

const agent = new Agent({ folder: await loadFolder("agents/jean") });
await agent.sync();
```

`sync` stores it as a config a session can then be created from by name. It carries a
fingerprint of everything in it, so syncing on every startup does nothing when nothing has
changed, and a setting left out leaves whatever is stored — a model chosen in the dashboard
survives a sync that says nothing about it.

What is written in code wins over what the directory says, so a directory is a starting
point rather than an override.

`loadFolder` needs a filesystem, which is why it is the one thing in the `/node` entry
point rather than the main one.

## Sockets

`Socket` is the hand-written half of this SDK. OpenAPI stops at the upgrade, so the session
events socket, the dispatch socket and the four modality sockets are written rather than
generated.

```ts
import { Socket } from "@stream-io/vision-agents";

const socket = await Socket.open(api.backend, await api.backend.socketURL("/v1/stt/stream"));
socket.send({ type: "start", options: { language: "en" } });
socket.sendAudio(pcm);

for await (const message of socket.messages()) {
  if (message instanceof Uint8Array) {
    play(message);
  } else {
    console.log(message.type);
  }
}
```

Credentials go in the query string, because a browser WebSocket carries no headers of its
own. `Stream-Auth-Type` has no query counterpart on purpose, which is why a socket opened
from a browser cannot claim to be a backend.

## Working on this package

```bash
npm install
npm run types                          # regenerate src/generated/api.ts from the spec
npm test                               # typecheck, then the suite
npm run test:local                     # against a router running on this machine
npm run test:qa                        # against the hosted deployments
npm run build                          # dist/, what gets published
npm run example build/examples/voice.js
```

The generated types are committed, so installing the package needs no code generation, and
`npm run types -- --check` is what CI uses to catch a spec that has moved without them.
`acceleration/api/openapi.yaml` is the source of truth: after editing it, regenerate every
client — see [acceleration/README.md](../../acceleration/README.md).

The tests run against a real HTTP server and a real WebSocket upgrade rather than a stubbed
`fetch`, so what they exercise is the request this SDK actually puts on the wire.

`npm test` is hermetic and is what CI runs. The two live suites are not part of it and are
not in CI: they talk to a deployment, and they skip rather than fail when there is none to
talk to — a laptop with no router up, or a checkout with no credentials, is not a broken
build. `test:local` wants a router on `LOCAL_ACCELERATION_URL`, default
`http://127.0.0.1:8098`. `test:qa` wants `STREAM_API_KEY` and `STREAM_API_SECRET`, and
reaches `STAGING_ACCELERATION_URL` and `STAGING_DOCS_SEARCH_URL`. They are the only place
the claims above are checked against a real backend rather than against a test one, which
is worth running before trusting any of them: three of them are `todo` there today, each
naming what the deployment does instead.

They hold real conversations, so they spend the app's daily token allowance, and run often
enough they will spend all of it. A test that finds it gone skips saying so rather than
reporting the deployment broken — worth recognising in the output, because everything else
still passes and only the one test that needed an answer steps aside.
