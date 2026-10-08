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

// On your own server, with STREAM_API_KEY and STREAM_API_SECRET in the environment.
const api = new Client();

// In a browser, with a token your backend minted for this user.
const api = new Client({ apiKey: "vak_live_…", token: () => fetch("/api/token").then((r) => r.text()) });
```

`url` falls back to `STREAM_ACCELERATION_URL`, then Stream's hosted router, so it is only
passed for a self-hosted or local one. `customerId`, `apiKey` and `apiSecret` fall back to
`STREAM_ACCELERATION_CUSTOMER_ID`, `STREAM_API_KEY` and `STREAM_API_SECRET`.

The hosted router is reached through Stream's authenticating proxy, which wants the
credential spelled its own way; that is on by default for it. A self-hosted deployment
behind the same proxy passes `authenticate: true`, or sets `STREAM_ACCELERATION_AUTHENTICATE`:

```ts
const api = new Client({ url: "https://agents.example.com", authenticate: true });
```

A router running locally with nothing in front of it is reached by customer id:

```ts
const api = new Client({ url: "http://localhost:8080", customerId: "examples" });
```

It is a switch rather than an extra header or two because the two spellings contradict each
other: a router reached directly needs `Stream-Auth-Type: server` before it admits a
backend, and that is the one thing the proxy refuses. Opt in rather than let the presence of
a credential decide, because a Stream key and secret are in the environment for plenty of
reasons that have nothing to do with this router.

A secret belongs on a server. A browser holding one could rewrite every agent in the app,
which is why the browser passes a token instead — and why the operations that configure an
agent answer a browser with a 403 rather than trusting it.

## Resources, typed from the spec

The API is grouped by resource, with the request and response types generated from
`acceleration/api/openapi.yaml`:

```ts
const simulation = await api.simulations.create({ name, config_id, scenario, assertion });
let run = await api.simulations.run(simulation.id);
run = await api.simulations.runs.get(run.id);
await api.memories.truncate("user-123");
```

A failure raises `RouterError`, carrying the status, the operation and what the router said
went wrong.

## A conversation somebody comes back to

A text conversation is kept in Stream Chat, so what was said outlives the session that
heard it.

```ts
const support = api.agent("support");
const session = await support.sessions.create({ agent_id: "ana-support" });
// Keep session.conversationId. It is the channel, and the way back to what was said.
await session.responses.create("Where is my order?");

// Later: ana's conversations still running, without paging through every one that ended.
const live = await support.sessions.query({ agentId: "ana-support", state: "live" });
```

The channel is the backend's to name: a first open takes no `conversation_id`, and every
later one takes the one the first was given. Come back as the same `agent_id` too; the
backend checks a conversation is reopened by whoever held it.

`session.close()` stops a conversation and keeps everything it recorded and remembered, so
a conversation in writing is usually left running. `session.delete()`, or
`sessions.delete(id)` without a handle, deletes it with its turns and what it taught
memory.

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
  if (message.sessionId) {
    await dispatch.answer(message);
    return;
  }
  const session = await dispatch.sessionFor(message, () => new Agent({ name: "John" }));
  await session.responses.create(message.text);
});

await dispatch.run();
```

`capacity` is a promise about what this process can answer: the router passes over a full
worker rather than queueing behind it. Several workers can wait at once, and the work is
shared between them. Each call and message is reported `done` to the router when its handler
returns, with the error when it throws, which is what frees the room it took.

A message usually arrives here only when no agent is running on its channel — one written to
an agent that is already running is answered by the router from that session, because that
agent is the one that knows what has been said. `sessionFor` keeps one conversation per
channel for the same reason. The exception is an agent whose agent.yaml says
`dispatch: {text: enabled}`: what its end users write comes here with the `sessionId` it was
written to and a `commandId`, unanswered, and `dispatch.answer(message)` has the model answer
it, with this worker's credential acting for the user who wrote it.

A worker can also host tools for every session opened under an agent id, including one a
browser opened, where the session's own process cannot reach what the tool needs:

```ts
const agent = client.agent("my-agent");
agent.tools.register({ name: "lookup", description: "Look up an order", run: lookup });
dispatch.host(agent, { toolTimeoutMs: 60_000 });
```

The router offers them to each session naming the agent and sends every call here. `run`
works with only hosted tools and no handler, and throws `HostingRefusedError` if the router
refuses them.

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
  agent.yaml            required: the name and what it runs on (llm, stt, tts, speed, harness, tags, ...)
  instructions.md
  guardrail.md
  skills/think.md
  knowledge/pricing.md
  knowledge/urls.yaml   pages to read, each optionally with refresh_hours
  simulations/lunch.yaml
  .agent_sync           written by sync: the fingerprint last synced and when
```

```ts
import { Agent } from "@stream-io/vision-agents";
import { loadFolder } from "@stream-io/vision-agents/node";

const agent = new Agent({ folder: await loadFolder("agents/jean") });
await agent.sync();
```

`sync` stores it as a config a session can then be created from by name, in one request
carrying the files, the pages and what `agent.yaml` declares. A key `agent.yaml` does not
know is refused. The request carries a fingerprint of everything in it and `.agent_sync`
records it, so syncing on every startup only reads the config back when nothing has changed.
A setting left out leaves whatever is stored — a model chosen in the dashboard survives a
sync that says nothing about it. The harness (`harness`, subagent, sandbox and skills) is
stored on the config this way too, never on a session: a session runs its config's.

Each file in `simulations/` is a list of `name`, `scenario` and `assertion`, with `mode`,
`variations`, `max_turns`, `caller_*`, `judge_target` and `tags` optional. With a
`simulations/` directory the config's simulations become exactly what it declares, so an
empty one deletes them; without one, the stored ones are left alone.

What is written in code wins over what the directory says, so a directory is a starting
point rather than an override.

`loadFolder` needs a filesystem, which is why it is the one thing in the `/node` entry
point rather than the main one. It hands `sync` the stamp to read and write, so `sync`
itself touches no filesystem.

## Going back, and branching off

```ts
const items = await session.responses.items.all();
await session.responses.rewind(items[2]); // carry on as though nothing after it was said
const branch = await session.fork({ response_id: items[2].response_id, title: "asked again" });
```

`rewind` takes a response, a response's id, or any item of one. The model forgets the later
turns, and they drop out of what `responses` reads back. A text conversation is kept in
Stream Chat unless it is incognito, and so cannot be rewound, since the channel would still
hold the later turns: fork it at the
response instead, which starts a new session carrying the history only that far. Both work
from a browser as well as a server.

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
own. A socket never says it is a backend: `Stream-Auth-Type: server` has no query
counterpart on purpose, and the proxy is only ever told `stream-auth-type=jwt`, which is how
it reads a user's token.

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
