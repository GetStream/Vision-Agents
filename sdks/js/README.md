# @stream-io/calypso

The JavaScript client for the Stream acceleration backend. One package for both ends: a
Node process that runs agents, and a browser that holds a conversation with one.

Nothing here does inference or touches media. The backend joins the call, hears the caller,
answers and speaks. What arrives here are the events saying so, and what stays here is
function calling — because the functions are here.

```bash
npm install @stream-io/calypso
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
import { Client } from "@stream-io/calypso";

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
went wrong. Branch on `code` (`not_configured`, `validation_failed`, ...) or `type`, not on the
message, and expect codes you do not know; quote `requestId` to support, since a 500 says only
"something went wrong".

## A conversation somebody comes back to

A text conversation is kept in Stream Chat, so what was said outlives the session that
heard it.

```ts
const support = api.agent("support");
const session = await support.sessions.create({ agent_id: "ana-support" });
// Keep session.id. It is the way back to what was said.
await session.responses.create("Where is my order?");

// Later: ana's conversations still running, without paging through every one that ended.
const live = await support.sessions.query({ agentId: "ana-support", state: "live" });

// Carry one on, running or ended, by the id of the session it was held in.
const again = await support.sessions.resume(session.id);
```

The channel is the backend's to name, so a conversation is carried on by its session id
rather than by its channel. A session id may also be one the caller picks, up to 64 of
`A-Za-z0-9_-`, passed as `id` when it is created.

`session.close()` stops a conversation and keeps everything it recorded and remembered, so
a conversation in writing is usually left running. `session.delete()`, or
`sessions.delete(id)` without a handle, deletes it with its turns and what it taught
memory.

## An agent

```ts
import { Agent } from "@stream-io/calypso";

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

`join` opens a session with voice on, and the backend joins the session's own call,
`agent:<session id>`. `chat` holds the same conversation in writing instead — the same
instructions, the same skills, the same knowledge, with nothing transcribed and nothing
spoken. `session.voice.stop()` carries a conversation on in writing and
`session.voice.start()` puts the agent back on the call. Instructions reach the backend
with `agent.sync()`: a session runs on the stored config's.

The model asks for a tool over the session socket, this process runs it, and the answer
goes back the same way. A tool that throws is reported to the model, because the model is
mid-sentence waiting for it and can only say something useful if it is told.

A tool can say how the backend treats its calls, and read the call it is running for:

```ts
agent.tools.register<{ flight_id: string }>({
  name: "book_flight",
  description: "Book a flight for the caller",
  parameters: { type: "object", properties: { flight_id: { type: "string" } }, required: ["flight_id"] },
  approval: { title: "Book flight {flight_id}?" },
  timeoutMs: 20_000,
  run: async ({ flight_id }, signal, call) => {
    call.update("checking seats");
    const writing = await call.disallowInterruptions();
    return bookings.book(flight_id, { user: call.user.id, key: call.id, signal: writing });
  },
});
```

- `interruptible: true` lets the caller talking over the agent cancel a call in flight. By
  default it keeps running and the model is told it is still going.
- `timeoutMs` is how long one call may take, in place of the session's. A call that takes
  longer is aborted and fails.
- `approval` is asked before each call runs. In writing the call waits in
  `session.approvals`. On a voice call the agent asks the caller out loud, with each `{name}`
  in the title filled in from the call's arguments, and runs the call on a yes within 15
  seconds, with `call.approved` set. A `{name}` that is not in `parameters` is refused when
  the tool is registered.

`call` is the call being answered: its `id`, which a call sent again keeps, so it is the
key to make a side effect happen once; `sessionId`, `turnId`, the `user` whose turn asked,
and the `conversation`. `update(text)` says how the call is going: a voice agent tells the
caller in its own words once nobody is talking. `say(text)` has it say `text` as written,
which a speech-to-speech model may paraphrase. `disallowInterruptions()` keeps the call going
from then on, for a write that must not stop halfway, and resolves with the signal to do it
on, which only the timeout and the socket ending abort. When the call had already been
stopped, that signal is already aborted with a `CallNotHeldError`. A call the router
addresses to another watcher of the session (`run: false`) is left for whoever runs it.

## Agent dispatch

A caller reached a Stream call over SIP, or somebody wrote in a channel, and the router
found out by webhook. The agent, though, runs in your process. So the worker connects out
and waits, and the router pushes work down the connection as it arrives — nothing you run
has to be publicly reachable.

```ts
import { Agent, Dispatch } from "@stream-io/calypso";

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
written to and a `requestId`, unanswered, and `dispatch.answer(message)` has the model answer
it, with this worker's credential acting for the user who wrote it.

A worker can also host tools for every session opened under an agent id, including one a
browser opened, where the session's own process cannot reach what the tool needs:

```ts
const agent = client.agent("my-agent");
agent.tools.register({ name: "lookup", description: "Look up an order", run: lookup });
dispatch.host(agent, { toolTimeoutMs: 60_000 });
```

The router offers them to each session naming the agent and sends every call here. A hosted
tool takes the same `interruptible`, `timeoutMs` and `approval` (asked only on a voice
call), is handed its `call`, and is aborted when the router gives up on it. `run`
works with only hosted tools and no handler, and throws `HostingRefusedError` if the router
refuses them.

Server side only. A worker is offered other people's callers, so anything that can open
that socket could answer for the whole app.

Placing a call outward is the other direction of the same thing:

```ts
const session = await agent.startCall("+15551234567", "+15557654321");
```

The agent is told it is navigating, so recordings are let finish and menus are answered
rather than talked over. Every caller to an attached number gets a session of their own:
`agent.answer(call)` joins the one the router names, and `agent.waitForCall(number)` waits
for the next one without a worker of your own.

## An agent written down as a directory

```
agents/jean/
  agent.yaml            required: the name and what it runs on (llm, stt, tts, greeting, harness, tags, ...)
  instructions.md
  guardrail.md
  skills/think.md
  skills/pdf/SKILL.md   a skill as a folder: SKILL.md plus the text files it brings
  skills/pdf/scripts/fill.py
  knowledge/pricing.md
  knowledge/urls.yaml   pages to read, each optionally with refresh_hours
  simulations/lunch.yaml
  .agent_sync           written by sync: the fingerprint last synced and when
```

```ts
import { Agent } from "@stream-io/calypso";
import { loadFolder } from "@stream-io/calypso/node";

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

## A reply as it is heard

The Chat channel has the agent's reply as fast as the model writes it, seconds ahead of the
voice. `session.speech` says how much of each reply has been heard on the call, so a page can
show a message as it is said:

```ts
import type { LocalMessage } from "stream-chat";

function shownText(message: LocalMessage, now = performance.now()) {
  const text = message.text ?? "";
  const heard = session.speech.heard(message, now);
  // Characters are Unicode code points, as the session socket counts them.
  return heard ? Array.from(text).slice(0, heard.chars).join("") : text;
}

session.speech.subscribe(() => requestAnimationFrame(render));
```

Every message the agent said carries the `turn_id` the session socket's `speaking`, `spoke`
and `interrupted` name (on `message.turn_id` in stream-chat 9, `message.custom.turn_id` in
10). `heard` takes the message and reads either, or the turn id itself, and with `now` is
`{ chars, done }`: the code points of the message heard so far, exact once the playing
utterance's `spoke` has said how long it is and estimated from the voice's measured rate
(about 15 a second until then), and `done` once the turn's last known utterance has played out
or the caller talked over it. The voice is not sent the message's own words (a direction it
acts, "last four digits, 1234" for "last four 1234", a sentence held back), so the router says
where each utterance is in the message (`message_start` and `message_end` on `speaking`,
`spoke` and `interrupted`), and each is revealed across its own place there; a router that
does not say is counted in the voice's characters. A cut freezes the turn at what the router says was heard of the utterance it cut
(`heard_ms` on `interrupted`), or at the estimate when it does not say. An utterance reported
before the track got to it, behind speech still queued, starts when that has played
(`delay_ms` on `speaking`). A voice that takes a sentence at a time can still start another
after a pause, so show a reply in full once `done` is true and its message is no longer
`generating`.

A reply's message usually arrives before its audio starts, and Chat can even be faster than the
session socket. While the agent is on the call, a turn is `{ chars: 0, done: false }` until its
`speaking` once it is known to be coming: its text has begun on the socket (`said`,
`response_delta`, or a `responded` with words), or `heard` was asked about a message of the
agent's (`source: "agent"` or `ai_generated`) that is still `generating`. It is let go of, and
`undefined` from then on, once it turns out to say nothing (an empty `responded`, or a `turn`
that published no audio) or when `SPEECH_GRACE_MS` (5 s) pass without a `speaking`; speech that
comes later does not take it back, since the page has shown it in full. The agent leaving the
call (`left`), `voice.stop()` and the session ending stop anything being held, so a reply in
writing afterwards shows at once. A turn let go of, a message with no turn and a router too
old to send `speaking` are all `undefined`: show those as they are. A reply cut off before any
of it was heard stays at `{ chars: 0, done: true }`.

`now` is on the clock of `performance.now()`, which `requestAnimationFrame` hands its callback
too, and `heard` is cheap enough to call on every frame. `subscribe` is told whenever what
`heard` says of a turn changes other than with the clock; `get` says where the voice last was.
The store is fed from the frames the session reads, beside `events()` rather than through it,
so it needs no loop and takes nothing from yours. It keeps the last 20 turns.

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
import { Socket } from "@stream-io/calypso";

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
npm run test:integration               # device posture and one Stream connection, see below
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

`test:integration` checks the two promises a client SDK makes. A device holding a user's token
is refused every server-side operation (minting a call or chat token, claiming a guest,
configs, sync, simulations, memories, the dispatch socket) with a 403 `server_side_only`. And
however many sessions ask, there is one Stream Chat and one Stream Video socket for the user,
none at all beside a `StreamChat` the app handed over with `client.use`, nobody left online
after `client.disconnect()`, and no push device but the app's own. Sockets are counted off
Node's diagnostics channels, so they are the connections that actually left the process.

The posture half needs a router that verifies credentials, which the Docker one in proxy mode
does not. Mint a key against its Postgres and run a second router in `api_key` mode beside it:

```bash
cd acceleration && set -a && source ../.env && set +a
export ROUTER_POSTGRES_DSN='postgres://postgres:postgres@localhost:55432/model_router?sslmode=disable' ROUTER_REDIS_ADDR=localhost:56379
go run ./cmd/router keys create --app-name sdk-suite      # prints the key and secret once
ROUTER_ADDR=:8099 ROUTER_AUTH_MODE=api_key ROUTER_PUBLIC_URL=http://localhost:8099 go run ./cmd/router
```

Then `INTEGRATION_ROUTER_KEY=vak_live_… INTEGRATION_ROUTER_SECRET=vas_live_… npm run
test:integration` (`INTEGRATION_ROUTER_URL` defaults to `http://localhost:8099`). The Stream
half wants the Docker router on `STREAM_ACCELERATION_URL` with `STREAM_ACCELERATION_CUSTOMER_ID`,
`STREAM_API_KEY` and `STREAM_API_SECRET`, all read from `.env`. Each half skips without its own.
Run it from a shell without `.env` sourced, or `npm test` reads the same variables and fails.

They hold real conversations, so they spend the app's daily token allowance, and run often
enough they will spend all of it. A test that finds it gone skips saying so rather than
reporting the deployment broken — worth recognising in the output, because everything else
still passes and only the one test that needed an answer steps aside.

### Releasing

The **Release JS SDK** workflow publishes to npm, run by hand from the `accelerate` branch.
semantic-release (`release.config.mjs`) reads the commits since the last `sdk-js-v*` tag and
counts only those scoped `sdk-js`: `feat(sdk-js):` moves the minor, `fix(sdk-js):` and
`perf(sdk-js):` the patch, and while the package is 0.x a breaking change moves the minor too.
A commit with any other scope, or none, ships nothing, whatever it touches. The workflow runs
as a dry run unless that box is unticked, printing the version and notes it would publish.

It publishes with npm trusted publishing, so there is no token, and npm only lets a package
trust a workflow once the package exists. The first version is therefore published by hand:

```bash
npm publish                               # 0.2.0, as package.json says
git tag sdk-js-v0.2.0 && git push origin sdk-js-v0.2.0
```

then on npmjs.com, under the package's settings, add GitHub Actions as a trusted publisher:
`GetStream/Vision-Agents`, workflow `release-sdk-js.yml`. From then on the version in
`package.json` is not maintained: semantic-release writes the next one into the package it
publishes and nothing is committed back.
