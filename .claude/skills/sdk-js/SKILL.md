---
name: sdk-js
description: How to build and extend the JavaScript SDK in sdks/js. Read this before changing the client, the socket, the agent or dispatch, or before reaching for a dependency.
---

# JavaScript SDK conventions

The per-language half of [sdk](../sdk/SKILL.md). It records the decisions
[`sdks/js`](../../../sdks/js) already follows, so a change lands consistently rather than
re-litigating them. Where a rule has an exception, the exception is written down.

Assume TypeScript 5.7 with `strict`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`,
`verbatimModuleSyntax` and `module: NodeNext`. ESM only. Node 22 or a browser.

## One package, both ends

Not `-core`/`-ui`/`-rtc` the way Swift splits. That split exists because `StreamWebRTC` is a
47 MB binary; there is no such weight here, and npm has no equivalent of SwiftPM's resolution
problem. One package with two entry points:

```
.        src/index.ts   everything that runs in a browser and on a server
./node   src/node.ts    loadFolder, which needs node:fs
```

`node:fs` is the *only* thing that earns a place in `./node`. `Dispatch` stays in the main
entry even though a worker is server-side, because it is server-side by *credential*, not by
runtime: it contains no Node API, and a browser opening that socket gets a 403 from the
router, which is the correct answer rather than a bundling error.

**What a caller may do is decided by what they authenticated as, never by which file they
imported.** Do not add a `server-only` guard, a `typeof window` check, or a second class for
the browser. `Backend.serverSide` exists to let a caller ask before a call instead of reading
a 403 afterwards; it is not an enforcement point, and the router is.

## Zero runtime dependencies

`dependencies` is empty and stays empty. `fetch`, `WebSocket`, `crypto.subtle`, `btoa`,
`TextEncoder`, `URL` and `AbortController` are all in both runtimes. This is why:

- **JWTs are signed with Web Crypto**, not `jsonwebtoken` or `jose`. HS256 is about fifteen
  lines (`signToken` in `backend.ts`). A JWT library here would be a dependency in every
  browser bundle that imported the package for something else.
- **`ws` is a dev dependency only**, for the test server. Node 22 has a global `WebSocket`.
- **`urls.yaml` is parsed by hand** (`parsePages`), not with `yaml`. It is a list of urls; the
  subset is a screenful. A key it does not recognise is refused rather than dropped, so the
  narrowness is visible rather than silent.
- **Stream Video calls are created with one `fetch`** (`edge.ts`), not with
  `@stream-io/node-sdk`. The generic `sdk` skill says to depend on Stream's server SDK; here
  the SDK needs exactly one endpoint (`POST /api/v2/video/call/{type}/{id}`) and a token it
  can already sign, and the dependency would land in browser bundles.

If a change needs a dependency, that is a design question, not an install. Ask first.

## The client is generated types plus a hand-written request layer

`openapi-typescript`, not `openapi-fetch`, not `oapi-codegen`'s TS equivalent, and **not a
generated client**. A generated client brings a runtime, and the runtime is the part that has
to decide how a token is minted and which global `fetch` to use — which is the part that
differs between a browser and a server. So: types only, no runtime emitted, output committed
so installing the package needs no code generation.

`npm run types` regenerates; `npm run types -- --check` is what CI runs. `--default-non-nullable
false` is deliberate: a field the spec gives a default is one the caller may leave out, and
generated non-nullable it would be required on the way in, so every session request would have
to spell out the defaults it wanted.

**One method per HTTP method, not one per endpoint.** The spec has 93 operations and the shapes
are already generated, so 93 wrappers would say nothing the types do not, and a new endpoint
would need one written before it could be called.

```ts
const configs = await api.get("/v1/agents/configs");
await api.delete("/v1/agents/sessions/{id}", { path: { id } });
```

The type machinery in `client.ts` is load-bearing and worth reading before editing:

- `PathsWith<M>` filters paths by method with a tuple guard, `[Operation<P,M>] extends [never]`.
  A bare `extends never` is always true, because `never` extends everything.
- `BodyOf` has three cases, not two. An optional body and no body at all are different:
  attaching a number takes one and does not need one, and a caller should be able to leave it
  out without being able to invent one where the spec has none.
- `QueryOf` widens query values with `| undefined`, because `exactOptionalPropertyTypes` would
  otherwise reject a value passed through from somewhere that may not have it. They are dropped
  from the query string rather than sent as the word.
- `Result` maps 204 to `void`, which is most of the ways a session is acted on.

Never add a method that names one endpoint. Never re-declare a schema by hand — name it
`Schemas["CreateSessionRequest"]`.

## Sockets

Hand-written, because OpenAPI stops at the upgrade. One `Socket` for all six: session events,
dispatch, and the four modality streams.

- **Frames are loosely typed on purpose.** `Frame` is `{ type?: string } & Record<string,
  unknown>`, read through `text`, `number`, `flag` and `nested`. A deployment that has learned
  a new event must reach a caller reading `frame.type` rather than be dropped here. The
  generated REST shapes are strict; these cannot be.
- **`messages()` is an `AsyncGenerator` with one reader.** Two `for await` loops over one
  socket take half the frames each. Say so in the doc comment of anything that exposes one.
- A text frame that is not JSON is skipped, not fatal. Ending the stream over it would lose
  everything said after it.
- `close()` is idempotent. `finish()` removes the listeners and resolves every waiting reader
  as done, so a `for await` ends rather than hanging.
- Credentials go in the query string, because a browser WebSocket carries no headers.
  **`Stream-Auth-Type` has no query counterpart on purpose** — that is what stops a socket
  opened from a browser claiming to be a backend. Do not add one.
- **No automatic reconnection.** `respond` and `tool_result` are not idempotent and the
  protocol has no sequence number to resume from. Replaying would duplicate turns and tool
  results. Adding reconnection means adding resume semantics to the router first.

## Answering a tool call may not depend on anybody reading events

`Session.watch()` starts in the constructor and runs whether or not the caller has begun a
loop. `tool_call` and `tool_cancel` are handled there and never reach `events()`: the model is
mid-sentence waiting, and a tool call that depended on the caller having started iterating
would hang on a session nobody is watching.

- Each call runs on its own `void this.runTool(frame)`, not awaited, because reading the socket
  is also what delivers `tool_cancel`.
- A tool that throws is answered with `{ error }`, never dropped. The model can only say
  something useful about a tool that did not work if it is told that it did not work.
- `tool_cancel` aborts that call's `AbortController`; ending the session aborts all of them.
- `events()` buffers 256 and then drops the oldest rather than stalling the socket. Dropping an
  event never drops a turn, because turns are answered by `watch`.

## nil means omit

A field left out of `AgentOptions` or `Pipeline` is left out of the request, so the config or
the router decides. **Never copy a schema default into the request.** A `backchannel: false`
sent because the type had a default is how a caller silently loses what their config named —
which is also why `--default-non-nullable false` is set.

Hence the conditional-spread style throughout `agent.ts`:

```ts
...(pipeline.llm ? { llm: pipeline.llm } : {}),
...(pipeline.backchannel === undefined ? {} : { backchannel: pipeline.backchannel }),
```

A boolean tests `=== undefined`, because `false` is a value somebody chose. A string tests
truthiness, because an empty target means the same as no target.

An absent skill list and an empty one differ: absent leaves the built-in set, `useSkills:
false` or `skills: []` sends `skills: []` and turns delegation off.

## A backend rule the caller cannot guess gets a function, not a doc note

`conversation.ts` is the pattern. Holding a text conversation has two rules that are nowhere in
the spec and both load-bearing: the channel is the backend's to name, so a first open passes no
`conversation_id` and a resume passes the one the first was given; and a resume has to come back
as the same `agent_id`, because the backend checks a conversation is reopened by whoever held it.
So they are a named function that owns them, validating up front (`ConfigurationError`, before a
request) rather than a paragraph every caller has to find. When a rule like this turns up, add
the function here instead of explaining it downstream.

Both rules were got wrong first time from reading the router, and only the live suite found it.
Naming the channel on a first open reads like the obvious thing — it makes `agent_id` and
`conversation_id` agree, which is what the message hook wants — and the backend refuses it
outright, because a resume reads the channel without creating it. What follows is that a session
a browser opened cannot be driven by writing into its channel: the hook looks for a session
whose `agent_id` is the channel, the backend named the channel something else, and the write
lands nowhere. Chat is the transcript; `respond` over the session socket is the way in.

## Check a claim about the backend against the backend

`tests/live/` is where a claim in this README stops being a reading of the router and starts
being true. The unit suite proves the SDK sends what it meant to; only a deployment proves that
was the right thing to send. Two suites, neither in CI, both skipping when there is nothing to
talk to: `local.test.ts` against a router on this machine, `staging.test.ts` against the hosted
deployments.

They earn it. The QA suite is what found that the hosted deployments want the credential spelled
Stream's own way — `api_key`, not `X-Api-Key`, and `stream-auth-type: jwt` whoever the token is
for — which is the `authenticate` switch in `backend.ts` and which the SDK simply could not do
before. It also found that an ambient `STREAM_API_SECRET` was quietly turning a browser-shaped
client into a backend.

Where a deployment does something other than what it should, the test says what it should and is
marked `todo` with the cause, so the suite stays green and the gap stays named. Three are today:
the proxy tells the router nothing about which end user is calling, so a proxied caller is read
as that app's backend — it reaches every session the app has and the operations that configure an
agent — and the proxy requires a header a browser WebSocket cannot set, so a page cannot watch a
session on a hosted deployment at all. Do not write a test that asserts the broken behaviour; it
would have to be found and undone by whoever fixes it.

## Agent

`Agent` is configuration plus function calling. The conversation runs in the backend.

- `join` creates a Stream call then opens a session; `chat` opens one with `text: true` and no
  call; `answer` joins a call that already exists, because dispatch already put the caller in
  it; `startCall` places the call *before* joining, since placing it makes the routing rule
  pinned to that call, and attaching the number first would be a second rule for the same
  number.
- What is written in code wins over what a `folder` says. A directory is a starting point.
- `sync` uses **`POST /v1/agents/sync`**, one request carrying the whole directory, not the
  configs/skills/knowledge sequence the Go SDK hand-rolls. It sends a SHA-256 `fingerprint` of
  everything in the body, so syncing on startup does nothing when nothing changed, and it
  writes the knowledge urls only when the router says something did.
- `resolveConfig` turns a config *name* into an id once and caches it. A name matching nothing
  stored is passed through, because it is then either an id or a mistake the router can report
  better than a guess here.

## Dispatch

The worker connects out and the router pushes work down the connection, so nothing the customer
runs has to be publicly reachable.

- `capacity` is a promise about what this process can answer: the router passes over a full
  worker rather than queueing behind it.
- Handlers run on their own, never awaited in the read loop. Answering one caller in line
  leaves the next listening to a ringing phone.
- A call is reported `accepted` or `rejected` with the reason; **a message is not**. Accepting
  and rejecting are about a caller waiting on a line, and there is no line for a message.
- A message only arrives when no agent is running on its channel. `sessionFor` keeps one
  session per channel for the same reason: the session that answered the last message is the
  one that knows what has been said.
- `load` reports only `active_agents` and a round trip this side measured. Host CPU and memory
  are not portable across runtimes, and an invented figure would be read as a real one.
- A signal that aborts *while the socket is still opening* must still stop it: a listener added
  to an already-aborted signal is never called, so `run` checks `signal.aborted` after the
  upgrade. This was a real bug; keep the check.
- `run` drains work in flight before returning. Dropping it would hang up on whoever is talking.

## Tests

`node --test`, no test framework. Never mock.

**A real HTTP server and a real WebSocket upgrade**, not a stubbed `fetch`. `tests/router.ts` is
a `node:http` server plus a `ws` upgrade that records what arrived and lets a test script frames
and read the ones the SDK sent. What the suite exercises is the request this SDK actually puts
on the wire, which is the only way a header, a query parameter or an encoded path is really
checked.

- Assert on outputs, state and **what reached the far end**. Never that a method was called.
- Name a test after the behaviour and why it matters: "tells the model a tool did not work,
  since it is mid-sentence waiting".
- Injecting `fetch` into `Edge` is the documented seam for a test, since the URL is Stream's
  and not a local server's. That is the one place it is acceptable.
- Wait on a condition, not a fixed sleep, wherever the thing waited for is observable.
- `--test-timeout` is set, because a socket bug otherwise hangs CI instead of failing it.

## Style

- British-flavoured prose in doc comments, lower case after a colon, no marketing. Explain
  *why*, and what would go wrong otherwise. Match the surrounding tone.
- Private members are TypeScript `private`, not `#`. `#` breaks structural typing in ways the
  generated shapes trip over.
- `export function` helpers at the bottom of the module, not a `utils.ts`.
- `unknown` over `any`, always. There is no `any` in this package and no `// @ts-ignore`.
- One error type per failure kind: `RouterError` (status, operation, what the router said),
  `SocketClosedError`, `ConfigurationError`. A request that never arrived is `status: 0`,
  because a caller retrying a network failure and one retrying a 500 are different.
- Do not add a logger. A library that logs is a library deciding where a customer's transcripts
  go.

## Reviewing a change

Reject it if it:

- adds a runtime dependency, or reaches for `node:` anything outside `folder.ts`;
- adds a client method for a single endpoint, or re-declares a generated schema;
- copies a schema default into a request, or sends a field the caller did not set;
- handles a tool call anywhere a caller must be iterating for it to run;
- treats an unknown frame as fatal, or drops its payload;
- adds reconnection, a `stream-auth-type` query parameter, or a `typeof window` branch;
- opens two readers on one socket;
- asserts that a method was called, or replaces the test server with a stubbed `fetch`;
- hand-edits `src/generated/api.ts`.
