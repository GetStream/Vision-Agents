---
name: sdk
description: How to build an SDK for the acceleration backend
---

* we use openAPI, so generate your SDK from the openAPI spec
* some endpoints are server side only. such as configuring agents, or listening to agent dispatch

## Resource methods, never raw requests

Every resource gets a clean, named API: `client.simulations.create(...)`, `client.simulations.update(id, ...)`,
`client.simulations.run(id)`, `client.simulations.runs.get(id)`. The generated client is the layer
underneath, not the public API. Users, examples and docs never write `client.post("/v1/agents/simulations", ...)`,
call generated operation functions (`CreateSimulationWithResponse`), or import generated modules.

```js
// never
const simulation = await client.post("/v1/agents/simulations", { body });
// always
const simulation = await client.simulations.create(body);
```

* Group by resource, and nest sub-resources: `simulations.runs.list()`, `agent.sessions.create()`.
* Use the standard verbs: `create`, `get`, `list`, `update`, `delete`, plus actions named after the endpoint (`run`, `cancel`, `fork`).
* Take keyword arguments or the language's options struct, and return the typed model.
* A new endpoint isn't finished until it has a resource method in Go, and the docs use that method.

## Supported SDKs

Client side: JS, swift, kotlin, dart/flutter
Backend: Go, .net, Ruby, .net, Rust, PHP, Node

## SDK best practices

Do deep research on SDK best practices. Use OpenAI sol for this using the tokens available in .env
Based on this deep research create an sdk-mylanguage skill in this repo

## Where to place the SDK

sdks/mylanguage

## Folder sync structure

The structure of an agent folder is like this

- agent.yaml (use this to detect/validate the folder for syncing)
- instructions.md
- guardrail.md
- skills 
- knowledge (markdown files and urls)
- simulations (`*.yaml`, each a list of simulations: `name`, `scenario`, `assertion`, and optionally `mode`, `variations`, `max_turns`, `caller_target`, `judge_target`, `caller_stt`, `caller_tts`, `caller_voice`, `tags`)

For a router a folder can also contain router.yaml

Every backend SDK has a sync method which syncs the folder to the go acceleration backend
In .agent_sync store the sync status:
- hash of the files last synced
- when the last sync happened

This setup prevents duplicate syncs when they nothing changed.

## SDK updates

A change to the API starts in Go. The note for the other SDKs is a new file in
`.claude/skills/sdk/changes/`, not a paragraph added to this skill. One file per
change:

```markdown
---
pending: [js, python, dotnet, ruby, rust, php, swift, kotlin, dart]
---

What changed, the name of the method, and which SDKs have already moved.
```

`pending` starts as every SDK. Remove a name only when that note has nothing for
it, including nothing to draw, refuse, or omit. A client SDK stays listed when
the note tells it to do any of those, even if there is nothing to regenerate. A
port removes its own name and deletes the file when `pending` is present and
empty. Python's log is `plugins/stream/.sdk_update_log/`; the others are
`sdks/<lang>/.sdk_update_log/`.

`changes/backlog.md` is the queue that sat at the bottom of this file when the
notes moved out. Read it while it exists, and do not append to it. Its
`pending` is the client work an OpenAPI diff does not carry. Delete it only
when that list is empty and every server SDK's `openapi.yaml` snapshot was
committed with or after the commit that last changed `changes/backlog.md`.
Do not pin a SHA: a squash of the PR that added the file does not keep that
commit. A checkout gives every snapshot the same mtime, so compare commits,
not mtimes:

```
git merge-base --is-ancestor "$(git log -1 --format=%H -- .claude/skills/sdk/changes/backlog.md)" "$(git log -1 --format=%H -- <sdk>/.sdk_update_log/openapi.yaml)"
```

The server snapshots are `sdks/go`, `sdks/js`, `plugins/stream`, `sdks/dotnet`,
`sdks/ruby`, `sdks/rust`, and `sdks/php`.

To bring one SDK up to date:

1. Diff `acceleration/api/openapi.yaml` against that SDK's `.sdk_update_log/openapi.yaml`.
2. Read `changes/backlog.md` while it exists, and every other file in `changes/` whose `pending` names this SDK.
3. Copy `openapi.yaml` into that `.sdk_update_log/`. Do not copy this skill.
4. Remove this SDK from `pending` on each file that lists it, other than `changes/backlog.md`. Delete a file only when its `pending` key is present and the list is empty. Do not delete `backlog.md` here, and do not remove a name from it. `js` on that list is the browser client: it comes off only after the paragraphs that name the browser client are done, not when the generated client is regenerated.

## Client side SDKs

* Expose nice stateflow in Kotlin, or your language equivalent so it's easy to customize
* Use the modern UI frameworks (compose or swiftUI)

We want to expose 2 different SDKs client side

* ai-language-core (state layer and APIs only)
* ai-language-rtc (add video and voice capabilities which are relatively large)

Include Stream's chat and voice SDKs as dependencies

Here is an example of the syntax the JS SDK. Do something similar for other SDKs, but keep it aligned with the language best practices

```js
import { Client } from "@stream-io/vision-agents";

const api = new Client({ url: accelerate, apiKey });
await client.setUser(
  {
    id: "jlahey",
    name: "Jim Lahey",
  },
  "{{ user_token }}",
);


const agent = api.agent(“docs”);

const options = {incognito: true/false, custom: {}, title: “”, description: “”, model_overwrites: {thinking: high}, project=”Health”}
const session = agent.sessions.create(options);
const oldSessions = agent.sessions.query(); # support search
const oldSessions = agent.sessions.search(); # support search

session.responses.create(“Is Stream better than Sendbird?”)
session.responses.items() // list of responseItems for actions on them/rewind
session.responses.rewind(responseItem) // go back to a response and continue from there
session.interrupt()

session.fork(options) // similar options to channel creation

// one method changes a session: title, description, custom, instructions, models, voice
session.update({title: "Pricing", llm: "openai/gpt-5"})
agent.sessions.update(sessionId, {title: "Pricing"}) // an ended session can still be renamed
```

### Updating a session

Changing a session is one method named `update`, spelled the way the language spells it:
`session.update` (JS, Python, Ruby, Rust, Dart, Kotlin, Swift, PHP), `session.Update` (Go),
`session.UpdateAsync` (.NET). It calls `PATCH /v1/agents/sessions/{id}` (`updateSession`)
and returns the session as it now is. Take the generated `UpdateSessionRequest` or the
language's keyword arguments; a field left out is left as it is.

* Editable: `title`, `description`, `custom`, `instructions`, `llm`, `stt`, `tts`, `sts`,
  `subagent`, `voice`, `thinking`, `temperature`, `max_output_tokens`, `verbosity`.
* Not editable: `id`, the call, `incognito`. Fork for a session that differs in those.
* A session that ended accepts only `title`, `description` and `custom`, so also offer
  `agent.sessions.update(id, ...)` for renaming one without a live handle.
* A device may call it too, but only for `title`, `description` and `custom`; anything else
  is a 403. A client SDK (Swift, Kotlin, Dart) offers `update(title:description:custom:)`
  and nothing more.
* Don't add separate `setInstructions` / `updateSettings` methods for new SDKs; the
  `/instructions` and `/settings` endpoints are deprecated.

```js


Guest users

user = client.guestUser(options); // gets or creates a guest user (searches in cookie or device storage)

client.claimGuestUser(guestUser, realUser); // only supported server side. 
```

## Server side SDKs

* for server side we just provide 1 sdk per language
* starting an agent for an inbound call, text message, whatsapp message, slack message etc
* placing an outbound call
* syncing the folder config for an agent

Include Stream's server side SDK as a dependency. 

Here's an example of python 

Normal call

```
async def create_agent(**kwargs) -> Agent:
    agent = Agent(
	    config="simple_voice_ai", 
	    cost_tracking={env: "production"},
	    memory_filter={user_id: 123}, # memory visibility
    ) 
    return agent


async def join_call(agent: Agent, call_id: str, **kwargs) -> None:
    async with agent.join(call_id):
        await agent.responses.create("greet the user in one short sentence")
```

Inbound call

```
dispatch = acceleration.StreamDispatch()


@dispatch.wait_for_call() # websocket based
async def inbound_call(call: acceleration.CallContext):
    agent = Agent(
        config="restaurant_orders",
    )
    async with agent.join(call):
        await call.wait_for_phone_participant()
        await agent.responses.create(
            "greet the user and let them know you're a friendly AI agent"
        )
```

Outbound call

```
agent = Agent(
  config="recruiter_voice",
)

async with agent.outbound_call(
  from_=os.environ["OUTBOUND_FROM"],
  to=os.environ["OUTBOUND_TO"],
  call_id="hello",
):
  await agent.responses.create(
      "greet the user and let them know you're a friendly AI agent"
  )
```

Text/ respond cycle

```
async def create_agent() -> Agent:
    return Agent(config="chat_support")


@dispatch.wait_for_message()
async def inbound_message(message: acceleration.InboundMessage):
    agent = await dispatch.get_or_create_agent(message, create_agent)
    await agent.responses.create(message.text)
```

Sandbox

Call it the sandbox, never the VM: in method and type names, parameters, comments, logs, errors, READMEs and docs.

```
agent = Agent(config="chat_support", sandbox=Daytona())
```

Knowledge

```
page = await agent.knowledge.add_url(QUICKSTART)
```

Router for STT

```
router = client.router("clinic")  # "clinic" is a router config; it holds the target

async with router.stt.realtime() as stt:
	 await stt.process_audio(chunk, CALLER)
```

Two rules for the router, in code, examples, READMEs and docs:

* The router comes from the client: `client.router("clinic")` (`client.Router("clinic")` in Go and
  .NET). Never construct it standalone (`Router("clinic")`, `new Router(...)`, `Router::new(api)`),
  because the client is where the URL and credentials were settled, and a router built on its own
  quietly rebuilds a backend from the environment.
* Which model answers (`target`, such as `en-low-latency`) lives in the router config
  (`routers/<name>/router.yaml` or `configure_stt`/`define_router`), never in the `realtime()` call.
  `realtime()` takes only per-call overrides like `diarize`, `keyterms` or `voice`. Wrong:
  `router.stt.realtime(target="en-low-latency")`. Right: put `target: en-low-latency` under `stt:`
  in the config and call `router.stt.realtime()`.
