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

For each sdk, have an .sdk_update_log folder which stores a copy of this skill, and the openAPI spec that was last used when updating the SDK
this makes it easier to update an SDK and know that you just need to add a few fields etc. 

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
router = acceleration.Router("clinic")

async with router.stt.realtime() as stt:
	 await stt.process_audio(chunk, CALLER)
```

## Asking goes through responses.create

The Go SDK has no `session.Respond`: every question is `session.Responses.Create(ctx, text, inputs...)`,
which returns the response id, takes images and clips, and adds a `command_id` for a conversation kept in
Stream Chat. The socket `respond` frame is left for the router, not wrapped by an SDK. The other SDKs
still wrap the frame (`respond` in JavaScript, Python, Ruby, PHP and Rust, `RespondAsync` in C#, and
`send` in Kotlin, Swift and Dart) and should drop it for `responses.create` the same way, with the docs
moving with them.

`listSessions`, `searchSessions`, `listResponses` and `listResponseItems` page by cursor now (see the
`pagination` skill): `cursor` replaces `offset`, and each returns `{items, has_more, next_cursor}`
instead of an array. Go and JavaScript have moved. Python (`plugins/stream`), Swift, Kotlin, Dart, .NET,
Ruby, Rust and PHP still send `offset` and expect an array, and need to move with their generated
clients regenerated.

`createSession` takes an optional `id` (a UUID the caller chose) and answers 409 when a session already has it; generated ids are UUIDv7. Go (`SessionOptions.ID`) and JavaScript (`sessions.create({ id })`) have moved. Python (`plugins/stream`), Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP need their generated clients regenerated and the option exposed.

A session is changed with `update`, backed by `updateSession` (`PATCH /v1/agents/sessions/{id}`): title, description, custom, instructions, models and voice in one call. `setSessionSettings` (`PATCH .../settings`) is deprecated. Go (`session.Update`) and JavaScript (`session.update`) have moved. Python (`update_settings`), Ruby, Rust (`update_settings`), PHP (`updateSettings`) and .NET (`UpdateSettingsAsync`) still call the settings endpoint and should become `update` on `updateSession`, with the "Update a running session" tabs in the docs moving with them.

Credentials come from the environment. A client built with no arguments reads `STREAM_API_KEY` and `STREAM_API_SECRET` itself, so examples, READMEs and docs write `new Client()` (or the SDK's equivalent), never `new Client({ apiKey: process.env.STREAM_API_KEY, apiSecret: process.env.STREAM_API_SECRET })`. Pass them explicitly only when they come from somewhere other than those variables. Go, JavaScript, Python (`plugins/stream`), Ruby, PHP, .NET and Rust already fall back to them; any SDK that does not should, and snippets that pass them by hand should drop them.

`listSessions` and `searchSessions` are replaced by `querySessions` (`POST /v1/agents/sessions/query`), which takes `{filter, sort, limit, cursor}` in the body (see the `query` skill). The filter allows `agent`, `user_id`, `project_id` and `modality` (a bare value or `{"$eq": ...}`) and `text: {"$q": ...}`. It sorts by `updated_at`, or by `relevance` for a text search, which cannot be combined with `project_id`. `project` is now `project_id` on `createSession`, `forkSession` and `Session`. `Session` gains a required `modality` (`text`, `voice` or `video`). Go has moved (`Query.ProjectID`, `Query.Modality`, `SessionOptions.ProjectID`, `ForkOptions.ProjectID`, `Call.ProjectID`). JavaScript, Python (`plugins/stream`), Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP still call the two old endpoints and send `project`, and need to move with their generated clients regenerated.

Memory can be deleted. `truncateMemories` (`DELETE /v1/agents/users/{user_id}/memories`) deletes everything remembered about one user, from every session and agent; `deleteSessionMemories` (`DELETE /v1/agents/sessions/{id}/memories`) deletes what one session learned. Both answer 204 and are server-side only. Stopping a session (`stopSession`, what `close` calls) keeps its memories, so never wipe memory from `close`. Name them `memories.truncate(userId)`, `sessions.deleteMemories(id)` and `session.deleteMemories()`, spelled the way the language spells them. Go (`Client.Memories().Truncate`, `Sessions.DeleteMemories`, `Session.DeleteMemories`) and JavaScript (`client.memories.truncate`, `sessions.deleteMemories`, `session.deleteMemories`) have moved. Python (`plugins/stream`), Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP need their generated clients regenerated and the three methods added.

`closeSession` (`DELETE /v1/agents/sessions/{id}`) is split in two. `stopSession` (`POST /v1/agents/sessions/{id}/stop`) is what ending a call does: the agent leaves and everything the session recorded and remembered is kept. `deleteSession` (`DELETE /v1/agents/sessions/{id}`) now deletes the session: it stops it if it is running, deletes its turns and items, and deletes what it taught memory. Both answer 204 and are client-accessible. A conversation in writing is normally left running, so `close` should only stop a call. Go has moved (`Pipeline.Leave` stops, `Session.Close` stops, new `Sessions.Delete` and `Session.Delete`). JavaScript, Python (`plugins/stream`), Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP still send `DELETE` to close, which now deletes the conversation, and need their generated clients regenerated, `close` moved to `stopSession`, and `delete` added.

An agent config can be changed in part. `patchAgentConfig` (`PATCH /v1/agents/configs/{id}`) takes an `AgentConfigPatch` and writes only the fields sent, so a guardrail or instructions can be set without restating everything else; `updateAgentConfig` (PUT) and `syncAgent` still replace instructions, guardrail, skills and knowledge. Server-side only. Name it `updateConfig` on the agent handle, spelled the way the language spells it: look the config up by the agent's name, then patch it. Go (`client.Agent.UpdateConfig`) and JavaScript (`AgentHandle.updateConfig`) have moved. Python (`plugins/stream`), .NET, Ruby, Rust and PHP need their generated clients regenerated and the method added.

Simulations need resource methods (see "Resource methods, never raw requests"): `simulations.create/get/list/update/delete/run` and `simulations.runs.get/list/cancel`. JavaScript has moved (`client.simulations`, `client.simulations.runs`). Go only has the generated `CreateSimulationWithResponse`, Rust has the flat `create_simulation`, and Python (`plugins/stream`) has only `_generated`. The Python example in the simulations docs already uses `api.simulations.create`, `api.simulations.run` and `api.simulations.runs.get`, so Python needs them to match. Swift, Kotlin, Dart, .NET, Ruby and PHP follow.

An agent folder can declare simulations in `simulations/*.yaml`. Each file is a list, so related simulations can share a file, and names must be unique across files. `syncAgent` (now declared in Go with Huma) takes them as `simulations: [SimulationDeclaration]`. When the field is sent, the router makes the config's simulations exactly that list: each is found by name and updated in place, so its runs stay attached, and one no longer declared is deleted. When the field is left out, the stored ones are left alone. So send `simulations` only when the folder has a `simulations/` directory, and send an empty list when that directory is empty. Refuse unknown keys, as with `agent.yaml`. The fingerprint appends `"\nsimulations:"` and then each simulation's JSON (field order as in the Go `agents.Simulation`), only when `simulations/` exists, so folders without one keep their current hash. Go has moved (`agents.Folder.Simulations`, sent by `Agent.Sync`). Python (`plugins/stream`), JavaScript, .NET, Ruby, Rust and PHP need their generated clients regenerated and the folder loader extended. `syncAgent` now validates its body, so a skill must carry `config_id` (every SDK already sends `""`).

`Policy` (`getAppPolicy`, `updateAppPolicy`, `getOrganizationPolicy`, `updateOrganizationPolicy`) gains `allowed_models` and `tags`. `allowed_models` is a list of `provider/model` names the router may route to in every modality: left out allows every model, an empty list allows none, and an app is held to the models both it and its organization allow. `tags` are recorded on every usage row over the request's own, with the organization's winning over the app's. Keep the difference between an absent and an empty `allowed_models` when (de)serialising, since they mean opposite things. Go (generated `acceleration.Policy`) and JavaScript (generated types) have the fields. Python (`plugins/stream`), Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP need their generated clients regenerated and the two fields exposed wherever they wrap the policy endpoints.
An agent config has a `speed`: the voice's rate of delivery, 1 being its own, zero or absent leaving it there. It is on `AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest`, and an agent folder's `agent.yaml` may declare `speed:` (send it only when the file names a non-zero one). Only voices that can change speed are routed to when it is set (ElevenLabs flash v2.5 and multilingual v2 today, 0.7–1.2). Go has moved (`agents.Settings.Speed`, sent by `Agent.Sync`). Python (`plugins/stream`), JavaScript, Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP need their generated clients regenerated, and the server-side SDKs' folder loaders need to accept `speed`, since they refuse unknown keys.
An agent config has `visible_tools` (on `AgentConfig`, `AgentConfigRequest` and `AgentConfigPatch`): tool names or `path.Match` patterns such as `athena_*` whose steps end users see on a persistent conversation's replies. Empty shows `search` and `web_search`. Go has the field in its regenerated client. Python (`plugins/stream`), JavaScript, Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP need their generated clients regenerated so the field is sent and read.
A session tool (`SessionTool` on `createSession`) gains `executor` (`server`, the default, or `client`: a person's device runs it) and `display_title` (at most 80 characters, shown on the reply's `ai_tool_call` attachment), and `RespondRequest` gains `client_id`, the install a command came from, which a client tool called while answering it is addressed to. A client tool is still answered over the events socket, once the device has reported. Go and JavaScript have the fields in their regenerated clients. Python (`plugins/stream`), Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP need their generated clients regenerated, `executor` and `display_title` exposed where they declare tools, and `client_id` sent where they respond to a command.
`querySessions` takes two more filter fields, `state` (`live` or `ended`, as `Session.state` reports it) and `agent_id` (the id a session was created with), each a bare value or `{"$eq": ...}`, so a caller can list a user's live sessions without paging through every one that ended: `{"filter": {"user_id": "u1", "state": "live"}}`. Go has moved (`Query.State`, `Query.AgentID`). JavaScript has the regenerated types. Python (`plugins/stream`), Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP need their generated clients regenerated and the two fields exposed wherever they wrap `querySessions`.

The harness is agent config, never session config. `createSession` no longer takes `subagent`, `tasks`, `sandbox`, `skills` or `skill_names`, and `subagent` is gone from `model_overwrites` and `updateSession`; the router ignores them if sent. `tools` and `tool_timeout_ms` stay, which is how a client runs its own sandbox (artemis-impl's `investigate_sdk`). `AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest` gain `harness` (an enum with one value, `default`; absent means `default`), next to the `subagent`, `sandbox` and `skills` they already had, and `agent.yaml` may declare `harness:`. Go has moved: `stream.Call` lost `Subagent`, `Tasks`, `Sandbox` and `Skills`, `stream.Config` lost `Subagent`, and `agents.Harness` (`Name`, `Subagents`, `VM`, `Skills`) is written by `Agent.Sync` onto the config rather than onto each session, so a session only gets it by running under that config. JavaScript, Python (`plugins/stream`), Swift, Kotlin, Dart, .NET, Ruby, Rust and PHP need their generated clients regenerated, the session-level harness options removed, and the server-side folder loaders taught `harness`, since they refuse unknown keys.

A knowledge page can be read again on a schedule. `KnowledgeUrlRequest` (`addKnowledgeUrl`), `KnowledgeUrlDeclaration` (in `syncAgent`) and `KnowledgeUrl` gain `refresh_hours`, how many hours between reads; absent means never, which is what it was before. Adding a page again replaces it, so a declaration without it turns the schedule off. In `knowledge/urls.yaml` a page written as a mapping may say `refresh_hours: 24` (at least 1; refuse 0, a non-integer and unknown keys as before). The fingerprint appends `"\nrefresh_hours:" + N` after a page's description only when it has one, so directories without it keep their current hash. Go has moved (`agents.KnowledgeURL.RefreshHours`, sent by `Agent.Sync` and `SubscribeKnowledgeURLs`). Python (`plugins/stream`), JavaScript, .NET, Ruby, Rust and PHP need their generated clients regenerated and their `urls.yaml` loaders, fingerprints and add-url methods extended.

A dispatch worker can host tools for an agent: after every `ready` it sends `host_tools` (`agent_id`, `tools`, `timeout_ms`), runs each `tool_call` off the read loop and answers `tool_result` with `output` or `error`; `hosting_refused` ends the worker. The router offers them to a session whose `agent_id` or config name matches. Go, JavaScript (`dispatch.host`), Python (`stream.Dispatch.host`), Ruby, PHP, .NET (`Dispatch.Host`) and Rust all have it. Python and .NET also reconnect like Go; JavaScript, Ruby, PHP and Rust re-declare on each `ready` but do not reconnect.

`getAppSettings` (`GET /v1/settings/app`) reports what the router does for the calling app. Its `stream` block says whose Stream app the router acts in (`tenancy`: `deployment` or `app`), which app the app's conversations, transcripts, calls and phone lines are written into (`writes_into`: `this_app`, `deployment_app`, the router's own and shared, or `nowhere`), and whether that app holds the `agent` channel type and call type (`channel_type` and `call_type`: `present`, `missing`, `unsafe` for a channel type a client could forge a conversation's channel with, or `unknown` when Stream could not be asked), with `checked_at`. It is read-only, server-side only, and never carries an app id or a secret. Name it `settings.app()`, spelled the way the language spells it. Go has moved (`client.Settings().App`). JavaScript, Python (`plugins/stream`), .NET, Ruby, Rust and PHP need their generated clients regenerated and the method added.
