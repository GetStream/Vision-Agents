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

## Asking goes through responses.create

The Go SDK has no `session.Respond`: every question is `session.Responses.Create(ctx, text, inputs...)`,
which returns the response id, takes images and clips, and adds a `command_id` for a conversation kept in
Stream Chat. The socket `respond` frame is left for the router, not wrapped by an SDK. Every SDK has
moved: the `respond`, `RespondAsync` and `send` wrappers are gone.

`listSessions`, `searchSessions`, `listResponses` and `listResponseItems` page by cursor now (see the
`pagination` skill): `cursor` replaces `offset`, and each returns `{items, has_more, next_cursor}`
instead of an array. Every SDK has moved.

`createSession` takes an optional `id` (a UUID the caller chose) and answers 409 when a session already has it; generated ids are UUIDv7. Every SDK exposes it.

A session is changed with `update`, backed by `updateSession` (`PATCH /v1/agents/sessions/{id}`): title, description, custom, instructions, models and voice in one call. `setSessionSettings` (`PATCH .../settings`) is deprecated. Every server SDK has moved (`session.update`, `Update` in Go, `UpdateAsync` in .NET). `updateSession` is now `x-client-accessible`: a device may change its own session's title, description and custom, and is refused with a 403 for instructions, models or voice. Go is regenerated. Swift, Kotlin and Dart need `session.update(title:description:custom:)` (and `sessions.update(id, ...)`), and the server SDKs only need regenerating, since `UpdateSessionRequest` is now rendered from Go.

Credentials come from the environment. A client built with no arguments reads `STREAM_API_KEY` and `STREAM_API_SECRET` itself, so examples, READMEs and docs write `new Client()` (or the SDK's equivalent), never `new Client({ apiKey: process.env.STREAM_API_KEY, apiSecret: process.env.STREAM_API_SECRET })`. Pass them explicitly only when they come from somewhere other than those variables. Go, JavaScript, Python (`plugins/stream`), Ruby, PHP, .NET and Rust already fall back to them; any SDK that does not should, and snippets that pass them by hand should drop them.

`listSessions` and `searchSessions` are replaced by `querySessions` (`POST /v1/agents/sessions/query`), which takes `{filter, sort, limit, cursor}` in the body (see the `query` skill). The filter allows `agent`, `user_id`, `project_id` and `modality` (a bare value or `{"$eq": ...}`) and `text: {"$q": ...}`. It sorts by `updated_at`, or by `relevance` for a text search, which cannot be combined with `project_id`. `project` is now `project_id` on `createSession`, `forkSession` and `Session`. `Session` gains a required `modality` (`text`, `voice` or `video`). Every SDK has moved (Go: `Query.ProjectID`, `Query.Modality`, `SessionOptions.ProjectID`, `ForkOptions.ProjectID`, `Call.ProjectID`). The old `custom`, `created_after`/`created_before` and `offset` filters are gone with them.

Memory can be deleted. `truncateMemories` (`DELETE /v1/agents/users/{user_id}/memories`) deletes everything remembered about one user, from every session and agent; `deleteSessionMemories` (`DELETE /v1/agents/sessions/{id}/memories`) deletes what one session learned. Both answer 204 and are server-side only. Stopping a session (`stopSession`, what `close` calls) keeps its memories, so never wipe memory from `close`. Name them `memories.truncate(userId)`, `sessions.deleteMemories(id)` and `session.deleteMemories()`, spelled the way the language spells them. Every server SDK has moved (Go: `Client.Memories().Truncate`, `Sessions.DeleteMemories`, `Session.DeleteMemories`). Swift, Kotlin and Dart leave them out, being server-side only.

`closeSession` (`DELETE /v1/agents/sessions/{id}`) is split in two. `stopSession` (`POST /v1/agents/sessions/{id}/stop`) is what ending a call does: the agent leaves and everything the session recorded and remembered is kept. `deleteSession` (`DELETE /v1/agents/sessions/{id}`) now deletes the session: it stops it if it is running, deletes its turns and items, and deletes what it taught memory. Both answer 204 and are client-accessible. A conversation in writing is normally left running, so `close` should only stop a call. Every SDK has moved: `close` stops (including when the socket failed to open) and `delete` deletes.

An agent config can be changed in part. `patchAgentConfig` (`PATCH /v1/agents/configs/{id}`) takes an `AgentConfigPatch` and writes only the fields sent, so a guardrail or instructions can be set without restating everything else; `updateAgentConfig` (PUT) and `syncAgent` still replace instructions, guardrail, skills and knowledge. Server-side only. Name it `updateConfig` on the agent handle, spelled the way the language spells it: look the config up by the agent's name, then patch it. Every server SDK has moved (Go: `client.Agent.UpdateConfig`).

Simulations need resource methods (see "Resource methods, never raw requests"): `simulations.create/get/list/update/delete/run` and `simulations.runs.get/list/cancel`. JavaScript, Python, .NET, Ruby, Rust and PHP have them (`client.simulations`, `client.simulations.runs`). Go still only has the generated `CreateSimulationWithResponse` and needs them. Swift, Kotlin and Dart leave them out, being server-side only.

An agent folder can declare simulations in `simulations/*.yaml`. Each file is a list, so related simulations can share a file, and names must be unique across files. `syncAgent` (now declared in Go with Huma) takes them as `simulations: [SimulationDeclaration]`. When the field is sent, the router makes the config's simulations exactly that list: each is found by name and updated in place, so its runs stay attached, and one no longer declared is deleted. When the field is left out, the stored ones are left alone. So send `simulations` only when the folder has a `simulations/` directory, and send an empty list when that directory is empty. Refuse unknown keys, as with `agent.yaml`. The fingerprint appends `"\nsimulations:"` and then each simulation's JSON (field order as in the Go `agents.Simulation`), only when `simulations/` exists, so folders without one keep their current hash. Every server SDK has moved, with fingerprints checked against Go's (`agents.Folder.Simulations`, sent by `Agent.Sync`). `syncAgent` now validates its body, so a skill must carry `config_id` (every SDK already sends `""`).

`Policy` (`getAppPolicy`, `updateAppPolicy`, `getOrganizationPolicy`, `updateOrganizationPolicy`) gains `allowed_models` and `tags`. `allowed_models` is a list of `provider/model` names the router may route to in every modality: left out allows every model, an empty list allows none, and an app is held to the models both it and its organization allow. `tags` are recorded on every usage row over the request's own, with the organization's winning over the app's. Keep the difference between an absent and an empty `allowed_models` when (de)serialising, since they mean opposite things. Every SDK has the fields in its regenerated client; none wraps the policy endpoints yet.
An agent config has a `speed`: the voice's rate of delivery, 1 being its own, zero or absent leaving it there. It is on `AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest`, and an agent folder's `agent.yaml` may declare `speed:` (send it only when the file names a non-zero one). Only voices that can change speed are routed to when it is set (ElevenLabs flash v2.5 and multilingual v2 today, 0.7–1.2). Every SDK has the field, and every server SDK's folder loader accepts `speed` (Go: `agents.Settings.Speed`).
An agent config has `visible_tools` (on `AgentConfig`, `AgentConfigRequest` and `AgentConfigPatch`): tool names or `path.Match` patterns such as `athena_*` whose steps end users see on a persistent conversation's replies. Empty shows `search` and `web_search`. Every SDK has the field in its regenerated client.
A session tool (`SessionTool` on `createSession`) gains `executor` (`server`, the default, or `client`: a person's device runs it) and `display_title` (at most 80 characters, shown on the reply's `ai_tool_call` attachment), and `RespondRequest` gains `client_id`, the install a command came from, which a client tool called while answering it is addressed to. A client tool is still answered over the events socket, once the device has reported. Every SDK has the fields in its regenerated client. Swift, Kotlin, Dart, Ruby, Rust and PHP expose `executor` and `display_title` where they declare tools; .NET exposes only `display_title`, and Python neither, since its tools come from the core `FunctionRegistry`. No SDK sends `client_id`: it is only on the socket `respond` frame, which SDKs no longer wrap, and `createResponse` has no such field, so a client tool can't yet be addressed through `responses.create`.
`querySessions` takes two more filter fields, `state` (`live` or `ended`, as `Session.state` reports it) and `agent_id` (the id a session was created with), each a bare value or `{"$eq": ...}`, so a caller can list a user's live sessions without paging through every one that ended: `{"filter": {"user_id": "u1", "state": "live"}}`. Every SDK has moved (Go: `Query.State`, `Query.AgentID`). Kotlin leaves out `user_id`, which only a backend may set.

The harness is agent config, never session config. `createSession` no longer takes `subagent`, `tasks`, `sandbox`, `skills` or `skill_names`, and `subagent` is gone from `model_overwrites` and `updateSession`; the router ignores them if sent. `tools` and `tool_timeout_ms` stay, which is how a client runs its own sandbox (artemis-impl's `investigate_sdk`). `AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest` gain `harness` (an enum with one value, `default`; absent means `default`), next to the `subagent`, `sandbox` and `skills` they already had, and `agent.yaml` may declare `harness:`. Go has moved: `stream.Call` lost `Subagent`, `Tasks`, `Sandbox` and `Skills`, `stream.Config` lost `Subagent`, and `agents.Harness` (`Name`, `Subagents`, `VM`, `Skills`) is written by `Agent.Sync` onto the config rather than onto each session, so a session only gets it by running under that config. Every SDK has moved: the session-level options (and `use_skills`/`tasks` on the harness) are gone, and every server folder loader reads `harness:`.

A knowledge page can be read again on a schedule. `KnowledgeUrlRequest` (`addKnowledgeUrl`), `KnowledgeUrlDeclaration` (in `syncAgent`) and `KnowledgeUrl` gain `refresh_hours`, how many hours between reads; absent means never, which is what it was before. Adding a page again replaces it, so a declaration without it turns the schedule off. In `knowledge/urls.yaml` a page written as a mapping may say `refresh_hours: 24` (at least 1; refuse 0, a non-integer and unknown keys as before). The fingerprint appends `"\nrefresh_hours:" + N` after a page's description only when it has one, so directories without it keep their current hash. Every server SDK has moved (Go: `agents.KnowledgeURL.RefreshHours`). JavaScript has no add-url method, so only its loader reads it.

A dispatch worker can host tools for an agent: after every `ready` it sends `host_tools` (`agent_id`, `tools`, `timeout_ms`), runs each `tool_call` off the read loop and answers `tool_result` with `output` or `error`; `hosting_refused` ends the worker. The router offers them to a session whose `agent_id` or config name matches. Go, JavaScript (`dispatch.host`), Python (`stream.Dispatch.host`), Ruby, PHP, .NET (`Dispatch.Host`) and Rust all have it. Python and .NET also reconnect like Go; JavaScript, Ruby, PHP and Rust re-declare on each `ready` but do not reconnect.

The router holds a dispatch worker to the capacity it declared. A worker connects with `capacity`,
`active` (the calls and messages it is still handling, `0` on a first connection; hosted tool calls never
count) and `handles` (`call`, `message`, both, or empty for a worker that only hosts tools). Every `call`
and `message` frame carries a `work_id`, and the worker answers `{"type": "done", "work_id": ..., "error":
...}` once it is finished, failure and missing handler included, which is what gives it its room back.
There is no `accepted` or `rejected` any more. Go, Python (`plugins/stream`), JavaScript, .NET, Ruby, Rust
and PHP all do this.

An agent can leave text to its server. `agent.yaml` may say `dispatch: {incoming_call: enabled, text:
enabled}` (each `enabled` or `disabled`), sent as `dispatch` (`AgentDispatch`) in `syncAgent`, and readable
and patchable on the config. With `text` enabled, what an end user writes over the session socket or in
Chat goes to a dispatch worker as a `message` frame instead of the model, carrying `session_id`, and
`command_id` when it was a durable command, possibly with no channel. `incoming_call` is stored only, since
every inbound call is already dispatched. Every server SDK reads `dispatch:`, exposes the session and
command ids on its inbound message, refuses such a message in its get-or-create helper, and has `answer`
(`Answer` in Go, `AnswerAsync` in .NET): responses.create on the message's session with its `command_id`,
as the server acting for the writer. Acting for somebody needs its own backend setting, because a user id
behind the proxy mints a user token, and the router hands a user's text back to the worker: Go
`Backend.ActingFor`, Python `Backend.acting_for`, JavaScript `Backend.actingFor`, Ruby `Backend#acting_for`,
Rust `Client::acting_for`, PHP `Backend::onBehalfOf` (its `actingFor` already meant a user token) and .NET
`VisionAgentsClient.ActingFor`. They send the server credential plus `X-Stream-User-Id` in every mode.

`SttOptions` gains `eager_end_of_turn` (boolean, live only): the router turns it into Deepgram Flux's
`eager_eot_threshold` (0.6, never above `eot_threshold`), and every other model ignores it rather than
refusing it. It is on by default for `en-low-latency` and `multilingual-low-latency`. Every SDK with STT has it.

The router comes from the client and its target from the router config (see "Router for STT"). Every
SDK with a router has moved: `client.Router` (Go, .NET), `client.router` (Python, Ruby, Rust, PHP) and
`agents.router(config:)` (Swift, Kotlin, Dart). Constructing one directly is internal everywhere except
Python, where `Router(...)` stays for the `url`/`customer_id` case. JavaScript has no router yet; when it
gets one, start from `client.router(name)`.

Connector definitions can be listed, read and added. `listConnectors` (`GET /v1/agents/connectors`, with `q`, `limit` and `cursor`, answering a `ConnectorDefinitionPage`), `getConnector` (`GET /v1/agents/connectors/{id}`) and `createConnector` (`POST /v1/agents/connectors`, a `CustomConnectorRequest` for a custom MCP server whose id starts with `custom_`) are all server-side only. A `ConnectorDefinition` shows the schemes, inputs, scopes and client policy, never endpoints or the operator's client variables. Name them `connectors.list/get/create`, spelled the way the language spells them, with `list` following the cursor like `Items.Unwind`. Go has the regenerated client (`ListConnectors`, `GetConnector`, `CreateConnector`) and no resource methods yet. JavaScript, Python (`plugins/stream`), .NET, Ruby, Rust and PHP need their generated clients regenerated and the three methods added. Swift, Kotlin and Dart need nothing, since none of the three is client-accessible.

`agent.yaml` names `user_plugins` beside `plugins`: catalog MCP servers each end user connects with their own account, from the conversation, rather than the app once for everybody. `syncAgent`, `AgentConfig`, `AgentConfigRequest` and `AgentConfigPatch` carry it as a list of ids. The model gets `<id>__list_tools` and `<id>__call_tool`, and the first call for somebody who has not connected answers `{"status":"authorization_required","message":...,"attachment":{"type":"plugin_authorization","plugin_id","title","authorize_url"}}`; the reply's Chat message carries the attachment, and `conversation_updated` its `authorizations`. Only a session with a verified end user is offered them, and a conversation kept for one takes each `respond` with a `command_id`. `listConfigPlugins` lists a plugin the config names that the app has not connected as `not_connected`. Go reads `user_plugins` and has the regenerated client; Python (`plugins/stream`) reads it, takes `Accelerated(user_id=...)`, reports `authorization_required` events and follows a `responded` with `pending_work` until the reply has nothing pending. JavaScript has the regenerated types. .NET, Ruby, Rust and PHP need `user_plugins` in their folder readers. Client-side SDKs (Swift, Kotlin, Dart, JavaScript UI) should render a `plugin_authorization` attachment as a button opening `authorize_url`.

An agent config has `sandbox_options` (on `AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest`): `image`, `setup` (shell commands), `timeout_ms` (at most 1800000), `cpu`, `memory_gb` and `disk_gb`, saying how the Daytona sandbox is built and how long one run may take. `agent.yaml` may declare `sandbox_options:` with the same keys except `timeout`, a duration such as `5m` sent as `timeout_ms`; refuse unknown keys, a timeout over 30m and sizes that are not whole numbers, and send it only when the file has the block. `task_settled` frames gain `files`, a list of `{name, mime_type, url, size}` for what the subagent's code handed back. Go (`agents.SandboxSettings`, `stream.Event.Files`) and Python (`SandboxSettings` in `plugins/stream`, `RemoteEvent.files`) have moved. JavaScript, .NET, Ruby, Rust and PHP need their folder loaders to read `sandbox_options`, their generated clients regenerated, and their task-settled events to carry `files`. Swift, Kotlin and Dart only need `files` on their event, since config is server-side.

The agent config field `subagent` is now `thinking_llm`, on `AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch`, `SyncAgentRequest`, `Session` and `Call` (whose `subagent_used` is now `thinking_llm_used`), and `agent.yaml` names it `thinking_llm:`; an old `subagent:` key is refused as unknown. Only a voice agent may name one: the router answers 400 for a text agent that does, and a text session runs its skills on its own `llm`. Go, Python (`plugins/stream`, `define_agent(thinking_llm=...)`) and JavaScript have moved. .NET, Ruby, Rust and PHP need their folder loaders, generated clients and sync requests renamed. Swift, Kotlin and Dart need their generated `Session` and `Call` models regenerated (Swift's generated code is stale apart from this).

An agent config has `plugin_events` (on `AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest`): a list of `PluginEvent` `{plugin, event, arguments, instructions}`, the MCP events (the MCP Events draft, protocol `2026-07-28`) the agent subscribes to on a plugin it names under `plugins` or `user_plugins`. The router subscribes with every login the config holds to that plugin, and each event delivered opens a text session from the config, as the login's owner. `agent.yaml` declares it as `plugin_events:` with the same keys; refuse unknown keys and send it only when the file has the list. Go (`agents.PluginEventSettings`) has moved, and the Python, Swift and JavaScript generated clients are regenerated. Python (`plugins/stream` folder reader, which refuses the key today), .NET, Ruby, Rust and PHP need their folder loaders to read `plugin_events` and their generated clients regenerated. Swift, Kotlin and Dart need nothing, since config is server-side.

An agent config has `mcp_servers` (on `AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest`): a list of `McpServer` `{name, url}`, MCP servers outside the plugin catalog that every session opens by URL with no login, offering their tools as `<name>__<tool>` and adding what each server says at initialize to the agent's instructions. The router refuses a name that is a catalog id, holds `__` or repeats, and a url that is not https. `agent.yaml` declares it as `mcp_servers:` with the same keys; refuse unknown keys and send it only when the file has the list. Go (`agents.MCPServerSettings`) and Python (`MCPServerSettings` in `plugins/stream`) have moved, and the JavaScript types are regenerated. JavaScript, .NET, Ruby, Rust and PHP need their folder loaders to read `mcp_servers` and their generated clients regenerated. Swift, Kotlin and Dart need nothing, since config is server-side.

Catalog plugins have a logo, and the `plugin_authorization` attachment carries it. `Plugin` and `PluginConnection` gain a read-only `logo_url`, and `getPluginLogo` (`GET /v1/agents/plugins/{plugin_id}/logo`) serves it as an SVG with `security: []`, because what draws it is an `<img>` in a chat client with no credential of ours to send. The attachment that asks an end user to connect now also carries Chat's own `text` (the catalog description), `thumb_url` (that logo) and `title_link` (the authorize URL) beside `type`, `title`, `plugin_id` and `authorize_url`, so a client with no renderer for the type still shows a card somebody can press. Those three and no others, because they are what Chat keeps as an attachment's own field on a partial update; `author_name` would come back as custom data, and the title already names the plugin. The router refuses an attachment whose description, link or thumbnail is not what the catalog says for the plugin named. Go, Python (`plugins/stream`), JavaScript and Swift have the regenerated clients. .NET, Ruby, Rust and PHP need their generated clients regenerated. Client-side SDKs (Swift, Kotlin, Dart, JavaScript UI) should draw the logo and description on the button they already render for `authorize_url`.

An agent config has `plugin_options` (on `AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest`): a list of `PluginOptions` `{plugin, readonly, scopes, toolsets}`, saying how a catalog plugin is reached and what its login asks for. `readonly` reaches the plugin's read-only MCP server (Linear's `https://mcp.linear.app/mcp/readonly`, asking only for `read`), and the router refuses it for a plugin without one; `scopes` replaces the scopes asked for at consent; `toolsets` limits the server to some of the groups its catalog entry lists (Cal.com's `bookings`, `availability` and the rest). `Plugin` in the catalog gains `readonly` and `toolsets`, saying which a plugin accepts. `agent.yaml` declares it as `plugin_options:` with the same keys; refuse unknown keys and send it only when the file has the list. Go (`agents.PluginOptionsSettings`) and Python (`PluginOptionsSettings` in `plugins/stream`) have moved, and the Go, Python, JavaScript and Swift generated clients are regenerated. JavaScript, .NET, Ruby, Rust and PHP need their folder loaders to read `plugin_options` and their generated clients regenerated. Swift, Kotlin and Dart need nothing, since config is server-side.

`PluginOptions` and `McpServer` gain `tools`: tool names or `path.Match` patterns such as `get_*`, and only matching tools are offered (left out, all are). `agent.yaml` takes it as `tools:` under each `plugin_options` and `mcp_servers` entry. `Plugin` in the catalog gains a read-only `scopes_supported`, and a scope outside it is a 400. Go and Python (`plugins/stream`) folder readers have moved and the Go, Python, JavaScript and Swift generated clients are regenerated. JavaScript, .NET, Ruby, Rust and PHP need their folder loaders to read `tools` and their generated clients regenerated.
`SessionFilter` gains back `config_id`, `custom` and `created_at`, the three the move to `querySessions` dropped: `config_id` is an `$eq` on the agent config that ran the session, `custom` is the pairs a session's custom object must all hold, and `created_at` is a `TimeRange` `{$gte, $lt}`, half open so two windows that meet share no session. The Go SDK's `Query` takes them as `ConfigID`, `Custom`, `CreatedAfter` and `CreatedBefore`, and the Go, Python, JavaScript and Swift generated clients are regenerated. JavaScript, .NET, Ruby, Rust and PHP need the same four fields on their session query and their generated clients regenerated.
