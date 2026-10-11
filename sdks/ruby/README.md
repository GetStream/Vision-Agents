# getstream-vision-agents

The server-side Ruby client for the Stream acceleration backend: agents, calls, dispatch,
folder sync and the router, from a process you run.

Nothing here does inference or touches media. The backend joins the call, hears the caller,
answers and speaks. What arrives here are the events saying so, and what stays here is
function calling, because the functions are here.

```ruby
gem "getstream-vision-agents"
```

Ruby 3.3 or newer. Two runtime dependencies: `getstream-ruby`, Stream's own server-side gem,
for Stream calls of your own, and `websocket-driver` for the sockets.

## An agent on a call

```ruby
require "getstream/vision_agents"

agent = GetStream::VisionAgents::Agent.new(
  config: "support",
  cost_tracking: { env: "production", team: "voice" },
  memory_filter: { user_id: current_user.id, plan: "pro" },
)

agent.tools.register("get_weather", description: "Weather for a city",
                     parameters: { type: "object", properties: { city: { type: "string" } } }) do |args|
  weather_in(args["city"])
end

agent.join do |session|
  puts agent.monitor_url
  agent.responses.create("greet the user in one short sentence")
end
```

`join` opens a session with voice: the backend joins the session's own call,
`agent:<session id>`, and `join` waits for somebody else to be in it. `id:` picks the session
id, up to 64 of `A-Za-z0-9_-`. The block form closes the session when the block leaves, after
waiting for the call to end unless `wait_for_end: false`. Without a block the open session is
returned.

The session starts from the stored config, and only what the code sets is sent over it.
Instructions reach the backend when the agent syncs. `cost_tracking` labels every request the
session makes; `memory_filter` says who the memories are about, under `user_id`, and what else
narrows recall.

`chat` holds the same conversation in writing: the same instructions, skills and knowledge,
nothing transcribed or spoken. It is kept in Stream Chat unless `incognito: true` is given.
`session.voice.start` puts the agent on the session's call and `session.voice.stop` carries
the conversation on in writing again; `session.voice.started?` says which.
`api.agent("support").sessions.resume(session_id)` carries on a conversation held earlier.

```ruby
agent.chat { |session| session.responses.create("What changed in v3?") }
```

## Credentials

| Credential | Who it is |
| --- | --- |
| `customer_id` | a router with nothing in front of it, which is a laptop |
| `api_key` + `api_secret` | a process you run |

```ruby
api = GetStream::VisionAgents::Client.new
agent = api.agent("support", cost_tracking: { env: "production" })
```

With no arguments the client reads its credentials from the environment: `url` from
`STREAM_ACCELERATION_URL`, then Stream's hosted router, and the rest from
`STREAM_ACCELERATION_CUSTOMER_ID`, `STREAM_API_KEY` and `STREAM_API_SECRET`. Pass them only
when they come from somewhere else. The hosted router is reached through Stream's
authenticating proxy, which is on by default for it; a self-hosted deployment behind the
same proxy passes `authenticate: true` or sets `STREAM_ACCELERATION_AUTHENTICATE`. A monitoring link needs
`STREAM_API_KEY` and `STREAM_API_SECRET` whichever way the router is reached.

## Checked against the spec

Every resource method (`api.agent`, `api.router`, `api.simulations`, `api.memories`) is built
on one method per HTTP verb, which looks the path up in a table generated from
`acceleration/api/openapi.yaml`, so a path, query parameter or body key the spec does not
have is refused before anything is sent. Answers are the router's JSON as string-keyed
hashes. A failure raises `RouterError` with the status, the operation id and what the router
said; status 0 means the request never arrived. Its `type`, `code` and `doc_url` are the
router's error envelope (branch on `code`, which may be new), and `request_id` is the
`X-Request-Id` to quote to support; a refused socket carries the same.

## Dispatch

The worker connects out and the router pushes calls and messages down the socket, so nothing
you run has to be publicly reachable.

```ruby
dispatch = GetStream::VisionAgents::Dispatch.new(capacity: 4)

dispatch.wait_for_call do |call|
  GetStream::VisionAgents::Agent.new(config: "support").join(call)
end

dispatch.wait_for_message do |message|
  next dispatch.answer(message) unless message.session_id.empty?

  dispatch.get_or_create_agent(message) { GetStream::VisionAgents::Agent.new(config: "support") }
          .reply(message)
end

dispatch.run
```

Every call and message carries a `work_id`, and when its handler returns or raises the worker
sends `done` for it, with the error message when it raised; work with no handler registered is
reported done with an error too. The handshake says `capacity`, `active` (calls and messages
still being handled) and `handles` (`call`, `message`, or neither).

A message arrives when no agent is running on its channel, and `get_or_create_agent` keeps one
agent per channel for the same reason. An agent whose `agent.yaml` says
`dispatch: {text: enabled}` hands what end users write to the worker instead, with
`message.session_id` and `message.request_id` set: `dispatch.answer(message)` has the model
answer it on that session, with the worker's own credential acting for `message.user_id`.
`get_or_create_agent` refuses such a message.

A worker can also run an agent's tools for every session under it, whoever opened it, such
as a conversation started from a browser. The tools are hosted under the agent's config name:

```ruby
support = GetStream::VisionAgents::Client.new.agent("support")
support.tools.register("lookup_order", description: "An order by id",
                       parameters: { type: "object", properties: { id: { type: "string" } } }) do |args|
  orders.find(args["id"])
end

dispatch = GetStream::VisionAgents::Dispatch.new
dispatch.host(support, tool_timeout: 30)
dispatch.run
```

A tool can say who runs it and how it is shown: `executor: "client"` for one a person's device
runs (this process still answers it, once the device has reported), and `display_title:` for
the words shown while it runs, such as `"Checking your order"`.

The tools are declared each time the router says it is ready, and each call runs on its own
thread. `tool_timeout` is how many seconds the router waits for one tool call; nil takes its default of two minutes. Hosting alone is enough
to `run`, and a router that refuses the tools ends `run` with the reason.

A dispatched call names the session it is for: `join(call)` opens it with voice, on the call
the caller is already in.

Ringing somebody is the other direction:

```ruby
agent.outbound_call(from: "+15551234567", to: "+15557654321") do |session|
  session.say("Hi, this is the clinic calling about your appointment.")
end
```

The router places the call for a session of its own, and the agent joins that session.

## An agent written down as a directory

```
agents/jean/
  agent.yaml            required: the name and what it runs on (llm, stt, tts, greeting, plugins, harness, tags, ...)
  instructions.md
  guardrail.md
  skills/think.md
  knowledge/pricing.md
  knowledge/urls.yaml   pages, each a url or a mapping with title, description, refresh_hours
  simulations/lunch.yaml
  .agent_sync           written by sync: the fingerprint last synced and when
```

```ruby
agent = GetStream::VisionAgents::Agent.new(folder: "agents/jean")
agent.sync
agent.join { ... }
```

`sync` stores the directory as a config named after it, in one request. A key `agent.yaml`
does not know is refused. `.agent_sync` records the fingerprint, which is the one the Go,
Python and JavaScript SDKs take, so syncing on every start only reads the config back when
nothing changed. What the code sets wins over what the directory says.

The harness is agent config, never session config: `harness:`, `skills:`, `sandbox:` and a
`pipeline: { subagent: }` given to `Agent.new` are written by `sync`, and a session only runs
them by starting from that config.

Each file in `simulations/` is a list of simulations (`name`, `scenario`, `assertion`, and
optionally `mode`, `variations`, `max_turns`, `caller_target`, `judge_target`, `caller_stt`,
`caller_tts`, `caller_voice`, `tags`). With the directory there, the config's simulations
become exactly that list, so an empty one deletes them; without it they are left alone.

```yaml
- name: lunch order with a change
  scenario: Order a turkey club, then swap it for a veggie wrap.
  assertion: The final order is one veggie wrap.
  variations: 3
```

```ruby
agent.knowledge.add_url("https://example.com/pricing", title: "Pricing", refresh_hours: 24)
agent.update_config(guardrail: File.read("guardrail.md"), visible_tools: ["athena_*"])
```

`add_url` adds a page to the agent's knowledge base, read again every `refresh_hours`, and
waits for it to be read. `update_config` changes only the fields it is given.

## Simulations

```ruby
simulation = api.simulations.create(name: "refund", config_id: config["id"],
                                    scenario: "Ask for a refund", assertion: "A refund is offered")
run = api.simulations.run(simulation["id"])
run = api.simulations.runs.get(run["id"]) while run["state"] == "running"
```

`create`, `get`, `list`, `update`, `delete` and `run` on `api.simulations`; `get`, `list` and
`cancel` on `api.simulations.runs`.

## Going back, and branching off

```ruby
agent.chat do |session|
  first = session.responses.create("Pick a number")
  session.responses.create("Double it")
  branch = session.fork(response_id: first, title: "asked again")
end

page = api.agent("support").sessions.query(user_id: "u1", state: "live", limit: 20)
page = api.agent("support").sessions.query(user_id: "u1", state: "live", cursor: page["next_cursor"]) if page["has_more"]
api.agent("support").sessions.search("billing")["items"]
```

`query` and `search` answer a page, `{items, has_more, next_cursor}`; pass `next_cursor` back
as `cursor` with the same filters for the next one. `query` narrows by `project_id`,
`user_id`, `modality`, `state` (`live` or `ended`) and `agent_id`; `search` takes the same
but `project_id`. `responses.list` and `responses.items.list` page the same way, and
`responses.items.each` follows the cursor itself.

A written conversation is kept in Stream Chat unless it is opened with `incognito: true`.
Neither can be rewound: the channel still holds the later turns, and an incognito one recorded
nothing to rebuild from. The router answers 400, and forking at the response is the way back.
`session.responses.rewind(response)` is for a call.

To keep the conversation and change it, from the next turn, for this session only:

```ruby
session.update(title: "Pricing", llm: "llm-thinking", thinking: "high")
api.agent("support").sessions.update(session_id, title: "Pricing")  # an ended session can still be renamed
```

`update` takes `title`, `description`, `custom`, `llm`, `stt`, `tts`, `sts`,
`voice`, `thinking`, `temperature`, `max_output_tokens` and `verbosity`; a field left out is
left as it is.

`session.close` stops a conversation and keeps everything it recorded and remembered.
Deleting is separate:

```ruby
session.delete                                   # the session, its turns and what it taught memory
session.delete_memories                          # only what it taught memory
api.agent("support").sessions.delete(session_id)
api.memories.truncate("user-42")                 # everything remembered about one user
```

## Guests

```ruby
guest = api.guest_user(name: "Visitor")          # id, token, expires_at
api.as_guest(guest).agent("support").sessions.query
api.claim_guest_user(guest["id"], "user-42")     # once they sign up
```

## The router

A router comes from the client, and which model answers lives in its config, never in the
call:

```yaml
# routers/healthcare/router.yaml
stt:
  target: en-low-latency
llm:
  target: llm-fast
tts:
  voice: Kore
```

```ruby
GetStream::VisionAgents::Router.sync("routers", client: api)
router = api.router("healthcare", tags: { env: "production" })

router.stt.realtime do |stt|
  stt.send_audio(pcm)
  stt.each { |frame| puts frame["text"] if frame["type"] == "transcript" }
end
router.tts.realtime { |tts| File.binwrite("hello.pcm", tts.speak("Hello")) }
router.llm.realtime { |llm| llm.respond("Say hi") { |delta| print delta } }

router.stt.recording("https://example.com/call.mp3", diarize: true)
router.search("perioperative antibiotic guidance", results: 5)
router.configure_stt(target: "en-low-latency", keyterms: ["Vision Agents"])
```

`realtime` takes only per-call overrides such as `diarize:`, `keyterms:` or `voice:`. Options
are checked against the spec's option blocks, so a misspelt one is refused here.

## Working on this gem

Build and test in the official Ruby image, with the gems in a named volume:

```bash
docker run --rm -v "$PWD/../..:/repo" -v va-ruby-bundle:/usr/local/bundle -w /repo/sdks/ruby \
  ruby:4.0 sh -c "bundle install && bundle exec rake"
```

`rake` checks the generated table is current, then runs the unit tests against a real HTTP
and websocket server on a local port. `script/generate` regenerates
`lib/getstream/vision_agents/generated/spec.rb` after the spec moves.

The live suite is opt-in and talks to a running router:

```bash
docker run --rm -v "$PWD/../..:/repo" -v va-ruby-bundle:/usr/local/bundle -w /repo/sdks/ruby \
  -e VISION_AGENTS_URL=http://host.docker.internal:8091 ruby:4.0 sh -c "bundle install && bundle exec rake live"
```

It holds real conversations, so it spends the deployment's token allowance.
