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
to create the calls the backend joins, and `websocket-driver` for the sockets.

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

agent.join("call-1") do |session|
  puts agent.monitor_url
  agent.responses.create("greet the user in one short sentence")
end
```

`join` creates the Stream call, has the backend join it, and waits for somebody else to be
in it. The block form closes the session when the block leaves, after waiting for the call to
end unless `wait_for_end: false`. Without a block the open session is returned.

The session starts from the stored config, and only what the code sets is sent over it.
`cost_tracking` labels every request the session makes; `memory_filter` says who the memories
are about, under `user_id`, and what else narrows recall.

`chat` holds the same conversation in writing: the same instructions, skills and knowledge,
nothing transcribed or spoken.

```ruby
agent.chat(persist: true) { |session| session.responses.create("What changed in v3?") }
```

## Credentials

| Credential | Who it is |
| --- | --- |
| `customer_id` | a router with nothing in front of it, which is a laptop |
| `api_key` + `api_secret` | a process you run |

```ruby
api = GetStream::VisionAgents::Client.new(api_key: ENV["STREAM_API_KEY"], api_secret: ENV["STREAM_API_SECRET"])
agent = api.agent("support", cost_tracking: { env: "production" })
```

`url` falls back to `STREAM_ACCELERATION_URL`, then `http://localhost:8080`, and the rest to
`STREAM_ACCELERATION_CUSTOMER_ID`, `STREAM_API_KEY` and `STREAM_API_SECRET`. Behind Stream's
authenticating proxy pass `authenticate: true` or set `STREAM_ACCELERATION_AUTHENTICATE`.
Creating a call needs `STREAM_API_KEY` and `STREAM_API_SECRET` whichever way the router is
reached.

## Every endpoint, checked against the spec

```ruby
api.get("/v1/agents/configs", query: { name: "support" })
api.post("/v1/search", body: { query: "what changed in v3" })
api.delete("/v1/agents/sessions/{id}", path: { id: "sess_1" })
```

One method per HTTP method. The path is the spec's own template, looked up in a table
generated from `acceleration/api/openapi.yaml`, so a path, query parameter or body key the
spec does not have is refused before anything is sent. Answers are the router's JSON as
string-keyed hashes. A failure raises `RouterError` with the status, the operation id and
what the router said; status 0 means the request never arrived.

## Dispatch

The worker connects out and the router pushes calls and messages down the socket, so nothing
you run has to be publicly reachable.

```ruby
dispatch = GetStream::VisionAgents::Dispatch.new(capacity: 4)

dispatch.wait_for_call do |call|
  GetStream::VisionAgents::Agent.new(config: "support").join(call)
end

dispatch.wait_for_message do |message|
  dispatch.get_or_create_agent(message) { GetStream::VisionAgents::Agent.new(config: "support") }
          .reply(message)
end

dispatch.run
```

A call handler that returns accepts the call; one that raises rejects it with the message as
the reason. A message only arrives when no agent is running on its channel, and
`get_or_create_agent` keeps one agent per channel for the same reason.

Ringing somebody is the other direction:

```ruby
agent.outbound_call(from: "+15551234567", to: "+15557654321") do |session|
  session.say("Hi, this is the clinic calling about your appointment.")
end
```

## An agent written down as a directory

```
agents/jean/
  agent.yaml            required: the name and what it runs on (llm, stt, tts, tags, ...)
  instructions.md
  guardrail.md
  skills/think.md
  knowledge/pricing.md
  knowledge/urls.yaml
  .agent_sync           written by sync: the fingerprint last synced and when
```

```ruby
agent = GetStream::VisionAgents::Agent.new(folder: "agents/jean")
agent.sync
agent.join("call-1") { ... }
```

`sync` stores the directory as a config named after it, in one request. A key `agent.yaml`
does not know is refused. `.agent_sync` records the fingerprint, which is the one the Go,
Python and JavaScript SDKs take, so syncing on every start only reads the config back when
nothing changed. What the code sets wins over what the directory says.

```ruby
agent.knowledge.add_url("https://example.com/pricing", title: "Pricing")
```

adds a page to the agent's knowledge base and waits for it to be read.

## Going back, and branching off

```ruby
agent.chat(persist: false) do |session|
  first = session.responses.create("Pick a number")
  session.responses.create("Double it")
  session.responses.rewind(first)
  branch = session.fork(response_id: first, title: "asked again")
end

api.agent("support").sessions.search("billing")
```

A conversation kept in Stream Chat cannot be rewound, because the channel still holds the
later turns: the router answers 400, and forking at the response is the way back.

## Guests

```ruby
guest = api.guest_user(name: "Visitor")          # id, token, expires_at
api.as_guest(guest).get("/v1/agents/sessions")
api.claim_guest_user(guest["id"], "user-42")     # once they sign up
```

## The router

```ruby
router = GetStream::VisionAgents::Router.new("healthcare", tags: { env: "production" })

router.stt.realtime(languages: ["en"]) do |stt|
  stt.send_audio(pcm)
  stt.each { |frame| puts frame["text"] if frame["type"] == "transcript" }
end
router.tts.realtime(voice: "Kore") { |tts| File.binwrite("hello.pcm", tts.speak("Hello")) }
router.llm.realtime(target: "llm-fast") { |llm| llm.respond("Say hi") { |delta| print delta } }

router.stt.recording("https://example.com/call.mp3", diarize: true)
router.search("perioperative antibiotic guidance", results: 5)
router.configure_stt(providers: %w[deepgram], profanity_filter: true)
GetStream::VisionAgents::Router.sync("routers")
```

Options are checked against the spec's option blocks, so a misspelt one is refused here.

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
