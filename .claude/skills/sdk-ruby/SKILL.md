---
name: sdk-ruby
description: How to build and extend the Ruby SDK in sdks/ruby. Read this before changing the client, the sockets, the agent, dispatch, folder sync or the router, or before reaching for a gem.
---

# Ruby SDK conventions

The per-language half of [sdk](../sdk/SKILL.md). It records the decisions
[`sdks/ruby`](../../../sdks/ruby) already follows, so a change lands consistently rather than
re-litigating them. Server side only: there is no Ruby client for devices.

Ruby 3.3 or newer (3.2 is end of life; 3.3 gets security fixes only), built and tested in the
official `ruby:4.0` image. Never install gems on the host.

## The gem

One gem, `getstream-vision-agents`, `require "getstream/vision_agents"`, module
`GetStream::VisionAgents`. Stream's own gems are `getstream-ruby` (`GetStreamRuby::Client`,
generated code under `GetStream::`), `stream-chat-ruby` (maintenance mode) and `stream-ruby`;
the name sits beside the current one without claiming its namespace.
`lib/getstream-vision-agents.rb` exists only so `Bundler.require` finds it.

```
lib/getstream/vision_agents.rb     requires everything, in order
  generated/spec.rb                OPERATIONS and SCHEMAS, from the spec. Never edit
  backend.rb client.rb             who is calling; one method per HTTP verb
  socket.rb                        TCP/TLS + websocket-driver, one reader thread
  session.rb tools.rb responses.rb a conversation, its tools, its turns
  agent.rb folder.rb knowledge.rb  Agent, Sessions, Sandbox; agent.yaml and .agent_sync
  inbound.rb dispatch.rb edge.rb   dispatch frames, the worker, Stream calls
  router.rb                        Router, the four realtime streams, recordings, search
```

## Dependencies

Two at runtime, and adding a third needs a reason written here.

- **`getstream-ruby ~> 12.1`**, Stream's server-side gem, only in `edge.rb` to create the call
  the backend joins (`client.video.get_or_create_call`). It pulls in Faraday; that is the
  price of using Stream's client rather than re-implementing call creation. It has no
  public user-token method, so tokens are signed in `Backend.sign`.
- **`websocket-driver ~> 0.8`**: the protocol only. The connection, TLS, reader thread and
  lifecycle are ours in `Socket`. Not `faye-websocket` (wants EventMachine), not
  `async-websocket` (a fiber stack the rest of the gem does not use), not
  `websocket-client-simple` (four more gems, callbacks on its own thread). Never frame
  websockets by hand.

What is not a dependency, and why:

- HTTP is **`Net::HTTP`**, as Stripe's gem does. The router is one host with JSON bodies;
  Faraday's middleware buys nothing here and its adapters are a support matrix.
- JWT is `OpenSSL::HMAC` plus unpadded URL-safe base64, because the gem only issues one kind
  of HS256 token and never verifies one. Reach for `jwt ~> 3.3` if that ever changes.
- JSON is `json`, YAML is `Psych.safe_load(text, aliases: false)`. Never `YAML.load`.

## Generated from the spec, not a generated client

`script/generate` reads `acceleration/api/openapi.yaml` and writes
`lib/getstream/vision_agents/generated/spec.rb`: every operation's method, path, path and
query parameters, body schema, whether it is a socket, and every object schema's property
names. `Client#get/post/put/patch/delete(template, path:, query:, body:)` look the template up
there, so a path, query key or body key the spec lacks is refused before a request is made,
and every operation is callable the day the spec gains it.

Not OpenAPI Generator's ruby client: it produces a model class per schema and a Faraday
client, neither of which the socket protocols, dispatch or folder sync can use, and Stream's
own `getstream-ruby` is made by an internal generator rather than that one. Answers stay
string-keyed hashes, whole, so a field added after the gem shipped still reaches the caller.
`Data.define` only for small values the gem constructs itself (`Participant`, `Event`,
`InboundMessage`, `Skill`, `Audio`).

```bash
docker run --rm -v "$PWD:/repo" -v va-ruby-bundle:/usr/local/bundle -w /repo/sdks/ruby ruby:4.0 \
  sh -c "bundle install && ruby script/generate"
```

`rake check` (part of the default task) fails when the committed table is stale. The router
option blocks (`SttOptions`, `TtsOptions`, `LlmOptions`, `StsOptions`, `SearchOptions`) are
checked against the same table.

`nil` means omit, everywhere: a key left nil is left out so the config or the router decides.
Never copy a schema default into a request.

## Concurrency

Threads, not fibers. One reader thread per socket; it never runs user code.

- A session answers `tool_call` on its own thread per call, at most 16 at once, with the slot
  taken before the thread starts. `tool_cancel` drops the answer; a Ruby thread cannot be
  killed safely, so the tool finishes and its result is not sent.
- Events are buffered, 256 deep, oldest dropped. Answering tools never depends on anybody
  reading `events`.
- Dispatch runs each call and message handler on its own thread. `capacity` bounds it: the
  router does not hand a full worker more work.
- Writes to a socket are serialised; the driver is only touched under its lock.

The Ruby spelling of `async with` is a block with `ensure`: `agent.join(call) { |session| }`,
`agent.chat { }`, `router.stt.realtime { }` close on every exit path and return the block's
value. Without a block the open object is returned and closing it is the caller's job.

No reconnection, on purpose: `respond` and `tool_result` are not idempotent and the protocol
has no sequence number to resume from. A socket that drops ends the session or `Dispatch#run`.

## Errors

`Error < StandardError`; `ConfigurationError` for anything refused before a request is made;
`RouterError` with `status` (0 means it never arrived), `operation` (the spec's operation id),
`body` and `retry_after`; `SocketClosedError` for writing to a closed socket. A refused
websocket upgrade is a `RouterError` with the handshake's status. Never `rescue Exception`.

## Behaviour that is easy to get wrong

- The session request carries `agent:` (the config name) and only what the code set. The
  stored config decides the rest.
- `memory_filter[:user_id]` goes to `memory.user_id`; everything else to `memory.filter`, as
  strings. `cost_tracking` goes to `tags` on the session, on `placeCall`, and on sync.
- `Agent#sync` ports Go's `Hash`/`fingerprint` byte for byte (`Folder.fingerprint`); the
  fixture in `folder_test.rb` pins `02a7b2c8428f31e3a2b93ca2f5a6ec70`. With a subagent or cost
  labels set the hash is re-fingerprinted with `map[k:v ...]`, which is how Go's `fmt.Sprint`
  writes a map. A matching `.agent_sync` reads the config back instead of posting.
- Outbound: create the Stream call, `POST /v1/phone/calls`, then open the session with
  `navigating: true` and `phone.vendor_call_id`. In that order.
- An `InboundCall` is joined as it arrived, with `phone.number` set to the number rung.
- Rewind is 204 and refused with 400 on a persisted conversation; fork with `response_id`
  instead. A response id is not the `turn_id` socket frames carry.
- Dispatch `ping.at` is a number the router echoes, not a timestamp.
- A modality stream's start frame needs a top-level `target` or `config_id`; the router does
  not read the block's `target`, so `Router#open` lifts it.

## Tests

Minitest, `rake test`. Never mock and never assert that a method was called; assert on what
reached the wire and what came back.

`test/support/local_router.rb` is a real server on `127.0.0.1:0`: a `TCPServer` that parses
HTTP, answers scripted routes, records every request, and upgrades websockets with
`WebSocket::Driver.server` so a test can script frames and read what the client sent. Not
WEBrick (a dev dependency that cannot hand over the socket for an upgrade). `getstream-ruby`
is pointed at the same server with `base_url:`, so creating a call is tested too.

Wait with a deadline (`eventually`, `receive(timeout:)`), never a bare `sleep` guess.

The live suite, `test/live`, runs with `rake live` and skips unless `VISION_AGENTS_URL` is
set (customer `VISION_AGENTS_CUSTOMER_ID`, default `examples`). From Docker the local router
is `http://host.docker.internal:8091`. It holds real conversations and spends tokens.

```bash
docker run --rm -v "$PWD:/repo" -v va-ruby-bundle:/usr/local/bundle -w /repo/sdks/ruby ruby:4.0 \
  sh -c "bundle install && bundle exec rake"
docker run --rm -v "$PWD:/repo" -v va-ruby-bundle:/usr/local/bundle -w /repo/sdks/ruby \
  -e VISION_AGENTS_URL=http://host.docker.internal:8091 ruby:4.0 sh -c "bundle install && bundle exec rake live"
```

Run from the repo root. The full image, not `-slim`: `websocket-driver` has a C extension.

## Publishing

`rubygems_mfa_required` is set in the gemspec. Publish with RubyGems trusted publishing from
CI (`rubygems/release-gem`), never a stored API key. `Gemfile.lock` is not committed for a
library.

## Reviewing a change

Reject it if it:

- hand-edits `generated/spec.rb`, or adds a per-endpoint method where `Client#get` would do;
- adds a runtime gem without a reason here, or reaches for Faraday outside `edge.rb`;
- runs user code on a socket's reader thread, or opens a second reader on one socket;
- replays `respond` or `tool_result`, or reconnects a session socket;
- sends a schema default, or a key the code did not set;
- symbolises wire hashes, or drops a frame kind it does not know;
- changes the folder fingerprint without the pinned hash still passing;
- uses `YAML.load`, `rescue Exception`, or a mock.
