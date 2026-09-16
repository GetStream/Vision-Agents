# Swift demo

One agent, configured by Go and talked to by an iOS app. It is the whole split the SDKs assume:
a backend decides what an agent is, and a phone only holds conversations with it.

```
agent.yaml           what the agent is called
instructions.md      what the agent is told
skills/              what it can go away and think about
knowledge/           what it can look things up in
configure/main.go    the backend half: pushes all of the above, then exits
backend/main.go      the backend half that stays up: mints the token a phone joins a call with
app/                 the phone half: chat, voice, and a tool that runs on the device
```

## Run it

Start the router and its Postgres and Redis, from the repo root:

```bash
docker compose up --build
```

Store the agent. This is the Go SDK doing the things a phone is not allowed to do:

```bash
cd examples/voice_agents/swift_demo
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
go run ./configure
```

A config belongs to one customer, and `compose.yaml` builds the dashboard with
`NEXT_PUBLIC_CUSTOMER_ID: examples` — baked in at build time, since Next inlines a
`NEXT_PUBLIC_` variable into the bundle. So store the agent under `examples` and it shows up
on the dashboard; store it under anything else and the app will find it but
[the agents page](http://localhost:3000/agents) will not. `Demo.customerID` in
`app/SwiftDemo/Demo.swift` has to match whatever you use here.

```
agent      swift_demo (config 581427bbd7a40e71d6c049e225a59ca0)
skill      refund_decision (20s)
knowledge  policy.md (774 characters)
```

Put that config id in `Demo.agentID` in `app/SwiftDemo/Demo.swift`. The app is told which
agent it talks to rather than looking it up, because reading the configs is server-side only.

Leave the other backend running, which is what mints the token the phone joins a call with:

```bash
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
go run ./backend
```

Then open `app/SwiftDemo.xcodeproj` and run it on a simulator. Ask about an order in the Chat
tab or tap talk in the Voice tab.

Knowledge needs an embeddings provider. Without one the sync says so and carries on: the agent
keeps its instructions and its skill and loses the returns policy.

## What each half is allowed to do

`configure` and `backend` send only `X-Customer-Id`, which a router with no proxy in front of
it reads as a backend, so they may write configs, define skills, ingest knowledge and mint call
tokens. The app sends `Stream-Auth-Type: jwt` as well, which declares it a device, and the
router answers 403 for all four.

The router is server-side only by default, and opens five operations: search, and opening,
listing, closing and watching a session. That is everything the app does. Nothing in the Swift
SDK can even ask for the rest — `sdks/swift/generate.py` refuses to generate a method for an
operation the spec does not mark `x-client-accessible`.

Two calls do the configuring, and the order matters:

- `DefineAgent` says what *runs* the agent — which models transcribe, answer, speak and think.
- `SyncAgent` pushes what the agent *knows* — the three things in this directory. It keeps the
  models it finds, which is why it goes second: storing a config replaces it, so doing these
  the other way round would wipe the instructions.

`SyncAgent` also fingerprints the directory, so running it twice with nothing changed does
nothing.

## What the app shows

- **Chat** is a text session: no call is joined, nothing is transcribed or spoken, and the
  replies still come through the same model with the same instructions, skills and knowledge a
  call would have had. They arrive one delta at a time over the session socket.
- **Voice** starts a session, which puts the agent on a call, asks `backend` for a token for
  that call, and joins it. Note that `Session.id` addresses the router and `Session.call_id` is
  what the video SDK joins — they are not the same id.
- **`lookup_order`** is a tool the model calls that runs in `Demo.swift`. Its data never leaves
  the phone; the agent asks, and only sees the answer. Ask about order `A-1042` or `A-1043`.

## No auth

Four constants in `app/SwiftDemo/Demo.swift`: the router's URL, this backend's URL, the
customer id and the agent id. That is the whole configuration, because the router is running in
the mode where it trusts the customer id it is given. In front of a real deployment the
customer id would come from your own backend along with a token, and nothing else in the app
would change.

Naming no user makes this app an anonymous caller, and a session belongs to whoever opened it,
so it reaches its own sessions and nothing else. A deployment verifying tokens gets the same
boundary drawn around the `user_id` its token names instead, which is what stops one person
reading another's conversation.

The simulator reaches your Mac's localhost, so it works as it stands. On a device, put your
Mac's address on the network in `Demo.routerURL`; `NSAllowsLocalNetworking` in `Info.plist` is
what lets plain HTTP through to it.
