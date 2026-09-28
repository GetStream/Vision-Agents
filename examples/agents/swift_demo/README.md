# Swift demo

One agent, configured by Go and talked to by an iOS app. It is the whole split the SDKs assume:
a backend decides what an agent is, and a phone only holds conversations with it.

```
instructions.md      what the agent is told
skills/              what it can go away and think about
knowledge/           what it can look things up in
configure/main.go    the backend half: pushes all of the above
app/                 the phone half: chat, voice, and two tools that run on the device
```

## Run it

Start the router and its Postgres and Redis, from the repo root:

```bash
docker compose up --build
```

Store the agent. This is the Go SDK doing the things a phone is not allowed to do:

```bash
cd examples/agents/swift_demo
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

Then open `app/SwiftDemo.xcodeproj` and run it on a simulator. Pick `swift_demo`, and ask about
an order in the Chat tab or tap talk in the Voice tab.

Knowledge needs an embeddings provider. Without one the sync says so and carries on: the agent
keeps its instructions and its skill and loses the returns policy.

## What each half is allowed to do

`configure` sends only `X-Customer-Id`, which a router with no proxy in front of it reads as a
backend, so it may write configs, define skills and ingest knowledge. The app sends
`Stream-Auth-Type: jwt` as well, which declares it a device, and the router answers 403 for all
three. Nothing in the Swift SDK can even ask: `sdks/swift/generate.py` refuses to generate a
method for an operation marked `x-server-side-only`.

Two calls do the configuring, and the order matters:

- `DefineAgent` says what *runs* the agent — which models transcribe, answer, speak and think.
- `SyncAgent` pushes what the agent *knows* — the three things in this directory. It keeps the
  models it finds, which is why it goes second: storing a config replaces it, so doing these
  the other way round would wipe the instructions.

`SyncAgent` also fingerprints the directory, so running it twice with nothing changed does
nothing.

A knowledge base needs an embeddings provider, and a router started five minutes ago may have
none. Rather than storing an agent that cannot answer the one question this demo is about,
`configure` writes `knowledge/policy.md` into the prompts instead — the agent's and the skill's,
because a skill runs on the subagent under its own instructions and never sees the agent's. Set
`TURBOPUFFER_API_KEY` and it is a real knowledge base, looked up a passage at a time.

Re-run it after changing `acceleration/internal/routing/router.yaml` too, not only after
changing this directory: a stored config names models, and a model that has been renamed or
retired there is a 400 in the middle of a call.

## What the app shows

- **The agent picker** lists `listAgentConfigs`, one of the reads a device may make.
- **Chat** is a text session: no call is joined, nothing is transcribed or spoken, and the
  replies still come through the same model with the same instructions, skills and knowledge a
  call would have had. They arrive one delta at a time over the session socket.
- **Voice** starts a session, which puts the agent on a call, mints a token for that call, and
  joins it from the device. Note that `Session.id` addresses the router and `Session.call_id` is
  what the video SDK joins — they are not the same id.
- **`lookup_order`** is a tool the model calls that runs in `Demo.swift`. Its data never leaves
  the phone; the agent asks, and only sees the answer. Ask about order `A-1042` or `A-1043`.
- **`refund_order`** is the same kind of tool with somebody in the loop. It is declared with an
  `approval` question, so the SDK does not run it when the model asks: a card appears with
  *Refund 78.00 for order A-1042?* and nothing happens until it is answered. The agent is
  holding its turn open on the router the whole time, so what it says next is what actually
  happened — approve it and it confirms the refund, decline it and it says nothing was
  refunded. Take your time: the SDK tells the router a person is being asked, so the wait is a
  person's rather than the seconds a tool gets, and if nobody ever answers the agent says the
  refund was not approved and the card comes down. It works the same in the Voice tab,
  mid-call.
- **Events** in the Chat toolbar is the same conversation as AG-UI protocol events, off
  `session.aguiEvents()`. The approval is `RUN_FINISHED` with an interrupt on it, and
  approving is what starts the run that carries `TOOL_CALL_RESULT`.

## Try the approval

```
Refund order A-1042.
```

The agent looks the order up, hands the money decision to the `refund_decision` skill, tells
you what it worked out, and then asks to be allowed to issue it. The returns policy says an
unopened order goes back for what was paid less return postage, so `A-1042` is refundable and
`A-1043`, which is worn, is not: ask for that one and the skill says no and nothing is ever
asked of you.

The orders in `Demo.swift` say how long ago they were delivered rather than on what date,
because nothing tells the model what today is: given a date and a thirty-day window it asks you
what the date is instead of deciding.

## No auth

Two constants in `app/SwiftDemo/Demo.swift`: the router's URL and the customer id. That is the
whole configuration, because the router is running in the mode where it trusts the customer id
it is given. In front of a real deployment those would come from your own backend along with a
token, and nothing else in the app would change.

The simulator reaches your Mac's localhost, so it works as it stands. On a device, put your
Mac's address on the network in `Demo.routerURL`; `NSAllowsLocalNetworking` in `Info.plist` is
what lets plain HTTP through to it.
