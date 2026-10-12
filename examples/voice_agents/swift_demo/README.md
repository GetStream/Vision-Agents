# Swift demo

One agent, configured by Go and talked to by an iOS app. It is the whole split the SDKs assume:
a backend decides what an agent is, and a phone only holds conversations with it.

```
agent.yaml           what the agent is called
instructions.md      what the agent is told
skills/              what it can go away and think about
knowledge/           what it can look things up in
configure/main.go    the backend half: pushes all of the above, then exits
backend/main.go      the backend half that stays up: signs the Stream token the phone is its user with
app/                 the phone half: chat, voice, and tools that run on the device
```

## Run it

Start the router and its Postgres and Redis, from the repo root, with `ROUTER_AUTH_MODE=proxy`
in the repo `.env` so it trusts the customer and user ids it is given. Not `noauth`, which takes
every caller for the backend: the phone's user would never join its own chat channel.

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

`Demo.agentName` in `app/SwiftDemo/Demo.swift` is already `swift_demo`, the name in
`agent.yaml`. The app is told which agent it talks to rather than looking it up, because
reading the configs is server-side only.

Leave the other backend running, which signs the Stream token the phone is its user with. It
takes the secret of the Stream app the router runs in, and `Demo.streamAPIKey` takes that app's
`STREAM_API_KEY`:

```bash
STREAM_API_SECRET=<the router's STREAM_API_SECRET> go run ./backend
```

Then open `app/SwiftDemo.xcodeproj` in Xcode 27 and run it on a simulator. Ask about an order
in the Chat tab or tap talk in the Voice tab. Xcode 27 because the app shows the conversation
with Stream Chat's AI components (`StreamChatAI`), which need it to add with SPM.

The Chat tab asks for `xai/grok-4.7`, a model that streams its reasoning, so the router needs
`XAI_API_KEY`; a call keeps the agent's faster model.

Knowledge needs `TURBOPUFFER_API_KEY` on the router. Without it the sync says so and carries on:
the agent keeps its instructions and its skill and loses the returns policy, and the refund
skill then cannot say a refund is owed.

## What each half is allowed to do

`configure` sends only `X-Customer-Id`, which a router with no proxy in front of it reads as a
backend, so it may write configs, define skills and ingest knowledge. The app sends
`Stream-Auth-Type: jwt` as well, which declares it a device, and the router answers 403 for all
three. `backend` never talks to the router: it holds the Stream secret, which is all signing a
user's token takes.

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
  call would have had. The session is kept in a Stream Chat channel, which the router writes
  and the app shows with Stream's AI components: the reply streaming in, the model's reasoning
  as it thinks (`StreamingReasoningView`, reassembled from the live updates by
  `LiveReasoning`), each tool call as it runs (`AIToolCallView`), and what the agent is doing
  before it answers (`AITypingIndicatorView`). `AIComposerView` asks, dictates, and stops a
  reply; `SuggestionsView` starts a conversation.
- **Voice** starts a session, which puts the agent on a call, and joins it over Stream Video as
  the user `setUser` named, with the token `backend` signed. Note that `Session.id` addresses the router and `Session.call_id` is
  what the video SDK joins — they are not the same id.
- **`lookup_order`** is a tool the model calls that runs in `Demo.swift`. Its data never leaves
  the phone; the agent asks, and only sees the answer. Ask about order `A-1042` or `A-1043`.
  `configure` names it in `visible_tools`, which is what shows its call on the reply.
- **`issue_refund`** asks first. Ask for a refund on `A-1042`: once the refund skill says it is
  owed, the agent calls it, and the call waits for you. In Chat the question is on the reply
  (`AIToolApprovalView`); on a call it is a card over the transcript (`AIToolApprovalCard`).
  Refund and the tool runs; Not now and the agent is told you declined.

## No auth

Five constants in `app/SwiftDemo/Demo.swift`: the router's URL, this backend's URL, the
customer id, the Stream key and the agent's name. That is the whole configuration, because
the router is running in the mode where it trusts the customer and user ids it is given, and
the key is Stream Chat's and Stream Video's alone. In front of a real deployment the router is
reached by the key and the same token, and nothing else in the app would change.

The app is the user `demo-caller`, and a session belongs to whoever opened it, so it reaches
its own sessions and nothing else. A deployment verifying tokens draws the same boundary around
the `user_id` its token names, which is what stops one person reading another's conversation.

The simulator reaches your Mac's localhost, so it works as it stands. On a device, put your
Mac's address on the network in `Demo.routerURL`; `NSAllowsLocalNetworking` in `Info.plist` is
what lets plain HTTP through to it.
