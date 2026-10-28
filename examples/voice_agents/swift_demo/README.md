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

The app talks to Stream's hosted router, which `VisionAgents(apiKey:)` reaches unless told
otherwise, so all it takes is a Stream app: its key and its secret. Leave
`STREAM_ACCELERATION_URL` and `STREAM_ACCELERATION_CUSTOMER_ID` unset, or the Go programs below
go to that router instead.

Store the agent. This is the Go SDK doing the things a phone is not allowed to do:

```bash
cd examples/voice_agents/swift_demo
STREAM_API_KEY=<your Stream app's key> \
STREAM_API_SECRET=<its secret> \
go run ./configure
```

The hosted router keeps the agent under the app the key names, so put the same key in
`Demo.streamAPIKey` in `app/SwiftDemo/Demo.swift`.

```
agent      swift_demo (config 5740714b87c00f458859e694cb87f09d)
skill      refund_decision (20s)
knowledge  policy.md (774 characters)
```

`Demo.agentName` in `app/SwiftDemo/Demo.swift` is already `swift_demo`, the name in
`agent.yaml`. The app is told which agent it talks to rather than looking it up, because
reading the configs is server-side only.

Leave the other backend running, which signs the Stream token the phone is its user with. It
takes the same app's secret:

```bash
STREAM_API_SECRET=<its secret> go run ./backend
```

Then open `app/SwiftDemo.xcodeproj` in Xcode 27 and run it on a simulator. Ask about an order
in the Chat tab or tap talk in the Voice tab. Xcode 27 because the app shows the conversation
with Stream Chat's AI components (`StreamChatAI`), which need it to add with SPM. The app takes
`stream-chat-swift` from its `develop` branch, because the voice glow (`AIVoiceGlowView`) is not
in a released `StreamChatAI` yet.

The Chat tab asks for `xai/grok-4.7`, a model that streams its reasoning; a call keeps the
agent's faster model.

### On a local router

The hosted router is deployed less often than `accelerate` moves, so a router change is tried
on one built from this checkout. Start it, from the repo root, with `ROUTER_AUTH_MODE=proxy` in
the repo `.env` so it trusts the customer and user ids it is given. Not `noauth`, which takes
every caller for the backend: the phone's user would never join its own chat channel.

```bash
docker compose up --build
```

Run `configure` with `STREAM_ACCELERATION_URL=http://localhost:8080` and
`STREAM_ACCELERATION_CUSTOMER_ID=examples` in place of the key and secret, `examples` being the
customer `compose.yaml` builds the dashboard with. In `Demo.swift`, build the app's
`VisionAgents` with
`VisionAgents(url: URL(string: "http://localhost:8080")!, customerID: "examples", apiKey: streamAPIKey)`,
with the `STREAM_API_KEY` the router runs with, and run `backend` with that app's secret. On a
device, put your Mac's address on the network in place of `localhost`.

The router then needs `XAI_API_KEY` for the Chat tab, and `TURBOPUFFER_API_KEY` for knowledge.
Without it the sync says so and carries on: the agent keeps its instructions and its skill and
loses the returns policy, and the refund skill then cannot say a refund is owed.

## What each half is allowed to do

`configure` signs its requests with the app's secret, in a token that names no user, which the
router reads as the app's own backend, so it may write configs, define skills and ingest
knowledge. The app's token names its user, which declares it a device, and the router answers
403 for all three. `backend` never talks to the router: it holds the Stream secret, which is all
signing a user's token takes.

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
  what the video SDK joins — they are not the same id. The agent is shown as Stream's voice
  glow (`AIVoiceGlowView`): connecting while the call is joined, thinking while it answers, and
  moving with whoever is talking, metered from the call's audio, with what is said captioned
  underneath. The style button at the top right of the Voice tab opens a sheet showing the glow
  and Stream's voice wave (`AIVoiceWaveView`), each moving, to pick from.
- **`lookup_order`** is a tool the model calls that runs in `Demo.swift`. Its data never leaves
  the phone; the agent asks, and only sees the answer. Ask about order `A-1042` or `A-1043`.
  `configure` names it in `visible_tools`, which is what shows its call on the reply.
- **`issue_refund`** asks first. Ask for a refund on `A-1042`: once the refund skill says it is
  owed, the agent calls it, and the call waits for you. In Chat the question is on the reply
  (`AIToolApprovalView`); on a call it is a card under the glow (`AIToolApprovalCard`).
  Refund and the tool runs; Not now and the agent is told you declined.

## No sign-in

Three constants in `app/SwiftDemo/Demo.swift`: this backend's URL, the Stream key and the
agent's name. That is the whole configuration, because the hosted router knows the app by the
key and the user by the token `backend` signs, and `backend` signs for whoever the app says it
is. A real app signs its person in first and its backend signs for them alone; nothing else in
the app would change.

The app is the user `demo-caller`, and a session belongs to whoever opened it, so it reaches
its own sessions and nothing else. The router verifies the token, so that boundary is drawn
around the `user_id` it names, which is what stops one person reading another's conversation.

The router is reached over HTTPS; `backend` runs on your Mac. The simulator reaches your Mac's
localhost, so it works as it stands. On a device, put your Mac's address on the network in
`Demo.backendURL`; `NSAllowsLocalNetworking` in `Info.plist` is what lets plain HTTP through
to it.
