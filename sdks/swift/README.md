# Swift SDK

One iOS package for talking to an agent from a phone, `core/`, whose module is
`VisionAgentsCore`: the generated client, the session socket and the conversation state, the
Stream Chat channel a text conversation is kept in, and `VoiceSession`, which joins the agent's
call over Stream Video. It depends on `stream-chat-swift` 5.13+ (`StreamChat`),
`stream-video-swift` (`StreamVideo`) and the `StreamCore` both of them stand on. `User` and
`RawJSON` are StreamCore's, and `import VisionAgentsCore` brings them in.

It has no views. The views are Stream Chat's AI components, the `StreamChatAI` library of the
same `stream-chat-swift` package, which render what the router writes into a conversation's
channel: the reply streaming in, each round of the model's reasoning, each tool call, and the
question a tool asks before it runs. Add `StreamChatAI` to the app target beside
`VisionAgentsCore`; it needs Xcode 27 to add with SPM.

iOS 15 is the floor. `AgentSession` and `VoiceSession` are `ObservableObject`s, so a view holds
one with `@StateObject` or `@ObservedObject`, and observing a `VoiceSession` follows its
conversation too. It is iOS only, because Stream Video is and Stream Chat's macOS build does
not compile under Swift 6.

An app adds it with Swift Package Manager from this repository,
`https://github.com/GetStream/Vision-Agents.git`, and takes the `VisionAgentsCore` product. Its
manifest is the `Package.swift` at the root of the repository, the only place SwiftPM looks, and
its versions are the repository's `v` tags, which release the Python packages too. The tags from
before the first one with Swift in it (0.7.0 or later) have no manifest, so keep them out of the
range. SwiftPM clones the whole repository the first time, about 300 MB, and only fetches what
is new after that. The examples reach the package by path, so they build against the checkout.

## Using them

```swift
let agents = VisionAgents(apiKey: "your_api_key")
agents.setUser(User(id: "jlahey")) { try await yourBackend.agentToken() }

// In writing. No call is joined, nothing is transcribed or spoken.
let session = try await agents.agent("myagent").sessions.create()
let turn = try await session.responses.create("What are your opening hours?")
// session.responses.items(responseID: turn.id) reads back what the agent did.

// Or watch it arrive: session.turns grows as the reply streams in, and every event is on
// session.events(), an AsyncStream of its own for each caller.
await session.start()
_ = try await session.responses.create("And on Sundays?")

// Out loud. The agent joins a call and so does this device.
let voice = try await VoiceSession.start(agents: agents, agent: "myagent")
await voice.join()

// The Stream Chat channel the text session is kept in.
let channel = try await session.chat()
```

`VisionAgents(apiKey:)` reaches Stream's hosted router in us-east4; pass `url:` for another router. For another
region, pass `url: try Backend.regionURL("europe-west4")`. The token
comes from a closure because your backend mints it and it expires: it is asked for once, then
again after a 401. The key goes in the query string and the token in the `Authorization`
header, never in a URL. A router running locally with nothing in front of it is reached by
customer id instead: `VisionAgents(url: URL(string: "http://localhost:8080")!, customerID: "acme")`.

Stream is set up once. The key, the user and the token from `setUser` are the one identity for
the router, Stream Chat and Stream Video, so `join()` and `chat()` need nothing more. Each
builds one client per key and user, connected as that user and shared by every session, and
asks for a fresh token when Stream says the one it has expired. `agents.disconnect()` closes
them; closing a session does not. An app that already has a `ChatClient` or a `StreamVideo`
hands it over instead, and it is used rather than a second one, and never disconnected here:

```swift
agents.use(chatClient)
agents.use(streamVideo)
```

One connected as somebody other than the user `setUser` named is refused. Beside a customer id,
`apiKey:` is Stream's alone: the router is still reached by customer id, and chat and video
connect with the key.

### Showing a conversation

A text session is kept in its channel, which the router writes: the person's message, then the
reply, marked `ai_generated`, with `generating` true until it is done and its steps as
attachments in the order they happened. Stream's AI components render all of it:

```swift
import StreamChatAI

let parts = AIMessagePart.parts(from: message.allAttachments.map { ($0.type.rawValue, $0.payload) })
VStack(alignment: .leading) {
    AIMessagePartsView(parts: parts) { part in
        if let step = part.reasoning {
            StreamingReasoningView(part: step, text: reasoning[message: message.id, step: step.id])
        } else {
            AIMessagePartView(part: part, approver: approver)
        }
    }
    StreamingMessageView(content: message.text, isGenerating: message.extraData["generating"]?.boolValue == true)
}
```

A step keeps only the opening of its reasoning. The whole of it comes with the reply's live
updates, a window at a time, and `LiveReasoning` puts it back together: read every
`MessageUpdatedEvent` into it, from `chatClient.eventsController()`, and look a step up by its
reply and its id: every reply numbers its steps from `r1`. The same events controller carries
`AIIndicatorUpdateEvent`, which says what the agent is doing before its answer starts, for
`AITypingIndicatorView`.

Ask with `session.responses.create`, from `AIComposerView`'s send; its stop button is
`session.interrupt()`. Start the session first, so its socket runs this phone's tools.

### A tool that runs on the phone

The agent runs in the backend; a tool you give it runs here. That is the point — it can read
what only the device knows, and the agent only ever sees the answer.

```swift
let lookup = AgentTool(
    name: "lookup_order",
    description: "Look up one of the caller's orders by its order number.",
    parameters: .strings(["order_id": "the order number"], required: ["order_id"]),
    executor: .client,
    displayTitle: "Looking up your order"
) { arguments in
    await Orders.local.find(arguments["order_id"]?.stringValue ?? "")
}

var options = SessionOptions(agent: "myagent")
options.tools = [lookup]
let session = try await agents.sessions.create(options)
await session.start()   // the socket is what carries tool calls to this device
```

`executor: .client` shows the people in a persistent conversation that a device is running it,
and `displayTitle` is what the reply's tool attachment says it is doing.

### A tool that shows the model a picture

Return a `ToolOutput` to answer with pictures beside the words, each a data URI or an https URL:

```swift
let snapshot = AgentTool(name: "photograph_room", description: "Take a photo of the room.") { _ in
    ToolOutput(
        text: "the room, just now",
        images: [ImageSource(url: "data:image/jpeg;base64," + jpeg.base64EncodedString())])
}
```

On a call the router hands the pictures to the agent's `vision` skill. In a chat, which has no
such worker, a model that sees images is shown them itself, with its next reply alone; one that
reads only text is told the tool failed. One socket message holds at most 5 MB, so shrink a
photo first.

### Showing the agent the camera

On a call, turn the camera on and the agent can look through it:

```swift
await voice.join(camera: true)          // or later: await voice.setCameraEnabled(true)
```

The camera is the back lens. Its frames are sampled twice a second through the call's video
filter, which leaves each one as it was, and the last four seconds are kept. When the agent's
`vision` skill needs to see, the router asks for what the camera showed at the moment the
question was taken (`get_video_frames`), and `voice.session` answers with those frames. Nothing
is declared as a tool, so the model is never offered it, and a camera turned off forgets what it
showed. The agent needs the `vision` skill, and the call a subagent to run it on. The call's video
filter is this one, so an app's own is replaced while the camera is on.

### A tool that asks first

Give a tool an `approval` and each call waits for the person whose message it answers, in
`session.approvals`, until `decide` answers it. Allowed, the tool runs; declined, it never does,
and the model is told so.

```swift
let refund = AgentTool(
    name: "issue_refund",
    description: "Refund an order to the card it was paid with.",
    parameters: .strings(["order_id": "the order", "reason": "why"], required: ["order_id", "reason"]),
    approval: .init(title: "Refund this order?", reasonArgument: "reason", allowTitle: "Refund")
) { arguments in
    try await Refunds.issue(arguments["order_id"]?.stringValue ?? "")
}

let approver = AIToolApprover(userID: user.id, clientID: AIClientIdentity.installID) { call, allowed in
    try await session.decide(call.id, allowed: allowed)
}
```

In a conversation kept in Stream Chat the call's step carries the question, and
`AIMessagePartView` asks it with `AIToolApprovalView` for whoever the approver says is signed in.
On a call there is no channel: show `session.approvals` yourself, with `AIToolApprovalCard`.
The router addresses a client tool's question to the install the message came from, which a
device cannot name yet, so declare a tool that asks first with the default executor.

### Finding old conversations

Lists page by cursor: pass a page's `nextCursor` back as `cursor` for the next one.

```swift
var query = SessionQuery(limit: 50)
query.state = .live
query.custom = ["ticket": "4721"]          // the labels SessionOptions.custom opened it with
query.createdSince = Calendar.current.date(byAdding: .day, value: -7, to: Date())
let page = try await agents.agent("myagent").sessions.query(query)
let found = try await agents.agent("myagent").sessions.search("pricing")
// page.hasMore, page.nextCursor

try await session.update(title: "Pricing questions")      // nil leaves a field as it is
_ = try await agents.agent("myagent").sessions.update(page.items[0].id, title: "Old pricing")

try await agents.close(sessionID: page.items[0].id)         // stops it, keeps the transcript
try await agents.agent("myagent").sessions.delete(page.items[0].id)   // and this deletes it
```

### Looking something up

`search` is the one routed modality a device may reach, because a question and its answer are
one round trip and the answer is for whoever asked. The router comes from the client, and which
model answers is the router config's to say, not the call's:

```swift
let router = agents.router(config: "healthcare")   // routers/healthcare/router.yaml holds the target
let found = try await router.search("perioperative antibiotic guidance")
```

### Going back, and branching off

`rewind` takes back every turn after the one kept, so the next message carries on from there.
`fork` branches a new session off one instead, leaving the original as it was. A turn's id is
the `id` of a `Response`, not the `turnID` a socket event carries:

```swift
let turns = try await agents.responses(sessionID: session.id).items
try await agents.rewind(sessionID: session.id, to: turns[0].id)
let branch = try await session.fork(ForkOptions(responseID: turns[0].id))

var harder = ForkOptions()
harder.modelOverwrites = ModelOverwrites(llm: "llm-thinking", thinking: .high)
let compared = try await session.fork(harder)   // follow it with attach(sessionID:)
```

A text session is kept in Stream Chat unless it is `incognito`, and a conversation kept there
cannot be rewound, since its transcript would bring the turns back; fork it at the response
instead. An open `AgentSession` keeps the transcript it
already showed, so reload it from `responses` after a rewind.

### Reminders

An agent with `scheduling.enabled` in its config can schedule tasks for the user it talks to.
The app lists them and does what a person does with a reminder; making one, and what the agent
is told when it comes due, is the backend's:

```swift
let page = try await agents.scheduledTasks.list()
_ = try await agents.scheduledTasks.update(page.items[0].id, next: .in("PT15M"))   // snooze
_ = try await agents.scheduledTasks.pause(page.items[0].id)
```

A task runs on the user's clock, so open sessions with `options.timezone =
TimeZone.current.identifier`, which the router remembers as the user's.

### Guests

Somebody can talk to an agent before they sign up. Keep the guest in a store, so the next
launch is the same guest with the same history rather than a second one with none:

```swift
let store = FileGuestStore(url: URL.applicationSupportDirectory.appending(path: "guest.json"))
let guest = try await agents.guestUser(GuestOptions(name: "Ada"), store: store)
agents.setUser(guest)
```

A remembered guest whose token has expired is minted a fresh one under the same id.
`GuestOptions(fresh: true)` is a "not me" button, and `forgetGuest(store:)` is signing out.
With no user set the guest is asked for anonymously; a router that admits no anonymous callers,
Stream's hosted one for now, throws `.configuration`, and your backend mints the guest instead.

## What is deliberately not here

The router is server-side only by default: a handful of operations are marked
`x-client-accessible` in the spec and everything else answers a device 403. This SDK uses
twenty-two of them, and `generate.py` fails if the filter names anything the spec does not open —
which is what stops it growing a method that only ever fails.

What that leaves out, and where it went instead:

| Not here | Ask your backend for |
| --- | --- |
| Writing a config, defining skills, ingesting knowledge | The agent id, which is all the app needs |
| A Stream user token | The token `setUser` is handed, which also joins calls and opens chat |
| Moving a guest's history onto the account they signed up with | Nothing: the backend that authenticated them does it |
| What was said on an earlier call, the call records | Whatever of it the app should see |
| Transcription, a voice, a model on their own | Nothing: a pipeline of your own is a backend |

[`examples/voice_agents/swift_demo`](../../examples/voice_agents/swift_demo) shows both halves:
`configure/` writes the agent and `backend/` signs the user's Stream token.

Every request and socket handshake sends `Stream-Auth-Type: jwt`, which is what declares this
caller a device. It is sent even against a local router with no proxy in front, where the
router would otherwise assume a caller is a backend.

**Somebody else's conversation.** A session belongs to whoever opened it, so `sessions.query()`
only ever lists this caller's own and `attach(sessionID:)` does not find one opened elsewhere. On a
deployment verifying tokens, that boundary is the `user_id` the token names; a caller with no
token is anonymous, and an anonymous claim to a signed-in user's name reaches nothing.

## Regenerating the client

The spec at [`acceleration/api/openapi.yaml`](../../acceleration/api/openapi.yaml) is the
source of truth. The generated Swift is committed, like the Python and Go clients, so building
needs no code generation and no build-tool plugin to be trusted in Xcode.

```bash
uv run sdks/swift/generate.py --check   # does the filter still agree with the spec?
uv run sdks/swift/generate.py           # regenerate
```

The generator is the one Stream Chat and Stream Video generate their clients with, in
`tools/openapi` of the `chat` repository, so regenerating needs that repository checked out
beside this one (`../chat`) and Go. Its `DefaultAPI` and models run on StreamCore's
`DefaultAPITransport` and `DefaultAPIClientMiddleware`, which `Backend.swift` implements.

The generated code is `internal`, and every type crossing the public API is hand-written in
`Models.swift`. That is what lets the spec churn without breaking anybody: a field added to
`Session` is a generated field nobody outside the module can see until it is wrapped.

Websockets are not generated — OpenAPI stops at the upgrade — so `SessionSocket.swift` is
hand-written against the contract in
[`acceleration/internal/api/sessionws.go`](../../acceleration/internal/api/sessionws.go).

## Tests

The package is iOS only, so the tests run on a simulator rather than with `swift test`, from
the root of the repository:

```bash
xcodebuild test -scheme vision-agents-core -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` is needed whenever
`xcode-select -p` points at `/Library/Developer/CommandLineTools`, which has no simulator.

The conversation's state machine is a value type (`Conversation`), so what a stream of frames
means for a transcript is tested on real router frames with no network and no mocks. Frames are
quoted verbatim from `frameOf`, so a change to the wire format on that side fails here.

Against a running router:

```bash
TEST_RUNNER_VISION_AGENTS_URL=http://localhost:8080 TEST_RUNNER_VISION_AGENTS_CUSTOMER_ID=examples \
  xcodebuild test -scheme vision-agents-core -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

That enables `LiveTests`, which creates a real session, asks the model something and waits for
a tool call to come back. `xcodebuild` hands a variable to the tests without its `TEST_RUNNER_`
prefix. Without `VISION_AGENTS_URL` they are skipped, which is the Swift answer to
`@pytest.mark.integration`.

`PostureTests` checks offline that every operation in the generated client is one the spec
marks `x-client-accessible`. Against a router in `api_key` mode (set up as in the
[JavaScript README](../js/README.md#working-on-this-package)) it also checks that a device token
is refused minting call and chat tokens, claiming a guest and reading configs. The same
`xcodebuild` invocation, with `TEST_RUNNER_INTEGRATION_ROUTER_KEY` and
`TEST_RUNNER_INTEGRATION_ROUTER_SECRET`.

`VoiceSession` and `chat()` are wrappers over Stream's SDKs, and are built rather than tested;
what they share is in `Backend` and tested there (`StreamCredentialsTests`).
