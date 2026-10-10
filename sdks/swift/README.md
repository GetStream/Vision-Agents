# Swift SDKs

Four iOS packages for talking to an agent from a phone. They are separate packages, not
products of one, because SPM resolves every dependency a manifest declares whether or not you
use the product it belongs to — and `StreamWebRTC` is a 47 MB binary. An app that only holds a
text conversation should not download it, nor Stream Chat unless it shows the channel.

| Package | Module | Depends on | What it is |
| --- | --- | --- | --- |
| `core/` | `VisionAgentsCore` | OpenAPI runtime, URLSession | The generated client, the session socket, and the conversation state |
| `ui/` | `VisionAgentsUI` | `core` | SwiftUI views over that state |
| `rtc/` | `VisionAgentsRTC` | `core`, `stream-video-swift` | Joining the call, so the conversation can be spoken |
| `chat/` | `VisionAgentsChat` | `core`, `stream-chat-swift` 5.3+ | The Stream Chat channel a text conversation is kept in |

iOS 17 is the floor. It is `@Observable`'s floor, and the alternative was an `ObservableObject`
path beside it to serve devices that will not be running a new SDK anyway.

`ui`, `rtc` and `chat` reach `core` with `.package(path: "../core")`, which works in this repository and
cannot survive publication: SPM has no way to depend on a subdirectory of a tagged repository.
Shipping these means splitting each into its own repository from CI, or a package registry.

## Using them

```swift
let agents = VisionAgents(apiKey: "your_api_key")
agents.setUser(User(id: "jlahey")) { try await yourBackend.agentToken() }

// In writing. No call is joined, nothing is transcribed or spoken.
let session = try await agents.agent("myagent").sessions.create()
let turn = try await session.responses.create("What are your opening hours?")
// session.responses.items(responseID: turn.id) reads back what the agent did.

// Or watch it arrive: session.turns grows as the reply streams in.
await session.start()
_ = try await session.responses.create("And on Sundays?")

// Out loud. The agent joins a call and so does this device.
let voice = try await VoiceSession.start(agents: agents, agent: "myagent")
await voice.join()

// The Stream Chat channel the text session is kept in, with VisionAgentsChat.
let channel = try await session.chat()
```

`VisionAgents(apiKey:)` reaches Stream's hosted router; pass `url:` for another. The token
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
agents.use(chatClient)   // VisionAgentsChat
agents.use(streamVideo)  // VisionAgentsRTC
```

One connected as somebody other than the user `setUser` named is refused. Beside a customer id,
`apiKey:` is Stream's alone: the router is still reached by customer id, and chat and video
connect with the key.

With `VisionAgentsUI` a whole conversation is one view:

```swift
ConversationView(session: chat)
```

`TranscriptView`, `Composer` and `AgentStatusView` are public and work on their own, so a host
that wants a different arrangement takes them apart rather than fighting `ConversationView`.

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

### Finding old conversations

Lists page by cursor: pass a page's `nextCursor` back as `cursor` for the next one.

```swift
var query = SessionQuery(limit: 50)
query.state = .live
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
let branch = try await agents.fork(sessionID: session.id, ForkOptions(responseID: turns[0].id))
```

A text session is kept in Stream Chat unless it is `incognito`, and a conversation kept there
cannot be rewound, since its transcript would bring the turns back; fork it at the response
instead. An open `AgentSession` keeps the transcript it
already showed, so reload it from `responses` after a rewind.

## What is deliberately not here

The router is server-side only by default: a handful of operations are marked
`x-client-accessible` in the spec and everything else answers a device 403. This SDK uses
twelve of them, and `generate.py` fails if the filter names anything the spec does not open —
which is what stops it growing a method that only ever fails.

What that leaves out, and where it went instead:

| Not here | Ask your backend for |
| --- | --- |
| Writing a config, defining skills, ingesting knowledge | The agent id, which is all the app needs |
| A Stream user token | The token `setUser` is handed, which also joins calls and opens chat |
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

Regenerating needs Apple's generator on disk at `.codegen/swift/swift-openapi-generator`:

```bash
mkdir -p /tmp/oapigen && cd /tmp/oapigen
cat > Package.swift <<'EOF'
// swift-tools-version:6.0
import PackageDescription
let package = Package(
    name: "oapigen",
    dependencies: [.package(url: "https://github.com/apple/swift-openapi-generator", exact: "1.13.1")]
)
EOF
swift build -c release --product swift-openapi-generator
mkdir -p <repo>/.codegen/swift
cp .build/release/swift-openapi-generator <repo>/.codegen/swift/
```

The generated code is `internal`, and every type crossing the public API is hand-written in
`Models.swift`. That is what lets the spec churn without breaking anybody: a field added to
`Session` is a generated field nobody outside the module can see until it is wrapped.

Websockets are not generated — OpenAPI stops at the upgrade — so `SessionSocket.swift` is
hand-written against the contract in
[`acceleration/internal/api/sessionws.go`](../../acceleration/internal/api/sessionws.go).

## Tests

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd sdks/swift/core && swift test        # offline, about a second
```

`DEVELOPER_DIR` is needed whenever `xcode-select -p` points at `/Library/Developer/CommandLineTools`,
whose toolchain has no `Testing` module — the failure is `no such module 'Testing'` rather than
anything about the toolchain. Set it permanently with
`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.

The conversation's state machine is a value type (`Conversation`), so what a stream of frames
means for a transcript is tested on real router frames with no network and no mocks. Frames are
quoted verbatim from `frameOf`, so a change to the wire format on that side fails here.

Against a running router:

```bash
VISION_AGENTS_URL=http://localhost:8080 VISION_AGENTS_CUSTOMER_ID=examples swift test
```

That enables `LiveTests`, which creates a real session, asks the model something and waits for
a tool call to come back. Without `VISION_AGENTS_URL` they are skipped, which is the Swift
answer to `@pytest.mark.integration`.

The iOS packages are built rather than tested, since they are views and wrappers over Stream's
SDKs. `chat` is iOS only: Stream Chat does not build for macOS under Swift 6.

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer   # if xcode-select points at the CLI tools
for pkg in core ui rtc chat; do
  (cd sdks/swift/$pkg && xcodebuild -scheme vision-agents-$pkg \
     -destination 'generic/platform=iOS Simulator' build)
done
```
