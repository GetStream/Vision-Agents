# Swift SDKs

Three iOS packages for talking to an agent from a phone. They are separate packages, not
products of one, because SPM resolves every dependency a manifest declares whether or not you
use the product it belongs to — and `StreamWebRTC` is a 47 MB binary. An app that only holds a
text conversation should not download it.

| Package | Module | Depends on | What it is |
| --- | --- | --- | --- |
| `core/` | `VisionAgentsCore` | OpenAPI runtime, URLSession | The generated client, the session socket, and the conversation state |
| `ui/` | `VisionAgentsUI` | `core` | SwiftUI views over that state |
| `rtc/` | `VisionAgentsRTC` | `core`, `stream-video-swift` | Joining the call, so the conversation can be spoken |

iOS 17 is the floor. It is `@Observable`'s floor, and the alternative was an `ObservableObject`
path beside it to serve devices that will not be running a new SDK anyway.

`ui` and `rtc` reach `core` with `.package(path: "../core")`, which works in this repository and
cannot survive publication: SPM has no way to depend on a subdirectory of a tagged repository.
Shipping these means splitting each into its own repository from CI, or a package registry.

## Using them

```swift
let agents = VisionAgents(url: URL(string: "https://your-router")!, customerID: "acme")

// In writing. No call is joined, nothing is transcribed or spoken.
let chat = try await agents.chat(agent: configID)
await chat.start()
try await chat.send("What are your opening hours?")
// chat.turns grows as the reply streams in; chat.state says what the agent is doing.

// Out loud. The agent joins a call and so does this device.
let voice = try await VoiceSession.start(agents: agents, agent: configID)
await voice.join(credentials: yourBackend.callCredentials)
```

`agent:` is a config id, not a name, and `join` is handed a closure rather than minting its
own token. Both are the same fact: reading the configs and minting a call token are server-side
only, so the app is told which agent it talks to and is handed the token to join with.

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
    parameters: .strings(["order_id": "the order number"], required: ["order_id"])
) { arguments in
    await Orders.local.find(arguments["order_id"]?.stringValue ?? "")
}

let chat = try await agents.chat(agent: configID, tools: [lookup])
```

### Looking something up

`Router.search` is the one routed modality a device may reach, because a question and its
answer are one round trip and the answer is for whoever asked:

```swift
let router = Router(url: url, customerID: "acme", config: "healthcare")
let found = try await router.search("perioperative antibiotic guidance")
```

### Going back, and branching off

`rewind` takes back every turn after the one kept, so the next message carries on from there.
`fork` branches a new session off one instead, leaving the original as it was. A turn's id is
the `id` of a `Response`, not the `turnID` a socket event carries:

```swift
let turns = try await agents.responses(sessionID: session.id)
try await agents.rewind(sessionID: session.id, to: turns[0].id)
let branch = try await agents.fork(sessionID: session.id, ForkOptions(responseID: turns[0].id))
```

A conversation kept in Stream Chat cannot be rewound, since its transcript would bring the
turns back; fork it at the response instead. An open `AgentSession` keeps the transcript it
already showed, so reload it from `responses` after a rewind.

## What is deliberately not here

The router is server-side only by default: a handful of operations are marked
`x-client-accessible` in the spec and everything else answers a device 403. This SDK uses
seven of them, and `generate.py` fails if the filter names anything the spec does not open —
which is what stops it growing a method that only ever fails.

What that leaves out, and where it went instead:

| Not here | Ask your backend for |
| --- | --- |
| Writing a config, defining skills, ingesting knowledge | The agent id, which is all the app needs |
| A token to join the agent's call | `CallCredentials`, which `join` is handed |
| A Stream Chat token | Credentials, if you bring the dependency |
| What was said on an earlier call, the call records | Whatever of it the app should see |
| Transcription, a voice, a model on their own | Nothing: a pipeline of your own is a backend |

[`examples/voice_agents/swift_demo`](../../examples/voice_agents/swift_demo) shows both halves:
`configure/` writes the agent and `backend/` mints the call tokens.

Every request and socket handshake sends `Stream-Auth-Type: jwt`, which is what declares this
caller a device. It is sent even against a local router with no proxy in front, where the
router would otherwise assume a caller is a backend.

**Somebody else's conversation.** A session belongs to whoever opened it, so `sessions()` only
ever lists this caller's own and `attach(sessionID:)` does not find one opened elsewhere. On a
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

The iOS packages are built rather than tested, since they are views and a WebRTC wrapper:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer   # if xcode-select points at the CLI tools
for pkg in core ui rtc; do
  (cd sdks/swift/$pkg && xcodebuild -scheme vision-agents-$pkg \
     -destination 'generic/platform=iOS Simulator' build)
done
```
