# Swift SDKs

Three iOS packages for talking to an agent from a phone. They are separate packages, not
products of one, because SPM resolves every dependency a manifest declares whether or not you
use the product it belongs to — and `StreamWebRTC` is a 47 MB binary. An app that only holds a
text conversation should not download it.

| Package | Module | Depends on | What it is |
| --- | --- | --- | --- |
| `core/` | `VisionAgentsCore` | OpenAPI runtime, URLSession | The generated client, the session socket, and the conversation state |
| `ui/` | `VisionAgentsUI` | `core`, `stream-chat-swift-ai` | SwiftUI views over that state |
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
let chat = try await agents.chat(agent: "swift_demo")
await chat.start()
try await chat.send("What are your opening hours?")
// chat.turns grows as the reply streams in; chat.state says what the agent is doing.

// Out loud. The agent joins a call and so does this device.
let voice = try await VoiceSession.start(agents: agents, agent: "swift_demo")
await voice.join()
```

With `VisionAgentsUI` a whole conversation is one view:

```swift
ConversationView(session: chat)
```

`TranscriptView`, `Composer` and `AgentStatusView` are public and work on their own, so a host
that wants a different arrangement takes them apart rather than fighting `ConversationView`.

The transcript and the composer are Stream's [AI chat components][ai], so an app that already
uses them gets one composer rather than two that almost agree. What the agent says is rendered
as markdown -- code blocks, tables, images -- and written out a letter at a time while it
streams. The composer carries the send and stop-generating buttons and dictation; it carries no
attachment button, because a session carries text.

Dictation means the host app needs two usage strings in its Info.plist, which a package cannot
supply and without which iOS terminates the app the first time it asks:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>So you can talk to the agent.</string>
<key>NSSpeechRecognitionUsageDescription</key>
<string>So you can dictate a message instead of typing it.</string>
```

[ai]: https://github.com/GetStream/stream-chat-swift-ai

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

let chat = try await agents.chat(agent: "swift_demo", tools: [lookup])
```

## What is deliberately not here

**Configuring an agent.** Writing a config, defining skills, ingesting knowledge and waiting
for dispatched calls are marked `x-server-side-only` in the spec, and the router answers a
device 403 for all of them. `generate.py` asserts that none of them are in the client, so the
SDK cannot grow a method that only ever fails. Use the Go or Python SDK from your backend —
[`examples/agents/swift_demo`](../../examples/agents/swift_demo) shows both halves.

Every request and socket handshake sends `Stream-Auth-Type: jwt`, which is what declares this
caller a device. It is sent even against a local router with no proxy in front, where the
router would otherwise assume a caller is a backend.

**A Stream Chat dependency.** The live conversation comes off the session socket and the stored
one comes from the router, which reads the chat channel for you, so Stream Chat would be a
second way to do what `core` already does. If you want it anyway, `agents.chatToken(agentID:)`
gives you the credentials and you bring the dependency.

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
