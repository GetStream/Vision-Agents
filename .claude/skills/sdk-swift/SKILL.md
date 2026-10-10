---
name: sdk-swift
description: How to build and extend the Swift SDKs in sdks/swift. Read this before changing VisionAgentsCore, VisionAgentsUI, VisionAgentsRTC or VisionAgentsChat, or before adding a Swift client for a new endpoint.
---

# Swift SDK conventions

The per-language half of [sdk](../sdk/SKILL.md). It records the decisions the packages in
[`sdks/swift`](../../../sdks/swift) already follow, so a change lands consistently rather than
re-litigating them. Where a rule has an exception, the exception is written down.

Assume Swift 6 language mode with complete strict concurrency, iOS 17, SPM only.

## Packages

Four packages, not one package with four products. Splitting *targets* stops compilation and
linking; splitting *packages* is the only reliable way to stop a consumer resolving and
fetching a dependency's whole graph, and SwiftPM's pruning has changed across releases. Stream
Video's `StreamWebRTC` is a 47 MB binary artifact, so this matters.

```
core  VisionAgentsCore  generated client, socket, conversation state
ui    VisionAgentsUI    SwiftUI views over that state          -> core
rtc   VisionAgentsRTC   joining the call over Stream Video     -> core
chat  VisionAgentsChat  the Stream Chat channel, iOS only      -> core
```

- `ui` holds no networking. `core` holds no Stream type.
- `rtc` and `chat` are the exception to keeping Stream's types out of the public API, because
  their point is an app using Stream's own SDK: they take the app's `StreamVideo` or
  `ChatClient` in `use`, and `chat()` answers Stream's `ChatChannelController`. Nothing else of
  Stream's crosses, and nothing generated.
- `chat` is iOS only, because Stream Chat's macOS build does not compile under Swift 6, and
  needs `stream-chat-swift` 5.3 or later: 4.x's `ChatClient` is not `Sendable`, and 5.0 to 5.2
  fail inside `stream-core-swift`.
- Cross-package internals (`Backend.give`, `Backend.shared`, `AgentSession.backend`) are
  `@_spi(Stream)`, so `rtc` and `chat` reach them and an app does not see them.
- Declare `platforms:` on every package. An unspecified platform makes consumers discover
  availability failures inside generated code.
- Never add an umbrella product. It hands the WebRTC binary to somebody who wanted text.

**Publishing is not solved by the monorepo.** SwiftPM has no `url + subdirectory` version
dependency, so `.package(path: "../core")` cannot survive publication. When these ship, either
subtree-split each package into its own repository from CI, or use a package registry. During
development, keep the local packages in one Xcode workspace or use
`swift package edit <identity> --path ../core`. Do not make `Package.swift` branch on an
environment variable between `path:` and `url:` — resolution stops being reproducible and
development paths leak into releases.

A library's `Package.resolved` is not a promise to consumers; it is gitignored for the four
packages and committed for the demo app.

## Concurrency

The conversation state is `@MainActor @Observable`; the socket is an `actor`. Not an actor
holding the state with a main-actor projection beside it: that is two truths and a window in
which they disagree.

The state machine itself is a **value type** (`Conversation`) with `mutating func apply(_:)`.
That is what makes it testable without a network, and it is where new event handling goes.

One read loop per connection, weakly capturing its owner *inside* the loop:

```swift
pump = Task { [weak self] in
    for try await event in stream {
        guard let self else { return }   // inside, so each iteration is the only strong hold
        self.apply(event)                // inherits @MainActor: no Task { @MainActor in }
    }
}
```

`guard let self` *outside* an infinite loop is the same retain cycle written differently: the
task owns the object that owns the task, and `deinit` never runs to cancel it.

- Exactly one `receive()` in flight per connection. Concurrent receives reorder frames.
- Never `Task { @MainActor in }` per frame. It loses ordering and cancellation.
- Never a callback the socket invokes to mutate state. One sequential pump.
- Tool handlers are `@Sendable ... async throws`, so their bodies run off the main actor even
  though they are called from it.
- Do not mark a type `@unchecked Sendable` to silence the compiler. Map a non-`Sendable`
  generated value into one of ours instead.
- Answering a tool call must not depend on anybody consuming a public event stream. It happens
  inside `AgentSession`.

## Observation

`@Observable`, which is why the floor is iOS 17. The alternative was an `ObservableObject`
path beside it; two state systems in one SDK diverge. Presenting `@Observable` as the API
while claiming iOS 15 would be a lie, since the type does not exist there.

If a public stream of events is ever added, it must be a `func events()` handing each caller
its own bounded continuation — never one stored `AsyncStream` property. A single stream is not
a broadcast: two iterators compete for elements. Use `.bufferingNewest(n)`, check the `yield`
result, and finish a subscriber that falls behind rather than dropping an unknown number of
events silently. `.unbounded` turns a stalled consumer into a leak.

## The socket

`URLSessionWebSocketTask`. Not a third-party library, which adds TLS, proxy and threading risk
for nothing; not `NWConnection`, which means implementing the upgrade, framing, masking and
control frames yourself.

- The task answers the router's pings itself. Do not decode ping frames as application
  messages, and do not send JSON `"ping"`.
- Set `maximumMessageSize`. A socket gone wrong at the far end otherwise buffers without bound.
- Closing is idempotent: cancel the reader, then `cancel(with: .normalClosure, reason:)`. Never
  a bare `cancel()` for a close the user asked for.
- A cancelled read is a close we asked for, not a failure. Finish the stream, do not throw.

**No automatic reconnection, on purpose.** `respond` and `tool_result` are not idempotent, and
the protocol has no sequence number, cursor or acknowledgement to resume from. Replaying after
a reconnect would duplicate turns and tool results. The socket reports `socketClosed` and the
caller decides. Adding reconnection means adding resume semantics to the router first.

Backgrounding will break the socket; that is iOS, not a bug. Do not promise uninterrupted
background streaming, and do not hold a chat socket open with repeated background tasks.

## Frames

Distinct types per direction: `AgentEvent` in, `Command` out. Never one bidirectional type
with invalid combinations.

Decoding is forward compatible by keeping the whole frame as `JSONValue` and exposing `type`
plus a `kind` that is nil for anything unknown. An event added to the router after this SDK
shipped still reaches the caller whole. A frame that cannot be decoded at all is skipped rather
than closing the socket, which is what the router does with a command it cannot read.

Never `[String: Any]`: not `Codable`, not `Sendable`, and it loses the type guarantees.

Tool-call `arguments` stay an opaque JSON string, because that is the wire contract. Decoding
them is a convenience on top (`argumentValues`), not a requirement.

Quote frames **verbatim from `frameOf`** in
[`sessionws.go`](../../../acceleration/internal/api/sessionws.go) in tests, so a rename on the
Go side fails here rather than in somebody's app.

## The generated client

Committed, generated by `sdks/swift/generate.py`, never hand-edited. Committed rather than a
build-tool plugin because an SDK's generated source should be reviewable and diffable, needs no
generator on a consumer's machine, and needs no plugin trusted in Xcode.

**Generated code is `internal`.** Every type crossing the public API is hand-written in
`Models.swift`. Leaking one generated type couples the SDK's semantic version to the schema and
to the generator. Never return `Components.Schemas.*`, an operation `Input`/`Output`, or an
OpenAPIRuntime wrapper.

Rewrap deliberately:

- **Enums** get an `unknown(String)`-style case or a raw-value fallback. Never `fatalError` on a
  value the server added.
- **Dates** go through one configured `DateTranscoder`. `RouterDates` exists because Go writes
  as many fractional digits as a value needs and none when it needs none, and the generator's
  default reads only the second. Test fractional seconds, no fractional seconds, offsets and
  malformed input.
- **Open-ended objects** become `JSONValue`.

`generate.py` carries an explicit allowlist of operation ids and **fails if any of them is
marked `x-server-side-only`**. That is how the server-side/client-side split is enforced at the
SDK boundary: marking an operation server-side only in the spec is the whole of removing it
from iOS. Adding an endpoint to the Swift SDK means adding it to `OPERATIONS` and rerunning.

The filter is an API-surface tool, not a security boundary — the router still enforces it, and
that is why every request sends `Stream-Auth-Type: jwt`.

## Errors

One `AgentsError`, deliberately: it has five cohesive cases over one service, not a dumping
ground. If it starts collecting unrelated cases, split it per domain (REST, socket, session)
rather than letting it grow.

`CancellationError` is never wrapped and never mapped. A view that goes away mid-request is not
a failure. It must never be confused with a timeout, a lost connection, an HTTP 408, a normal
socket close, or a session ending.

Distinguish, and keep distinguishing: transport failure before any response; an HTTP status;
a decoded error body; a socket close code. Never classify by parsing `localizedDescription`.

`http` carries an `HTTPFailure`: the status, the envelope's `type`, `code`, `message` and
`docURL`, and `requestID` from `X-Request-Id`, which is what a user quotes for a 500. `type` and
`code` are strings, so a value the router adds later still reads, and a body that is not the
envelope keeps its trimmed text as `message` with the rest empty. `FailureMiddleware` throws every
non-2xx, documented or not, before the generated client decodes it, so a switch over an
operation's output handles its success and nothing else. A refused socket upgrade is an `http`
too, with only the status and request id: URLSession never hands over its body.

## Public API

`VisionAgents(apiKey:)`, defaulting to Stream's hosted router, then `setUser` with a token
provider: the shape [sdk](../sdk/SKILL.md) gives every client SDK. `ID` and `URL` are
capitalised as Swift does. The initialiser does no I/O. `VisionAgents(url:customerID:)` is
only for a router running locally with nothing in front of it; an `apiKey:` beside it is
Stream's alone, for chat and video, and the router is still reached by customer id.

```swift
let agents = VisionAgents(apiKey: "your_api_key")
agents.setUser(User(id: "jlahey")) { try await yourBackend.agentToken() }
let session = try await agents.agent("myagent").sessions.create()
let turn = try await session.responses.create("Where is order 1042?")
```

and the advanced form is a request value, not thirty initialiser parameters:

```swift
var options = SessionOptions(agent: "myagent")
options.greeting = Greeting("Hello.", mode: .variation)
let session = try await agents.sessions.create(options)
```

The key is public and goes in the query, where Stream's proxy and the router both read it;
the token goes in `Authorization` and never in a URL.

`nil` means "omit the field and let the config or the router decide", never "send a copy of the
server's default". A schema default copied into the client is how a caller silently loses the
model their config named.

Async/await only. No completion handlers, no `.shared` singleton, no configuration read from
`Info.plist` or the environment.

**A token provider, not a token,** for anything with credentials. `Backend` takes a
`TokenProvider` in `setUser` and threads it through requests, the socket handshake and a
single 401 retry, single-flighted.

**Stream is set up once.** The key, the user and the token from `setUser` are the one identity
for the router, Stream Chat and Stream Video. `VoiceSession.join()` and `session.chat()` take
no credentials: `Backend.shared` builds one client per kind, key and user, single-flighted and
shared by every session, and hands Stream a provider that asks `streamCredentials(refresh:
true)`, so an hour-long call outlives its token and two clients refreshing at once fetch one.
`agents.use(_:)` hands over the app's own `ChatClient` or `StreamVideo`, which is used first,
kept across `setUser` and never disconnected here; one connected as another user is a
`.configuration` error. `agents.disconnect()` closes only what was built, and closing a session
closes nothing. Opting in is by package: there is no runtime registry.

## SwiftUI

`@ViewBuilder` slots for customisation, not a theme object. A forty-value `AgentTheme` becomes
a second design system that fights the host's real one.

- Respect the environment: `colorScheme`, `dynamicTypeSize`, `layoutDirection`, `tint`,
  `accessibilityReduceMotion`. Never set `.preferredColorScheme` or install fonts from a
  package.
- No component creates a `NavigationStack`, pushes a destination, dismisses itself, or assumes
  it is presented modally. Navigation is the host's.
- `ConversationView` is the arrangement most apps want; `TranscriptView`, `Composer` and
  `AgentStatusView` are public so a host can take it apart instead of fighting it.
- `LazyVStack` with stable turn ids, and one mutable in-flight turn — not a row per token, and
  never the message text as the id, which destroys and recreates the row on every delta.
- Follow the bottom without animating growing text; animate only a new line. Animating each
  delta is what makes a transcript judder.
- Assets belong to `ui` with `Bundle.module`. Prefer SF Symbols. Expose views, not asset-name
  strings.

Known gap: deltas are published straight to SwiftUI rather than coalesced to a frame boundary,
and `TranscriptView` always follows the bottom rather than stopping when the reader scrolls
away. Both are worth fixing before this carries a long transcript.

## Tests

Swift Testing, not XCTest. Never mock; assert on state and outputs, never on a call happening.

Three layers, in order of how much they cost to run:

1. **The state machine.** `Conversation` is a value type, so what a stream of real router
   frames means for a transcript is tested in milliseconds with no network. Most new behaviour
   belongs here.
2. **Live tests.** `LiveTests` is gated on `VISION_AGENTS_URL`, which is the Swift answer to
   `@pytest.mark.integration`. It creates a real session, asks the model something and waits
   for a real tool call. Mark the suite `@MainActor`, because `AgentSession` is.
3. **Builds.** `ui`, `rtc` and `chat` are views and wrappers over Stream's SDKs; build them
   for the simulator. What they share is in `core` and tested there (`StreamCredentialsTests`).

Wait on a condition with a deadline, never `Task.sleep` for a fixed guess:

```swift
try await until(20) { session.state == .idle && session.turns.count >= 2 }
```

If a loopback server is ever needed for close codes, malformed bodies or reconnect timing,
write a real one on `127.0.0.1:0` rather than a `URLProtocol` subclass — `URLProtocol` cannot
test a genuine WebSocket upgrade. Do not spend tests re-testing the generated client's
forwarding; it is machine-produced.

## Audio and permissions, for `rtc`

- **One owner of `AVAudioSession`.** A WebRTC SDK and the app both setting category, mode or
  active state gives dropped Bluetooth, wrong routing and activation failures. Use Stream's
  supported hooks; do not set audio policy at initialisation, and do not "save the category at
  startup and restore it later" — something else may have legitimately changed it in between.
- Remote audio plays through the audio session once joined. There is no view to add for it;
  `enableSpeakerPhone()` is what makes it hands-free instead of the earpiece.
- `NSMicrophoneUsageDescription` is the **host app's**, and a package cannot supply it. Missing
  it terminates the app — not a recoverable permission denial. Check authorisation before
  joining, and distinguish not-determined, denied, restricted and granted.
- The simulator does not test echo cancellation, Bluetooth routing, receiver versus speaker,
  interruption by a call, or route changes. Release RTC changes only after device testing.

## Logging

Transcripts, prompts, tool arguments and tokens are the most sensitive data here. Do not log
frames or bodies by default. Never put a token in a query string.

## Reviewing a change

Reject it if it:

- mutates conversation state off the main actor, or hops per frame;
- exposes a generated type publicly, or a Stream type outside `use` and `chat()`;
- builds a second Stream client for a user who already has one, or disconnects one the app
  handed over;
- uses `[String: Any]`, or `@unchecked Sendable` to quiet a warning;
- starts more than one receive per connection, or replays `respond`/`tool_result`;
- treats an unknown frame as fatal, or discards its payload;
- wraps `CancellationError`, or reports an HTTP status as a transport failure;
- hand-edits generated code, or adds a client method for an `x-server-side-only` operation;
- adds a theme object, a `NavigationStack`, or an asset-name string to `ui`;
- asserts that a method was called.
