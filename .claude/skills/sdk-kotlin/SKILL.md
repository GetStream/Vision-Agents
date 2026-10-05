---
name: sdk-kotlin
description: How to build and extend the Kotlin/Android SDKs in sdks/kotlin. Read this before changing vision-agents-core, vision-agents-ui or vision-agents-rtc, or before adding a Kotlin client for a new endpoint.
---

# Kotlin SDK conventions

The per-language half of [sdk](../sdk/SKILL.md). It records the decisions the modules in
[`sdks/kotlin`](../../../sdks/kotlin) already follow, so a change lands consistently rather than
re-litigating them. It mirrors [sdk-swift](../sdk-swift/SKILL.md): the two SDKs are the same
API in two languages, and a feature added to one belongs in the other.

Assume Kotlin 2.4, JVM 17 bytecode, AGP 9, Gradle with the version catalogue in
`gradle/libs.versions.toml`. Client side only: no folder sync, no dispatch, no config writing.

## Modules, and what they are called

```
core  io.getstream:vision-agents-core  io.getstream.visionagents.core  plain JVM: HTTP, socket, state
ui    io.getstream:vision-agents-ui    io.getstream.visionagents.ui    Compose views over it   -> core
rtc   io.getstream:vision-agents-rtc   io.getstream.visionagents.rtc   Stream Video and Chat   -> core
```

The sdk skill asks for `ai-kotlin-core` and `ai-kotlin-rtc`. The artifacts are named after the
Swift packages instead (`VisionAgentsCore`, `VisionAgentsUI`, `VisionAgentsRTC`), written the
way Maven writes names: the group is Stream's `io.getstream`, the artifact is kebab-case, and
the package is the reverse-DNS lower case Kotlin requires. `vision-agents-*` sits beside
Stream's own `stream-video-android-*` and `stream-chat-android-*` in a dependency list and says
which product it is; `ai-kotlin-*` repeats the language the artifact is already written in and
names no product. One name per concept across Swift and Kotlin is what lets the docs, the demo
apps and a bug report talk about "core" and "rtc" without saying which platform.

- `core` is `kotlin("jvm")`, not an Android library. It has nothing Android in it, so its tests
  run under `gradle:jdk21` with no Android SDK, and a text-only app never resolves WebRTC.
- `ui` holds no networking. `rtc` keeps Stream's types out of `core`'s public API.
- `-PcoreOnly` drops `ui` and `rtc` from `settings.gradle.kts`. It is a flag rather than a
  guess at whether an Android SDK is present, so a misconfigured machine fails loudly.
- AGP 9 compiles Kotlin itself: apply `com.android.library` and the Compose plugin, never
  `org.jetbrains.kotlin.android`.
- compileSdk 37 is Stream Video 1.34's floor; minSdk 24 is Stream's.
- `rtc` declares Stream Chat because the sdk skill says both Stream SDKs come with it, so an app
  that shows a call beside a channel gets versions that agree. Nothing here uses Chat yet; do
  not grow a Chat wrapper without a feature that needs one.

## HTTP and the socket

Ktor's client on the OkHttp engine. OkHttp because it is what Android apps already carry and
what Stream's SDKs use, so a host can hand in its own `OkHttpClient` (interceptors, pinning, a
proxy) and have it apply to requests and the socket alike. Ktor on top because its WebSocket
session and its request builders are coroutine-native; bare OkHttp is callbacks.

- `expectSuccess = false`: a status is read and mapped, never thrown by Ktor.
- Path segments go through `appendPathSegments(..., encodeSlash = true)`. A session id is data,
  and a `/` in it must not become a path.
- One read loop per socket. Answering a tool call happens in that loop, never in a collector of
  `events()`, so nobody collecting cannot stall a tool.
- **No automatic reconnection, on purpose.** `respond` and `tool_result` are not idempotent and
  the protocol has no cursor to resume from. The socket reports `SocketClosed` and the caller
  decides. Adding reconnection means adding resume semantics to the router first.
- Credentials go in handshake headers, never the query string, where they end up in logs.
- `interrupt` is a socket command. The REST `interruptSession` is not client-accessible.

## State

`AgentSession` exposes `StateFlow`s: `conversation`, `isConnected`, `failure`. One immutable
`Conversation` holds the turns and what the agent is doing, so a screen never sees a transcript
and a state that disagree. It is a value with `reduce(event)`, tested on verbatim router
frames with no network; most new event handling belongs there.

- `events()` is a `SharedFlow` with no replay and a bounded buffer that drops the oldest. A
  collector that falls behind loses events; the socket never waits for it.
- The session owns a `CoroutineScope(SupervisorJob())` and `close()` cancels it. It is not tied
  to a screen; a `ViewModel` holding it is the usual home.
- `CancellationException` is never caught and mapped. A screen going away is not a failure.

## The generated models

`generate.py` prunes the spec to the allowlisted operations and the components they reach,
then runs `openapitools/openapi-generator-cli` in Docker for **models only**, with
kotlinx.serialization. The request layer is hand-written: twelve calls do not need a generated
client, and a generated one would be public API we did not design.

- Generated code is `internal` (`nonPublicApi`). Every public type is hand-written in `core`
  and mapped with `of(schema)` / `.schema`. Never return a generated type.
- `generate.py` fails if an allowlisted operation is not `x-client-accessible`. Adding an
  endpoint means adding it to `OPERATIONS` and rerunning.
- Schema **defaults are stripped** before generating. A default is the router's to apply; in
  the model it is the value a field starts as, and a caller's `false` equal to it would be
  dropped while the config it was overriding kept its `true`. Every field starts null, the
  JSON is `explicitNulls = false`, and null means "omit and let the config decide".
- Open-ended objects are `JsonElement`/`JsonObject`, never `Map<String, Any>`.
- Enums map to ours with an `Unknown` case, or fold to the nearest meaning the way Swift does
  (`Session.State` is `Live` or `Ended`). A value the server adds is never an exception.
- Dates are `kotlin.time.Instant`. A date that does not parse is `Unreadable`, not a crash.

## Errors

One sealed `AgentsException`: `Http` (status, the router's reason, `Retry-After`), `Transport`,
`SocketClosed` (code, reason), `Unreadable`, `Configuration`. Keep transport failure, an HTTP
status, a decoded body and a close code distinct; never classify by parsing a message. A 401
with a `TokenProvider` refreshes once and retries once, single-flighted behind a `Mutex`.

## Compose

- Slots (`bubble: @Composable (Turn) -> Unit`) for customisation, not a theme object. The views
  use `MaterialTheme` colours and typography, so they follow whatever theme the host has.
- No component navigates, and none closes a session: a rotation recomposes, and ending the
  conversation is the owner's.
- `LazyColumn` keyed by turn id. Follow the bottom without animating growing text; animate
  only a new line.
- Collect with `collectAsStateWithLifecycle`.

## rtc

- Stream Video keeps one client per process and refuses a second. `VoiceSession` uses an
  existing client signed in as the same user, builds one otherwise, and removes only a client
  it built.
- A `TokenProvider` that asks the app's backend again, never one fixed token: a call token
  expires an hour in.
- Camera direction and state are set before joining, so no front-facing frame is published.
  Speakerphone on after joining, since a hands-free agent wants the speaker.
- `RECORD_AUDIO` and `CAMERA` are the host's to request. The emulator does not test echo
  cancellation, Bluetooth or routing; release RTC changes after device testing.

## Tests

JUnit 5 through `kotlin-test`. Never mock; assert on state and outputs, never on a call.

1. **The state machine and the wire.** `ConversationTest` and `WireTest` run real frames through
   `reduce` and the codecs, in milliseconds.
2. **A real loopback router.** `TestRouter` is a Ktor CIO server on `127.0.0.1:0` that records
   what arrived and answers what the test scripted, sockets included. That is how headers,
   bodies, error statuses, close codes and tool round trips are tested without a mock.
3. **Live.** `LiveTest` is gated on `VISION_AGENTS_URL` (`@EnabledIfEnvironmentVariable`), the
   Kotlin answer to `@pytest.mark.integration`.
4. **Builds.** `ui` and `rtc` are assembled in `cimg/android`, not tested.

Wait on a condition with a deadline (`until { ... }`), never a fixed delay. A collector that
must see an event has to be subscribed before the frame is sent: start it with
`CoroutineStart.UNDISPATCHED`, or it races the socket.

Docker only; the commands are in the README. Build output stays under `build/` and `.gradle/`,
which `sdks/kotlin/.gitignore` covers.

## Reviewing a change

Reject it if it:

- exposes a generated or Stream type from `core`, or adds Android to `core`;
- sends a schema default the caller did not set;
- answers tools from a collector, starts a second read loop, or replays `respond`/`tool_result`;
- treats an unknown frame or enum value as fatal;
- catches `CancellationException`, or reports an HTTP status as a transport failure;
- hand-edits generated code, or adds a method for an operation that is not client-accessible;
- adds a theme object or navigation to `ui`;
- mocks, or asserts that a method was called.
