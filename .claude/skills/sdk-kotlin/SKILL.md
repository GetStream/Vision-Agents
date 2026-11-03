---
name: sdk-kotlin
description: How to build and extend the Kotlin/Android SDK in sdks/kotlin. Read this before changing vision-agents-core, or before adding a Kotlin client for a new endpoint.
---

# Kotlin SDK conventions

The per-language half of [sdk](../sdk/SKILL.md). It records the decisions the modules in
[`sdks/kotlin`](../../../sdks/kotlin) already follow, so a change lands consistently rather than
re-litigating them. It mirrors [sdk-swift](../sdk-swift/SKILL.md): the two SDKs are the same
API in two languages, and a feature added to one belongs in the other.

Assume Kotlin 2.4, JVM 17 bytecode, AGP 9, Gradle with the version catalogue in
`gradle/libs.versions.toml`. Client side only: no folder sync, no dispatch, no config writing.

## The module, and what it is called

```
core  io.getstream:vision-agents-core  io.getstream.visionagents.android.core  Android library: HTTP, socket, state, Stream Chat, Stream Video
```

One module, as Swift is one package (`VisionAgentsCore`). The artifact is named after it,
written the way Maven writes names: the group is Stream's `io.getstream`, the artifact is
kebab-case, and the package is the reverse-DNS lower case Kotlin requires. `vision-agents-core`
sits beside Stream's own `stream-video-android-*` and `stream-chat-android-*` in a dependency list
and says which product it is.

- `core` is a `com.android.library`. It holds the conversation, `session.chat(context)`,
  `session.channelState(context, scope)` and `VoiceSession`, and depends on
  `stream-chat-android-client` and `stream-video-android-core`.
- It ships no views. Apps compose Stream Chat's AI components
  (`stream-chat-android-ai-compose`), the way Swift apps compose `StreamChatAI`.
- Stream Chat 7.15.0+ comes from `https://stream-io-repo.com`, not Maven Central;
  `settings.gradle.kts` lists it for `io.getstream` only.
- AGP 9 compiles Kotlin itself: apply `com.android.library`, never
  `org.jetbrains.kotlin.android`.
- compileSdk 37 is Stream Video 1.34's floor; minSdk 24 is Stream's.

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

`Http` is read by one function, `refusal` in `Requests.kt`, for a request and a refused socket
handshake alike: `reason` is the envelope's `message`, beside `type`, `code` and `docUrl` as
strings (the router adds values), and `requestId` is `X-Request-Id`, what a 500 is quoted by. A
body that is not the envelope (a proxy's page, an empty one, an older `{"error": "..."}`) keeps
its trimmed text, or the status phrase, as `reason` with those three null. Ktor drops a refused
upgrade's response, so `Backend` opens sockets through `Handshakes`, which reads it in OkHttp's
`onFailure`.

## Stream clients: chat and video

Stream is set up once: the key, and the user and token `setUser` is given, are the one identity
for the router, Stream Chat and Stream Video. There is no call token and no credentials object.

- `Backend.shared(kind, open, disconnect)` builds one client per kind, key and user however many
  sessions ask, keyed before any token is asked for, and forgets one that failed to open.
  `give(kind, client)` records the app's own, handed over with `agents.use(...)`. Both are
  `internal`.
- `agents.disconnect()` closes only what was built. Leaving a call or closing a session
  disconnects nothing; the next session reuses the client.
- `ChatClient` and `StreamVideo` are one per process, and building another replaces the app's.
  One the app built without handing it over is adopted if it is connected as the same user and
  refused with `Configuration` otherwise. `Chat.kt` and `VoiceSession.kt` each remember the
  last client they built, so they can tell their own instance from the app's.
- A token provider that asks `backend.streamCredentials(refresh = true)` again, never one fixed
  token: a token expires an hour in, and two clients refreshing at once fetch one.
- Camera direction and state are set before joining, so no front-facing frame is published.
  Speakerphone on after joining, since a hands-free agent wants the speaker.
- The camera is sampled for the agent through `call.videoFilter`, the call's one filter slot.
  Whatever filter is there stays underneath and runs first (moderation's blur is a
  `BitmapVideoFilter`, so the one over it is too), and `CameraFilter` puts itself back on top
  when something replaces it. Never clear the slot or replace a filter you did not set.
- `RECORD_AUDIO` and `CAMERA` are the host's to request. The emulator does not test echo
  cancellation, Bluetooth or routing; release RTC changes after device testing.

## Tests

JUnit 5 through `kotlin-test`. Never mock; assert on state and outputs, never on a call.

1. **The state machine and the wire.** `ConversationTest` and `WireTest` run real frames through
   `reduce` and the codecs, in milliseconds.
2. **A real loopback router.** `TestRouter` is a Ktor CIO server on `127.0.0.1:0` that records
   what arrived and answers what the test scripted, sockets included. That is how headers,
   bodies, error statuses, close codes and tool round trips are tested without a mock.
3. **Sharing and posture.** `StreamClientsTest` proves a second Stream client is never built:
   sessions asking at once share one, the app's own is used and left connected, `disconnect`
   closes only what was built. `PostureTest` drives every operation the SDK has against
   `TestRouter` and fails if one is not `x-client-accessible` in the spec. Add to both when a
   new operation or a new Stream client appears.
4. **Live.** `LiveTest` is gated on `VISION_AGENTS_URL` (`@EnabledIfEnvironmentVariable`), the
   Kotlin answer to `@pytest.mark.integration`. `KeyedRouterPostureTest` is gated on
   `INTEGRATION_ROUTER_KEY` and checks a device token's 403s against a router in `api_key` mode.

The tests are plain JVM unit tests run by AGP (`:core:testDebugUnitTest`), with no emulator.

CI is [`client-sdks.yml`](../../../.github/workflows/client-sdks.yml), on any change under
`sdks/kotlin` or to the spec: `:core:testDebugUnitTest` and `:core:assembleRelease`, then
`generate.py` again, failing if the generated models differ from what is committed.

Wait on a condition with a deadline (`until { ... }`), never a fixed delay. A collector that
must see an event has to be subscribed before the frame is sent: start it with
`CoroutineStart.UNDISPATCHED`, or it races the socket.

Docker only; the commands are in the README. Build output stays under `build/` and `.gradle/`,
which `sdks/kotlin/.gitignore` covers.

## Reviewing a change

Reject it if it:

- exposes a generated type, or a Stream type anywhere but `chat()`, `channelState()` and
  `VoiceSession`;
- sends a schema default the caller did not set;
- answers tools from a collector, starts a second read loop, or replays `respond`/`tool_result`;
- treats an unknown frame or enum value as fatal;
- catches `CancellationException`, or reports an HTTP status as a transport failure;
- hand-edits generated code, or adds a method for an operation that is not client-accessible;
- adds views: they belong to Stream Chat's AI components and the apps;
- builds a second Stream client, replaces the app's process-wide one, or disconnects one the
  app gave;
- mocks, or asserts that a method was called.
