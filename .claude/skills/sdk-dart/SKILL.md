---
name: sdk-dart
description: How to build and extend the Dart and Flutter SDKs in sdks/dart. Read this before changing vision_agents_core, vision_agents_chat or vision_agents_rtc, the Flutter demo's use of Stream's AI components, or before adding a Dart client for a new endpoint.
---

# Dart SDK conventions

The per-language half of [sdk](../sdk/SKILL.md). It records the decisions the packages in
[`sdks/dart`](../../../sdks/dart) already follow, so a change lands consistently rather than
re-litigating them. It is a **client-side** SDK: what a phone, a browser or a desktop app does.
Folder sync, dispatch and writing configs are backend work and stay out.

Dart 3.10 is the floor for `core` and `rtc` (Stream Video 1.6's); `chat` is 3.12 because Stream
Chat 10.4 is.

## Packages

```
core  vision_agents_core  pure Dart: generated client, socket, conversation state
chat  vision_agents_chat  pure Dart: the conversation's Stream Chat channel   -> core
rtc   vision_agents_rtc   joining the call over Stream Video                  -> core
```

**Three standalone packages, not a pub workspace.** A workspace resolves every member as one
graph, so once `rtc` (Flutter) is a member, `core` cannot be resolved or tested with the
plain Dart SDK — and a pure-Dart core testable in `dart:stable` is the point of having one. The
price is three lockfiles and a local link: `chat` and `rtc` declare
`vision_agents_core: ^x.y.z` as a hosted dependency, and a committed `pubspec_overrides.yaml`
points it at `../core`. Pub never publishes that file and consumers never see it. Before
releasing, publish `core` first and check the others resolve against the hosted version with the
override removed.

- `core` has no Flutter import and no Stream dependency. `dart:io` stays out of `lib/` too, so
  it runs on the web: `package:http` and `package:web_socket` pick the platform themselves.
- `rtc` keeps Stream Video's types out of `core`. `VoiceState.call` is the one Stream type it
  exposes, because rendering video needs it.
- **No conversation views.** The views are Stream Chat's AI components,
  `stream_chat_flutter_ai`, which the app adds itself, as the Swift SDK leaves them to
  `StreamChatAI`; see [Views](#views). There was a `ui` package with widgets of its own; it went
  the way Swift's did, so the two SDKs stay one shape.
- `chat` depends on `stream_chat`, which is pure Dart, so it is tested in `dart:stable` like
  `core`. Import it with a prefix (`as stream`): it exports a `TokenProvider` of its own.
- Stream's packages are pinned to exact versions (`stream_video_flutter: 1.6.0`,
  `stream_chat: 10.4.0`). Bumping them is a deliberate change: read the changelog,
  rerun the rtc tests, and check the APIs used (listed below) in the new source.
- `pubspec.lock` is gitignored: a library leaves resolving to the app.

Stream Video's singleton: use `StreamVideo.create`, never the `StreamVideo(...)` factory, which
installs a process-wide instance and collides with a host that has its own.

## Stream is set up once

The key, `userId` and token are the one identity for the router, Stream Chat and Stream Video.
`Backend.shared(kind, open:, disconnect:)` holds one client per `kind:apiKey:userId`: sessions
asking at once share one opening, one that failed to open is opened again, and `withUser` keeps
the registry so another user gets their own. `Backend.give` holds the app's own client, which
`VisionAgents.disconnect()` never touches; it closes only what `shared` built. Beside a
`customerId` the key is Stream's alone and the router is reached by customer id.

- `vision_agents_chat`: `agents.useChat(client)` and `session.chat()`, which answers the
  conversation's `Channel`, unwatched. A session with no conversation is a `StateError`.
- `vision_agents_rtc`: `agents.useVideo(client)` and `VoiceSession.join()`, which joins the
  session's `callType:callId`. Leaving disposes nothing.
- Dart has no overloading, hence `useChat` and `useVideo` rather than Kotlin's `use`. A
  client for another user, or a missing key, user or token, is an `ArgumentError`.

## State

`LiveValue<T>`: a `value` and a `stream` that emits the current value first, then each change.
`LiveValueController` skips equal values and conflates while a listener is paused. No
dependency on provider, riverpod, bloc or rxdart — a host adapts it in a line, such as a
`StreamBuilder` over `stream`.

The state machine is an **immutable value**, `Conversation`, with `apply(event)` returning the
next value (or `this` when nothing changed, so equality skips a rebuild). That is what makes it
testable without a network, and it is where new event handling goes.

`AgentSession.events()` hands each caller its own bounded stream (256), and finishes a listener
that falls that far behind rather than buffering without bound. Never one shared broadcast
stream property. Answering a tool call does not depend on anybody listening: it happens inside
`AgentSession`'s own read loop.

## The socket

`package:web_socket`, one read loop per connection.

- A close with 1000 (the router's "the session ended") or one we asked for finishes the stream;
  anything else throws `SocketClosedException` with the code.
- A frame that cannot be decoded is skipped, not fatal.
- Closing is idempotent and sends 1000.
- **No automatic reconnection.** `respond` and `tool_result` are not idempotent and the protocol
  has nothing to resume from. `AgentSession.connection` reports `Disconnected(failure)` and the
  caller decides.
- A tool result echoes `request_id` and `turn_id` only when the call carried a `request_id`; a
  bare `turn_id` routes to a path that fails for a persisted session.

Frames in tests are quoted verbatim from `frameOf` in
[`sessionws.go`](../../../acceleration/internal/api/sessionws.go).

## The generated client

`core/tool/generate.dart` reads the spec and writes `lib/src/generated/api.dart`: one model
class per schema reached by an allowlisted operation, and an `Operations` class with a method
per operation. Committed, formatted, never hand-edited, **never exported**. Every public type is
hand-written in `models.dart` and mapped in `convert.dart`, so a spec field is invisible until
it is wrapped and the SDK's semver is not the schema's.

- The allowlist (`operations`, `sockets`) fails generation if an id is missing or not marked
  `x-client-accessible`. Adding an endpoint means adding its id and rerunning.
- `--check` regenerates in memory and fails if the committed file differs. Run it in CI.
- Enums are strings in generated code and Dart enums with an `unknown` value in `models.dart`.
- Dates go through `parseRouterDate`: Go writes up to nine fractional digits, Dart reads six.
- Generated `toJson` omits nulls. **Null means omit** and let the config or the router decide,
  never a copy of the server's default.
- A wrong type in an answer throws `UnreadableException` naming the field, not a `TypeError`.

## Errors

`sealed class AgentsException implements Exception`: `RouterException` (status, message, the
operation; `isServerSideOnly` on 403), `TransportException` (no answer at all),
`UnreadableException` (an answer that is not what the spec says), `SocketClosedException`.
Callers `switch` over it exhaustively. Never classify by parsing a message. Argument mistakes
are `ArgumentError`, thrown before any I/O.

`RouterException` also carries the envelope `{"error": {"message", "type", "code", "doc_url"}}`
as `type` (a `RouterErrorType`, `unknown` for one newer than the SDK), `code` (an open string
to branch on), `docUrl` and `requestId` (the `X-Request-Id` header, what a 500 is quoted by).
`_refusalOf` in `wire.dart` reads it by hand, since the generator only reaches request and
success schemas; any other body keeps its trimmed text (at most 1000 characters) or the
status phrase as the message, with the request id and nothing else. A refused socket
handshake stays a `TransportException`: `package:web_socket` gives a message only, no status,
headers or body.

## Public API

```dart
final agents = VisionAgents(url: url, customerId: 'acme');
final chat = await agents.agent('support').chat();
```

The constructor does no I/O. Advanced forms take an options value (`SessionOptions`,
`ForkOptions`, `SessionQuery`, `SearchOptions`), not thirty named parameters.

Names chosen to not collide with what a Flutter app imports: `SearchRouter` (not `Router`),
`AgentImage` (not `ImageSource`), commands suffixed `Command`. Stream Chat also exports a
`Command`; a host importing both hides one.

Guests: `guestUser` reuses what the `GuestStore` holds until a minute before it expires, then
mints again under the same id. The store is the host's (shared preferences, secure storage);
`MemoryGuestStore` is for tests and CLIs.

## Views

`rtc` has the only widgets, `AgentVideoView` and `VoiceCallView`: no theme object, no route push
and no `Scaffold`. A conversation is shown with Stream Chat's AI components, and
[`flutter_demo`](../../../examples/voice_agents/flutter_demo) is where that composition lives and
is kept working. What the router writes is what it renders:

- A text session's reply is a message marked `ai_generated`, with `generating` true until done.
- Its steps are `ai_reasoning` and `ai_tool_call` attachments, in order: `AIMessagePart.parseAll`
  over `(type: attachment.rawType, payload: {'id': attachment.id, ...attachment.extraData})`.
- A reasoning step stores only its opening. The whole of it comes a window at a time on the
  `message.updated` events; `LiveReasoning` in `chat` puts it back together, and the demo's
  `AIMessage.reasoningText` reads it.
- A tool that asks first is answered by an `AIToolApprover` whose `decide` calls
  `session.decide`; on a call, by an `AIToolApprovalCard` over `session.approvals`.
- `AIConversationView` lays the channel out in turns, `ChatComposer` asks and stops, and
  `AIVoiceGlowView` / `AIVoiceWaveView` show the agent on a call.

## rtc

The Stream Video 1.6 surface used: `StreamVideo.create(apiKey, user: User.regular(...),
userToken:, tokenLoader:)`, `makeCall(callType:, id:)`, `call.join(connectOptions:
CallConnectOptions(microphone:, camera:, cameraFacingMode: FacingMode.environment,
speakerDefaultOn: true))` (which creates the call too), `setMicrophoneEnabled`,
`setCameraEnabled(constraints: CameraConstraints(...))`, `call.leave()`, `dispose()`,
`call.state.valueStream`, `StreamVideoRenderer`. Results are `Result<T>`; a `Failure` becomes
`VoiceState.failure`.

- A token provider, not a token: `tokenLoader` asks the agents' token function again when the
  token expires an hour in.
- The camera starts on the back lens as a join setting, so no front-facing frame is published.
- Permissions are the host app's: `RECORD_AUDIO`, `MODIFY_AUDIO_SETTINGS`, `CAMERA`,
  `BLUETOOTH_CONNECT`, `INTERNET`, `ACCESS_NETWORK_STATE`; `NSMicrophoneUsageDescription`,
  `NSCameraUsageDescription`. Emulators do not test echo cancellation or audio routing.

## Tests

`package:test` in core and chat, `flutter_test` in rtc. Never mock; assert on state and outputs,
never on a call happening.

1. **The state machine**, on real frames, with no network.
2. **The client** against a real HTTP and WebSocket server on loopback (`test/support/router.dart`
   in core; rtc has its own small one), recording what arrived.
3. **Live tests**, tagged `live` and skipped unless `VISION_AGENTS_URL` is set: real sessions, a
   real model answer, a real tool call, rewind and fork, guests, search.
4. **The posture**: `posture_test.dart` drives every public operation against the loopback
   router and checks each request against `x-client-accessible` in the spec;
   `keyed_router_posture_test.dart` (tagged `live`, skipped without `INTEGRATION_ROUTER_KEY`)
   checks a real `api_key` router answers a device 403 `server_side_only`.
   `stream_clients_test.dart` covers the shared clients.

Wait on a condition with a deadline (`_until`), never a fixed sleep.

CI is [`client-sdks.yml`](../../../.github/workflows/client-sdks.yml), on any change under
`sdks/dart` or to the spec: core's and chat's checks and tests, as below. `rtc` is not
in it, so run them yourself when you change them.

Docker only, pub's cache in the named volume `va-dart-pub`:

```bash
# core
docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
  -w /repo/sdks/dart/core dart:stable \
  sh -c 'dart pub get && dart format --set-exit-if-changed . && dart analyze && dart run tool/generate.dart --check && dart test'
# chat: the same, from sdks/dart/chat, without the generator check
# rtc
docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
  -w /repo/sdks/dart/rtc ghcr.io/cirruslabs/flutter:3.44.0 \
  sh -c 'flutter pub get && flutter analyze && flutter test'
# live, against a router on the host
docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
  -e VISION_AGENTS_URL=http://host.docker.internal:8080 -e VISION_AGENTS_CUSTOMER_ID=examples \
  -e VISION_AGENTS_AGENT=simple_voice_ai -w /repo/sdks/dart/core dart:stable dart test -t live
```

## Reviewing a change

Reject it if it:

- imports Flutter, `dart:io` or a Stream package into `core/lib`;
- exports generated code, or hand-edits it;
- adds a method for an operation the spec does not mark `x-client-accessible`;
- sends a default the caller did not choose;
- reconnects the socket, or replays `respond`/`tool_result`;
- treats an unknown frame as fatal;
- uses the `StreamVideo` singleton factory;
- adds conversation views to the SDK rather than to Stream Chat's AI components;
- adds a theme object, a route push or a `Scaffold` to `rtc`;
- mocks, or asserts that a method was called.
