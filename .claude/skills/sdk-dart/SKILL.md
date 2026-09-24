---
name: sdk-dart
description: How to build and extend the Dart and Flutter SDKs in sdks/dart. Read this before changing vision_agents_core, vision_agents_ui or vision_agents_rtc, or before adding a Dart client for a new endpoint.
---

# Dart SDK conventions

The per-language half of [sdk](../sdk/SKILL.md). It records the decisions the packages in
[`sdks/dart`](../../../sdks/dart) already follow, so a change lands consistently rather than
re-litigating them. It is a **client-side** SDK: what a phone, a browser or a desktop app does.
Folder sync, dispatch and writing configs are backend work and stay out.

Dart 3.10 is the floor for `core` and `rtc` (Stream Video 1.6's); `ui` is 3.12 / Flutter 3.44
because Stream Chat 10.4 is.

## Packages

```
core  vision_agents_core  pure Dart: generated client, socket, conversation state
ui    vision_agents_ui    Flutter widgets, Stream Chat transcript             -> core
rtc   vision_agents_rtc   joining the call over Stream Video                  -> core
```

**Three standalone packages, not a pub workspace.** A workspace resolves every member as one
graph, so once `ui` or `rtc` (Flutter) is a member, `core` cannot be resolved or tested with the
plain Dart SDK — and a pure-Dart core testable in `dart:stable` is the point of having one. The
price is three lockfiles and a local link: `ui` and `rtc` declare `vision_agents_core: ^x.y.z`
as a hosted dependency, and a committed `pubspec_overrides.yaml` points it at `../core`. Pub
never publishes that file and consumers never see it. Before releasing, publish `core` first
and check `ui`/`rtc` resolve against the hosted version with the override removed.

- `core` has no Flutter import and no Stream dependency. `dart:io` stays out of `lib/` too, so
  it runs on the web: `package:http` and `package:web_socket` pick the platform themselves.
- `rtc` keeps Stream Video's types out of `core`. `VoiceState.call` is the one Stream type it
  exposes, because rendering video needs it.
- `ui` depends on `stream_chat_flutter_core`, not `stream_chat_flutter`: the transcript is drawn
  with our own turns, and the full kit brings media pickers and their platform setup for nothing.
- Stream's packages are pinned to exact versions (`stream_video_flutter: 1.6.0`,
  `stream_chat_flutter_core: 10.4.0`). Bumping them is a deliberate change: read the changelog,
  rerun the rtc/ui tests, and check the APIs used (listed below) in the new source.
- `pubspec.lock` is gitignored: a library leaves resolving to the app.

Stream Video's singleton: use `StreamVideo.create`, never the `StreamVideo(...)` factory, which
installs a process-wide instance and collides with a host that has its own.

## State

`LiveValue<T>`: a `value` and a `stream` that emits the current value first, then each change.
`LiveValueController` skips equal values and conflates while a listener is paused. No
dependency on provider, riverpod, bloc or rxdart — a host adapts it in a line, and `ui` ships
`LiveValueBuilder` (widgets) and `LiveValueNotifier` (`ValueListenable`).

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
- A tool result echoes `command_id` and `turn_id` only when the call carried a `command_id`; a
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

`sealed class AgentsException implements Exception`: `RouterException` (status, the router's
`error`, the operation; `isServerSideOnly` on 403), `TransportException` (no answer at all),
`UnreadableException` (an answer that is not what the spec says), `SocketClosedException`.
Callers `switch` over it exhaustively. Never classify by parsing a message. Argument mistakes
are `ArgumentError`, thrown before any I/O.

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

## Flutter

- Builder slots (`bubble:`), not a theme object. Colours come from `Theme.of(context)`.
- No widget pushes a route, creates a `Navigator` or a `Scaffold`. Navigation is the host's.
- `ConversationView` starts the session but never closes it: the session outlives a screen.
- `TranscriptView` is a reversed `ListView.builder` keyed by turn id. Reversed so the newest turn
  sits at offset zero: a streaming reply stays in view without a scroll per delta, and a reader
  who scrolled up is left alone.
- `ChatTranscriptView` reads a persisted conversation from its Stream Chat channel. The router
  writes each message with a `support_message` payload whose `role` says who spoke, and the
  channel's `support_agent_id` names the agent; neither is in the spec, so `turnsOfMessages` is
  where a change on the router side lands.

## rtc

The Stream Video 1.6 surface used: `StreamVideo.create(apiKey, user: User.regular(...),
userToken:, tokenLoader:)`, `makeCall(callType:, id:)`, `call.join(connectOptions:
CallConnectOptions(microphone:, camera:, cameraFacingMode: FacingMode.environment,
speakerDefaultOn: true))` (which creates the call too), `setMicrophoneEnabled`,
`setCameraEnabled(constraints: CameraConstraints(...))`, `call.leave()`, `dispose()`,
`call.state.valueStream`, `StreamVideoRenderer`. Results are `Result<T>`; a `Failure` becomes
`VoiceState.failure`.

- A token provider, not a token: `tokenLoader` asks the host's backend again when the call token
  expires an hour in.
- The camera starts on the back lens as a join setting, so no front-facing frame is published.
- Permissions are the host app's: `RECORD_AUDIO`, `MODIFY_AUDIO_SETTINGS`, `CAMERA`,
  `BLUETOOTH_CONNECT`, `INTERNET`, `ACCESS_NETWORK_STATE`; `NSMicrophoneUsageDescription`,
  `NSCameraUsageDescription`. Emulators do not test echo cancellation or audio routing.

## Tests

`package:test` in core, `flutter_test` in ui and rtc. Never mock; assert on state and outputs,
never on a call happening.

1. **The state machine**, on real frames, with no network.
2. **The client** against a real HTTP and WebSocket server on loopback (`test/support/router.dart`
   in core; rtc has its own small one), recording what arrived.
3. **Live tests**, tagged `live` and skipped unless `VISION_AGENTS_URL` is set: real sessions, a
   real model answer, a real tool call, rewind and fork, guests, search.

Wait on a condition with a deadline (`_until`), never a fixed sleep.

Docker only, pub's cache in the named volume `va-dart-pub`:

```bash
# core
docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
  -w /repo/sdks/dart/core dart:stable \
  sh -c 'dart pub get && dart format --set-exit-if-changed . && dart analyze && dart run tool/generate.dart --check && dart test'
# ui, rtc
docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
  -w /repo/sdks/dart/ui ghcr.io/cirruslabs/flutter:3.44.0 \
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
- adds a theme object, a route push or a `Scaffold` to `ui`;
- mocks, or asserts that a method was called.
