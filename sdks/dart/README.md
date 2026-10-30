# Dart SDKs

Four packages for talking to an agent from a Dart or Flutter app. The core and chat are pure
Dart, so the same conversation runs in a Flutter app, a CLI or a server-side Dart process; ui and
rtc are Flutter.

| Package | pub name | Depends on | What it is |
| --- | --- | --- | --- |
| `core/` | `vision_agents_core` | `http`, `web_socket` | The generated client, the session socket, and the conversation state |
| `chat/` | `vision_agents_chat` | `core`, `stream_chat` 10.4.0 | The Stream Chat channel a conversation is kept in, on one shared client |
| `ui/` | `vision_agents_ui` | `core`, `stream_chat_flutter_core` 10.4.0 | Flutter widgets over that state, and a conversation kept in Stream Chat |
| `rtc/` | `vision_agents_rtc` | `core`, `stream_video_flutter` 1.6.0 | Joining the call, so the conversation can be spoken |

They are four packages rather than one pub workspace. A workspace resolves as one, so a
Flutter member would make the pure-Dart core unresolvable without the Flutter SDK. The others
depend on `vision_agents_core` from pub.dev, and a committed `pubspec_overrides.yaml` points them
at `../core` in this repository.

Until they are on pub.dev, an app depends on them by git, and names the core in its own
`dependency_overrides`, since pub ignores a dependency's `pubspec_overrides.yaml`:

```yaml
dependencies:
  vision_agents_chat:
    git: {url: https://github.com/GetStream/Vision-Agents, ref: accelerate, path: sdks/dart/chat}
dependency_overrides:
  vision_agents_core:
    git: {url: https://github.com/GetStream/Vision-Agents, ref: accelerate, path: sdks/dart/core}
```

## Using them

```dart
final agents = VisionAgents(apiKey: 'your_api_key', userId: 'jlahey', token: yourBackend.agentToken);

// In writing. No call is joined, nothing is transcribed or spoken.
final chat = await agents.agent('support').chat();
await chat.responses.create('What are your opening hours?');
// chat.conversation.value is the transcript and what the agent is doing, and
// chat.conversation.stream says each time it changes. The question shows there at once.

// The Stream Chat channel the conversation is kept in, with vision_agents_chat.
final channel = await chat.chat();

// Out loud. The agent joins a call and so does this device.
final voice = await VoiceSession.start(agents, agent: 'support');
await voice.join();

// Signing out: closes the Stream Chat and Video clients the SDK built.
await agents.disconnect();
```

`VisionAgents(apiKey: ...)` reaches Stream's hosted router in us-east4; pass `url:` for another router. For another
region, pass `url: Backend.regionUrl('europe-west4')`. A router
running locally with nothing in front of it is reached by customer id instead:
`VisionAgents(url: Uri.parse('http://localhost:8080'), customerId: 'acme')`. Pass `apiKey`,
`userId` and `token` beside it and the key is Stream's alone, so chat and video connect on a
local router too.

Stream is set up once. The key, the user and the token are what the router, Stream Chat and
Stream Video all connect with. Nothing connects up front: the first session to ask for `chat()`
or `join()` builds the client, and every session after it shares that one, so the user is one
connection and one presence to Stream however many conversations are open. The token function
is asked again when Stream's token expires, an hour in. Leaving a call or closing a session
disconnects nothing; `agents.disconnect()` does.

An app that already has a `StreamChatClient` or a `StreamVideo` hands it over, connected as the
same user, and it is used instead of a second one and never disconnected here:

```dart
agents.useChat(yourChatClient);
agents.useVideo(yourVideoClient);
```

State is a `LiveValue`: the current value, and a stream that starts with it. Nothing here
depends on a state-management package; `LiveValueBuilder` and `LiveValueNotifier` in
`vision_agents_ui` adapt it to widgets and `Listenable`.

With `vision_agents_ui` a whole conversation is one widget:

```dart
ConversationView(session: chat)
```

`TranscriptView`, `Composer` and `AgentStatusView` are public and work on their own, so a host
that wants a different arrangement takes them apart rather than fighting `ConversationView`.
`AgentVideoView` and `VoiceCallView` are the same for a call.

### A tool that runs on the device

The agent runs in the backend; a tool you give it runs here. It can read what only the device
knows, and the agent only ever sees the answer.

```dart
final lookup = AgentTool(
  name: 'lookup_order',
  description: "Look up one of the caller's orders by its order number.",
  parameters: AgentTool.strings({'order_id': 'the order number'}, required: ['order_id']),
  displayTitle: 'Looking up your order',
  executor: ToolExecutor.client,
  run: (arguments) async => Orders.local.find('${arguments['order_id']}'),
);

final chat = await agents.agent('support').chat(SessionOptions(tools: [lookup]));
```

`displayTitle` is what the people in a conversation kept in Stream Chat see while it runs, and
`ToolExecutor.client` shows the call as waiting on this person's device. Leave either out and
the router decides.

A tool can show the model pictures as well as words, such as a photo the person picked.
`AgentTool.withPictures` returns a `ToolOutput`; each picture is a data URI or an https URL,
and one socket message holds at most 5 MB, so shrink a photo before returning it.

```dart
final photograph = AgentTool.withPictures(
  name: 'photograph_room',
  description: 'Take a photo of the room.',
  run: (_) async => ToolOutput(text: 'the room', images: [AgentImage(await camera.jpegDataUri())]),
);
```

A call the router cancels, because its timeout passed or the caller interrupted it, is not
answered: the tool's future runs to its end and what it returns is dropped.

### Showing the agent the camera

On a call, turn the camera on and the agent can look through it:

```dart
await voice.join(camera: true); // or later: await voice.setCameraEnabled(true);
```

The camera is the back lens. When the agent's `vision` skill needs to see, the router asks the
session what the camera shows (`get_video_frames`), and `voice.session` answers with a picture of
the call's own video taken then, at most 1024 px on its longest side, as a JPEG. An agent set to
look at several frames gets them half a second apart. Nothing is taken until the agent asks, and
nothing is declared as a tool, so the model is never offered it. With the camera off the session
does not answer, so a video worker in your backend can. The agent needs the `vision` skill, and
the call a subagent to run it on.

Stream Video has no Dart hook on the frames a call sends, so each picture is a screenshot of the
local camera track at the camera's full size, scaled and re-encoded before it is sent. The Swift,
Kotlin and JS SDKs keep the last few seconds of frames, so theirs can come from just before the
question; here they come from just after it.

### A connector's sign-in

A connector call that needs the person to sign in waits on its step, `awaiting_authorization`.
`returnUrl` names where the sign-in comes back to: the app's own scheme in reverse domain form,
since the dashboard, where it goes otherwise, cannot be opened on a phone. "Not now" is
`declineLogin`, which the router answers itself: the call runs nothing and the agent is told the
person did not connect.

```dart
final chat = await agents.chat(SessionOptions(returnUrl: 'com.example.app:/connected'));
chat.declineLogin(toolCallId);
```

### Remembering the user

`memory` has the agent recall what it learned about the user in earlier sessions and keep what
it learns in this one. It is always the `userId` the client was built with: a device recalls
nobody else's, and a client with no `userId` remembers nothing. `filter` narrows what is
recalled. `timezone` is the IANA zone the agent reads dates in; Dart's `DateTime.timeZoneName`
is an abbreviation, so read the zone with a plugin such as `flutter_timezone`.

```dart
final chat = await agents.agent('support').chat(
  const SessionOptions(
    memory: SessionMemory(filter: {'topic': 'billing'}),
    timezone: 'Europe/Amsterdam',
  ),
);
```

### Going back, and branching off

`rewind` takes back every turn after the one kept, so the next message carries on from there.
`fork` branches a new session off one instead, leaving the original as it was. A turn's id is
the `id` of an `AgentResponse`, not the `turnId` a socket event carries:

```dart
final turns = (await chat.responses.list()).items;
await chat.rewind(turns.first.id);
final branch = await chat.fork(ForkOptions(responseId: turns.first.id));
await branch.start();
```

A text or voice session is kept in Stream Chat unless it is `incognito`, and a conversation
kept there cannot be rewound, since its transcript would bring the turns back; fork it at the
response instead.

### Renaming a conversation

`update` changes the title, description and custom labels, and leaves out what is left null.
Those are all a device may change; instructions and models are its backend's. An ended session
is renamed through `sessions`:

```dart
await chat.update(title: 'Refund for order A-1042');
await agents.sessions.update(endedId, title: 'Refund', custom: {'topic': 'billing'});
```

### Ending a conversation, and deleting one

`close` stops the session: the agent leaves, and the turns and what it remembered are kept, so
it is still listed and can be forked. `delete` takes it away, turns and memories included.

```dart
await chat.close();
await agents.sessions.delete(chat.id);
```

### Guests, finding conversations, looking something up

```dart
final guest = await agents.guestUser(name: 'Ada', store: yourGuestStore);
final asGuest = agents.withGuest(guest);

final found = await asGuest.sessions.search('refund');
final live = await asGuest.agent('support').sessions.query(
  const SessionQuery(state: SessionState.live, limit: 20),
);
final more = await asGuest.agent('support').sessions.query(
  SessionQuery(state: SessionState.live, limit: 20, cursor: live.nextCursor),
);
final thisWeek = await agents.sessions.query(
  SessionQuery(
    createdSince: DateTime.now().subtract(const Duration(days: 7)),
    custom: {'topic': 'billing'},
    sort: SessionSort.updatedAt,
  ),
);

// The router comes from the client; what it routes to lives in the "healthcare" config.
final answer = await agents.router(config: 'healthcare').search('perioperative antibiotic guidance');
```

`VisionAgents(apiKey: ...)` with no token asks for a guest anonymously and may do nothing else
until `withGuest` or `withUser`. A router that admits no anonymous callers, Stream's hosted one
for now, throws a `StateError`, and your backend mints the guest instead.

A guest is kept in the `GuestStore` you hand it (shared preferences, secure storage) and reused
until it expires. Lists come a page at a time: ask again with the page's `nextCursor`, which is
null on the last one.

## What is deliberately not here

The router is server-side only by default: operations marked `x-client-accessible` in the spec
are open to a device and everything else answers 403. `tool/generate.dart` fails if its
allowlist names anything the spec does not open, which stops this SDK growing a method that only
ever fails.

| Not here | Ask your backend for |
| --- | --- |
| Writing a config, syncing a folder, dispatching agents | The agent name, which is all the app needs |
| Minting the user's token | The token the router, Stream Chat and Stream Video all connect with |
| Claiming a guest's history for a signed-in user | `claimGuestUser`, which is server-side only |
| Changing a session's models | `updateSession` with them, which only a backend may send |

Every request sends `Stream-Auth-Type: jwt`, and every socket sends it as the
`stream-auth-type=jwt` query parameter, since a browser socket cannot set a header. It declares
this caller a device, even against a local router with no proxy in front. A socket opened with a
key also sends its token as `authorization`, which is how Stream's proxy reads it.

## Regenerating the client

The spec at [`acceleration/api/openapi.yaml`](../../acceleration/api/openapi.yaml) is the source
of truth. The generated models and operations in `core/lib/src/generated/api.dart` are committed
and never exported: every public type is hand-written, so a field added to the spec is invisible
until it is wrapped.

```bash
docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
  -w /repo/sdks/dart/core dart:stable sh -c 'dart pub get && dart run tool/generate.dart --check'
```

Drop `--check` to regenerate.

## Tests

Everything builds and runs in Docker, with pub's cache in a named volume. The core in the
official Dart image, offline:

```bash
docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
  -w /repo/sdks/dart/core dart:stable sh -c 'dart pub get && dart analyze && dart test'
```

`chat` the same way, from `sdks/dart/chat`.

The Flutter packages in Cirrus Labs' Flutter image:

```bash
for pkg in ui rtc; do
  docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
    -w /repo/sdks/dart/$pkg ghcr.io/cirruslabs/flutter:3.44.0 \
    sh -c 'flutter pub get && flutter analyze && flutter test'
done
```

Nothing is mocked: the unit tests run the client against a real HTTP and WebSocket server on
loopback. Against a running router, the core's `live` tests create real sessions, ask the model
something and wait for a tool call to come back:

```bash
docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
  -e VISION_AGENTS_URL=http://host.docker.internal:8080 -e VISION_AGENTS_CUSTOMER_ID=examples \
  -e VISION_AGENTS_AGENT=simple_voice_ai \
  -w /repo/sdks/dart/core dart:stable dart test -t live
```

Without `VISION_AGENTS_URL` they are skipped.

`posture_test.dart` checks offline that every request the SDK makes is one the spec marks
`x-client-accessible`. `keyed_router_posture_test.dart` checks a real router in `api_key` mode
refuses a device the server-side operations, and is skipped without `INTEGRATION_ROUTER_KEY`;
set the router up as in the [JavaScript README](../js/README.md):

```bash
docker run --rm -v "$PWD":/repo -v va-dart-pub:/pub-cache -e PUB_CACHE=/pub-cache \
  -e INTEGRATION_ROUTER_URL=http://host.docker.internal:8099 \
  -e INTEGRATION_ROUTER_KEY=vak_... -e INTEGRATION_ROUTER_SECRET=vas_... \
  -w /repo/sdks/dart/core dart:stable dart test -t live test/keyed_router_posture_test.dart
```
