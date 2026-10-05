# Dart SDKs

Three packages for talking to an agent from a Dart or Flutter app. The core is pure Dart, so the
same conversation runs in a Flutter app, a CLI or a server-side Dart process; the other two are
Flutter.

| Package | pub name | Depends on | What it is |
| --- | --- | --- | --- |
| `core/` | `vision_agents_core` | `http`, `web_socket` | The generated client, the session socket, and the conversation state |
| `ui/` | `vision_agents_ui` | `core`, `stream_chat_flutter_core` 10.4.0 | Flutter widgets over that state, and a conversation kept in Stream Chat |
| `rtc/` | `vision_agents_rtc` | `core`, `stream_video_flutter` 1.6.0 | Joining the call, so the conversation can be spoken |

They are three packages rather than one pub workspace. A workspace resolves as one, so a
Flutter member would make the pure-Dart core unresolvable without the Flutter SDK. `ui` and `rtc`
depend on `vision_agents_core` from pub.dev, and a committed `pubspec_overrides.yaml` points them
at `../core` in this repository.

## Using them

```dart
final agents = VisionAgents(url: Uri.parse('https://your-router'), customerId: 'acme');

// In writing. No call is joined, nothing is transcribed or spoken.
final chat = await agents.agent('support').chat();
await chat.responses.create('What are your opening hours?');
// chat.conversation.value is the transcript and what the agent is doing, and
// chat.conversation.stream says each time it changes. The question shows there at once.

// Out loud. The agent joins a call and so does this device.
final voice = await VoiceSession.start(agents, agent: 'support');
await voice.join(credentials: yourBackend.callCredentials);
```

`join` is handed a function rather than minting its own token: minting a call token is
server-side only, so the app is handed the token to join with, and asks again when it expires.

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

A text session is kept in Stream Chat unless it is `incognito`, and a conversation kept there
cannot be rewound, since its transcript would bring the turns back; fork it at the response
instead.

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

// The router comes from the client; what it routes to lives in the "healthcare" config.
final answer = await agents.router(config: 'healthcare').search('perioperative antibiotic guidance');
```

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
| A token to join the agent's call | `CallCredentials`, which `join` is handed |
| A Stream Chat token | A connected `StreamChatClient` for `ChatTranscriptView` |
| Claiming a guest's history for a signed-in user | `claimGuestUser`, which is server-side only |
| Renaming a session or changing its models | `updateSession`, which is server-side only |

Every request and socket handshake sends `Stream-Auth-Type: jwt`, which declares this caller a
device, even against a local router with no proxy in front.

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
