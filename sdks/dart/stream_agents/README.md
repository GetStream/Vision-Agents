# Official Flutter client for Stream Agents

> Talk to a Stream agent from Flutter: open a conversation, follow it as it happens, answer the
> agent's tool calls on the device, keep the conversation in Stream Chat, and talk to the agent
> out loud over Stream Video.

[![Pub](https://img.shields.io/pub/v/stream_agents.svg)](https://pub.dev/packages/stream_agents)

**Quick Links**

- [Register](https://dashboard.getstream.io/signup/) to get an API key for Stream
- [Agent Skills](https://getstream.io/agent-skills/) to build with Stream from your AI coding agent

## Overview

Stream Agents runs AI agents in Stream's backend. An agent is set up there, with its
instructions, tools and knowledge, and the backend does the rest: it runs the model, and on a
call it joins, listens, answers and speaks. An app opens a conversation with an agent, in
writing or out loud, and shows it to the person. This package is the app's side, for Flutter:
it connects the app to the agent and keeps the conversation in sync. The model and the audio and
video processing run in Stream's backend, not on the device.

It has three parts, all behind one import:

| Part | What it adds |
| --- | --- |
| core | The API client, the session socket, and the conversation as state an app can render. |
| chat | The Stream Chat channel a conversation is kept in, on one shared client. |
| rtc | Joining the agent's call over Stream Video, so the conversation can be spoken. |

It ships no conversation views. To show a conversation, use Stream Chat's AI components,
[`stream_chat_flutter_ai`](https://pub.dev/packages/stream_chat_flutter_ai).

## Getting Started

### Add dependency

Add this to your `pubspec.yaml`, using the latest version
[![Pub](https://img.shields.io/pub/v/stream_agents.svg)](https://pub.dev/packages/stream_agents):

```yaml
dependencies:
  stream_agents: ^0.1.0
  stream_chat: ^10.4.0 # for the Channel, Message and client types
  stream_chat_flutter_ai: ^0.2.0 # the conversation's UI
```

Then run:

```shell
flutter pub get
```

### Platform setup

A written conversation needs none. For a call, the app declares the microphone, and the camera
if it lets the agent see.

**Android** (`AndroidManifest.xml`): `RECORD_AUDIO`, `MODIFY_AUDIO_SETTINGS`, `CAMERA`,
`BLUETOOTH_CONNECT`, `INTERNET` and `ACCESS_NETWORK_STATE`.

**iOS** (`Info.plist`): `NSMicrophoneUsageDescription` and `NSCameraUsageDescription`.

See [Stream Video's Flutter setup](https://getstream.io/video/docs/flutter/) for the rest of
what a Stream Video app needs.

### Create the client

Create one `StreamAgents` per signed-in user and reuse it. The token is a Stream user token
from your backend, signed with your Stream secret. `token` is a function the SDK calls each time
it needs one, for every request to the router and when Stream Chat or Stream Video need a fresh
one, so cache the token in it rather than asking your backend each time.

```dart
import 'package:stream_agents/stream_agents.dart';

final agents = StreamAgents(
  apiKey: 'your_api_key',
  userId: 'jlahey',
  token: () => yourBackend.agentToken(),
);
```

### Start a conversation

```dart
final session = await agents.agent('support').chat();
await session.responses.create('What are your opening hours?');
```

`session.conversation` is a `LiveValue<Conversation>`: the current transcript and what the
agent is doing, plus a stream of every change. A widget renders it with a `StreamBuilder`:

```dart
StreamBuilder<Conversation>(
  stream: session.conversation.stream,
  initialData: session.conversation.value,
  builder: (context, snapshot) {
    final turns = snapshot.requireData.turns;
    return ListView(children: [for (final turn in turns) Text(turn.text)]);
  },
);
```

### Give the agent a tool that runs on the device

```dart
final lookup = AgentTool(
  name: 'lookup_order',
  description: "Look up one of the caller's orders by its order number.",
  parameters: AgentTool.strings({'order_id': 'the order number'}, required: ['order_id']),
  run: (arguments) async => orders.find('${arguments['order_id']}'),
);

final session = await agents.agent('support').chat(SessionOptions(tools: [lookup]));
```

The agent only sees what the tool returns.

## Keeping the conversation in Stream Chat

A text or voice session is kept in a Stream Chat channel unless it is incognito.
`session.chat()` returns that channel, so an app can render the conversation with
`stream_chat_flutter_ai` or read it with `stream_chat`. It is not watched yet.

```dart
import 'package:stream_chat/stream_chat.dart' as stream;

final stream.Channel channel = await session.chat();
await channel.watch();
```

Import `stream_chat` with a prefix: it exports a `TokenProvider` of its own.

Asking an incognito session for its channel throws a `StateError`, since it keeps none.

The channel opens on the app's own client when it hands one over, connected as the same user:

```dart
agents.useChat(yourChatClient);
```

Otherwise the SDK connects a `StreamChatClient` with the agents' API key, user and token the
first time a session asks, and reuses it for that user's sessions on the same `StreamAgents`.
`agents.disconnect()` closes the client the SDK connected and leaves the app's own alone.

### Following the agent's reasoning

The router stores only the start of each reasoning step. `LiveReasoning` puts the whole of it
back together from the live `message.updated` events:

```dart
final reasoning = LiveReasoning();
channel.client.on(stream.EventType.messageUpdated).listen((event) {
  if (event.message case final message?) reasoning.read(message);
});

// The thinking of reasoning step r1 of a reply, so far:
final text = reasoning[(message: replyId, step: 'r1')];
```

## Talking out loud

A voice session puts the agent on a Stream Video call. `VoiceSession` joins this device to the
same call, so the conversation is spoken.

```dart
final voice = await VoiceSession.start(agents, agent: 'support');
await voice.join();
```

`VoiceCallView` is the call's controls (mute, camera and hang up), and joins the call when it
is first built, so an app can show it instead of calling `join` itself:

```dart
VoiceCallView(voice: voice)
```

`voice.state` is a `LiveValue<VoiceState>`: whether the device is muted, whether the camera is
on, and why joining failed. `voice.session` is the `AgentSession`, with the same transcript as a
written conversation. To show the agent while it talks, use `AIVoiceGlowView` or
`AIVoiceWaveView` from `stream_chat_flutter_ai`.

`voice.leave()` leaves the call, and ends the session only if this device started it.
`voice.end()` leaves the call and ends the session, whoever started it.

### Let the agent look through the camera

```dart
await voice.join(camera: true); // or later: await voice.setCameraEnabled(true);
```

With the camera on, when the agent's vision skill asks what the camera shows, the session
answers with pictures of the call's own video, taken then, as JPEGs. Nothing is taken until the
agent asks. `AgentVideoView(voice: voice)` shows the device camera with the agent's video inset
beside it.

### The Stream Video client

The call is joined on the app's own client when it hands one over, connected as the same user:

```dart
agents.useVideo(yourStreamVideo);
```

Otherwise the SDK connects a `StreamVideo` with the agents' API key, user and token the first
time a session joins, and reuses it for that user's calls on the same `StreamAgents`.
`agents.disconnect()` closes the client the SDK connected and leaves the app's own alone.

## Main types

- `StreamAgents` - the client: `agent(name)`, `sessions`, `guestUser`, `withUser`, `useChat`,
  `useVideo`, `disconnect`
- `AgentSession` - one conversation: `responses`, `conversation`, `events()`, `decide`, `rewind`,
  `fork`, `update`, `chat`, `close`, `delete`
- `Conversation`, `Turn`, `ConversationState` - the transcript and what the agent is doing
- `LiveValue<T>` - a current value and a stream of its changes, with no state-management package
- `AgentTool`, `ToolOutput` - tools the agent can call on the device
- `Sessions`, `Responses` - past conversations and their turns
- `SessionOptions` - tools, memory, time zone, incognito and the other settings for a session
- `LiveReasoning` - an agent's reasoning, put back together from the channel's live updates
- `VoiceSession`, `VoiceCallView`, `AgentVideoView` - the call, its controls and its video
- `AgentsException` - what a failed call throws: `RouterException`, `TransportException` and
  the others

## License

Apache License 2.0. See [LICENSE](LICENSE).
