# Official Stream Chat integration for Stream Agents

> Keeps a Stream Agents conversation in a Stream Chat channel, so it can be shown with Stream
> Chat's AI components and read like any other channel. Pure Dart, so it runs anywhere
> `stream_agents_core` does.

[![Pub](https://img.shields.io/pub/v/stream_agents_chat.svg)](https://pub.dev/packages/stream_agents_chat)

**Quick Links**

- [Register](https://dashboard.getstream.io/signup/) to get an API key for Stream
- [Agent Skills](https://getstream.io/agent-skills/) to build with Stream from your AI coding agent

## Overview

Stream Agents runs AI agents in Stream's backend. An agent is set up there, with its
instructions, tools and knowledge, and the backend does the rest: it runs the model, and on a
call it joins, listens, answers and speaks. An app opens a conversation with an agent, in
writing or out loud, and shows it to the person. These packages are the app's side, for Dart
and Flutter: they connect the app to the agent and keep the conversation in sync. The model and
the audio and video processing run in Stream's backend, not on the device.

A text or voice session is kept in a Stream Chat channel unless it is incognito. This package
opens that channel on a `StreamChatClient`, connected as the same user as the agents, so an app
can render the conversation with
[`stream_chat_flutter_ai`](https://pub.dev/packages/stream_chat_flutter_ai) or read it with
[`stream_chat`](https://pub.dev/packages/stream_chat).

| Package | What it adds |
| --- | --- |
| [`stream_agents_core`](https://pub.dev/packages/stream_agents_core) | The client and the conversation state. Pure Dart. |
| **`stream_agents_chat`** | The Stream Chat channel a conversation is kept in. Pure Dart. |
| [`stream_agents_rtc`](https://pub.dev/packages/stream_agents_rtc) | Talking to the agent out loud, over Stream Video. Flutter. |

## Getting Started

### Add dependency

Add this to your `pubspec.yaml`, using the latest version
[![Pub](https://img.shields.io/pub/v/stream_agents_chat.svg)](https://pub.dev/packages/stream_agents_chat):

```yaml
dependencies:
  stream_agents_core: ^0.1.0
  stream_agents_chat: ^0.1.0
  stream_chat: ^10.4.0 # for the Channel, Message and client types
  stream_chat_flutter_ai: ^0.2.0 # in a Flutter app: the conversation's UI
```

Then run:

```shell
dart pub get
```

The package has no platform-specific setup.

### Open the channel

`session.chat()` returns the channel the session's conversation is kept in. It is not watched
yet.

```dart
import 'package:stream_agents_chat/stream_agents_chat.dart';
import 'package:stream_agents_core/stream_agents_core.dart';
import 'package:stream_chat/stream_chat.dart' as stream;

final session = await agents.agent('support').chat();
final stream.Channel channel = await session.chat();
await channel.watch();
```

Import `stream_chat` with a prefix: it exports a `TokenProvider` of its own.

Asking an incognito session for its channel throws a `StateError`, since it keeps none.

### The Stream Chat client

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

## License

Apache License 2.0. See [LICENSE](LICENSE).
