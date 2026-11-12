# Official Dart client for Stream Agents

> Talk to a Stream agent from Dart: open a conversation, follow it as it happens, answer the
> agent's tool calls on the device, and find past conversations. Pure Dart, so it runs in
> Flutter on every platform, on the Dart VM and in a browser.

[![Pub](https://img.shields.io/pub/v/stream_agents_core.svg)](https://pub.dev/packages/stream_agents_core)

**Quick Links**

- [Register](https://dashboard.getstream.io/signup/) to get an API key for Stream

## Changelog

Check out the [changelog on pub.dev](https://pub.dev/packages/stream_agents_core/changelog) to
see the latest changes in the package.

## Overview

The agent runs in Stream's backend. This package is the client side of a conversation with it:
the API client, the session socket, and the conversation as state an app can render. It ships
no UI and needs no other Stream package.

| Package | What it adds |
| --- | --- |
| **`stream_agents_core`** | The client and the conversation state. Pure Dart. |
| [`stream_agents_chat`](https://pub.dev/packages/stream_agents_chat) | The Stream Chat channel a conversation is kept in. Pure Dart. |
| [`stream_agents_rtc`](https://pub.dev/packages/stream_agents_rtc) | Talking to the agent out loud, over Stream Video. Flutter. |

To show a conversation in Flutter, use Stream Chat's AI components,
[`stream_chat_flutter_ai`](https://pub.dev/packages/stream_chat_flutter_ai).

## Getting Started

### Add dependency

Add this to your `pubspec.yaml`, using the latest version
[![Pub](https://img.shields.io/pub/v/stream_agents_core.svg)](https://pub.dev/packages/stream_agents_core):

```yaml
dependencies:
  stream_agents_core: ^0.1.0
```

Then run:

```shell
dart pub get
```

The package has no platform-specific setup.

### Create the client

Create one `StreamAgents` per signed-in user and reuse it. The token is a Stream user token
from your backend, signed with your Stream secret. `token` is a function the SDK calls each time
it needs one, for every request to the router and when Stream Chat or Stream Video need a fresh
one, so cache the token in it rather than asking your backend each time.

```dart
import 'package:stream_agents_core/stream_agents_core.dart';

final agents = StreamAgents(
  apiKey: 'your_api_key',
  userId: 'jlahey',
  token: () => yourBackend.agentToken(),
);
```

`StreamAgents` reaches Stream's hosted router in us-east4. For another region, pass
`url: Backend.regionUrl('europe-west4')`.

### Start a conversation

```dart
final session = await agents.agent('support').chat();
await session.responses.create('What are your opening hours?');
```

`session.conversation` is a `LiveValue<Conversation>`: the current transcript and what the
agent is doing, plus a stream of every change. A Flutter widget renders it with a
`StreamBuilder`:

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

## Main types

- `StreamAgents` - the client: `agent(name)`, `sessions`, `guestUser`, `withUser`, `disconnect`
- `AgentSession` - one conversation: `responses`, `conversation`, `events()`, `decide`, `rewind`,
  `fork`, `update`, `close`, `delete`
- `Conversation`, `Turn`, `ConversationState` - the transcript and what the agent is doing
- `LiveValue<T>` - a current value and a stream of its changes, with no state-management package
- `AgentTool`, `ToolOutput` - tools the agent can call on the device
- `Sessions`, `Responses` - past conversations and their turns
- `SessionOptions` - tools, memory, time zone, incognito and the other settings for a session
- `AgentsException` - what a failed call throws: `RouterException`, `TransportException` and
  the others

## Contributing

We welcome code changes that improve this library or fix a problem. Please follow the existing
conventions and add tests before submitting a Pull Request on
[GitHub](https://github.com/GetStream/Vision-Agents).

```shell
dart test
```

The API client in `lib/src/generated` is generated from the router's OpenAPI spec
(`acceleration/api/openapi.yaml` in the repository) and committed. Regenerate it with
`dart run tool/generate.dart`; `--check` verifies it is up to date.

## License

Apache License 2.0. See [LICENSE](LICENSE).
