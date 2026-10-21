# vision_agents_chat

The Stream Chat channel a Stream Vision Agents conversation is kept in.

```dart
final chat = await agents.agent('support').chat();
final channel = await chat.chat();
```

It opens on one `StreamChatClient` every session shares, connected with the agents' key, user
and token, or on the app's own handed over with `agents.useChat(client)`. `agents.disconnect()`
closes only the one built here. Only a text session that is not incognito keeps a channel.

Pure Dart, so it runs anywhere the core does. Stream Chat's AI components
(`stream_chat_flutter_ai`) render the channel; see
[`flutter_demo`](../../../examples/voice_agents/flutter_demo).

See [the Dart SDKs](https://github.com/GetStream/Vision-Agents/tree/main/sdks/dart) for the rest.
