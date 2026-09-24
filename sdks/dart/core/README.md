# vision_agents_core

The client for talking to a Stream Vision Agent from Dart: sessions, responses, the session
socket, and the conversation as state. Pure Dart, so it runs in Flutter, a CLI or a server.

```dart
final agents = VisionAgents(url: Uri.parse('https://your-router'), customerId: 'acme');
final chat = await agents.agent('support').chat();
chat.send('What are your opening hours?');
chat.conversation.stream.listen((conversation) => print(conversation.turns.last.text));
```

See [the Dart SDKs](https://github.com/GetStream/Vision-Agents/tree/main/sdks/dart) for the rest.
