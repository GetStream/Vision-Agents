## 0.1.0

First release of `stream_agents_chat`, which keeps a Stream Agents conversation in a Stream Chat
channel. Pure Dart, so it runs anywhere `stream_agents_core` does.

✅ Added

- `session.chat()` - the Stream Chat `Channel` a text or voice session is kept in, to render with
  `stream_chat_flutter_ai` or read with `stream_chat`.
- `agents.useChat(client)` - opens the channels on the app's own `StreamChatClient`. Without it,
  the SDK connects one for the user and reuses it for that user's sessions, and
  `agents.disconnect()` closes it.
- `LiveReasoning` - puts an agent's reasoning back together, step by step, from the live
  message updates.
