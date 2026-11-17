## 0.1.0

First release of `stream_agents`, the Flutter client for Stream Agents. It holds what were
`stream_agents_core`, `stream_agents_chat` and `stream_agents_rtc`, with the same API, behind one
import: `package:stream_agents/stream_agents.dart`.

🔄 Changed

- It is a Flutter package, so a CLI or a server-side Dart process, which could use
  `stream_agents_core` and `stream_agents_chat`, cannot use it.
- An app gets `stream_chat` and `stream_video_flutter` with it whether it writes or talks, so
  both must resolve beside the app's own dependencies.

✅ Added

- `StreamAgents` - the client for one user: opens a written or spoken conversation with an agent
  (`agent(name).chat()`, `agent(name).voice()`), finds past conversations (`sessions`) and signs
  in guests (`guestUser`).
- `AgentSession` - one conversation: send a message with `responses.create`, follow it with
  `conversation`, and `rewind`, `fork`, `update`, `close` or `delete` it.
- `Conversation` - the transcript and what the agent is doing, as a `LiveValue` that a
  `StreamBuilder` renders.
- `AgentTool` - tools that run on the device; the agent sees only what they return.
  `AgentTool.withPictures` returns pictures as well as text.
- Tool approvals with `session.decide`, and connector sign-ins with `SessionOptions.returnUrl`
  and `session.declineLogin`.
- `SessionOptions` - memory of the user across sessions, the time zone, incognito sessions and
  the other settings of a session.
- `session.chat()` - the Stream Chat `Channel` a text or voice session is kept in, to render with
  `stream_chat_flutter_ai` or read with `stream_chat`.
- `agents.useChat(client)` - opens the channels on the app's own `StreamChatClient`. Without it,
  the SDK connects one for the user and reuses it for that user's sessions, and
  `agents.disconnect()` closes it.
- `LiveReasoning` - puts an agent's reasoning back together, step by step, from the live
  message updates.
- `VoiceSession` - starts a spoken conversation (`VoiceSession.start`) or attaches to one already
  running (`VoiceSession.attach`), and joins this device to the agent's call: `join`, `setMuted`,
  `setCameraEnabled`, `leave` and `end`.
- `VoiceCallView` - the call's controls: mute, camera and hang up.
- `AgentVideoView` - the device camera with the agent's video inset beside it.
- The agent can look through the camera: with the camera on, the session answers the agent's
  vision skill with pictures of the call's video, taken when it asks.
- `agents.useVideo(client)` - joins the calls on the app's own `StreamVideo`. Without it, the SDK
  connects one for the user and reuses it for that user's calls, and `agents.disconnect()` closes
  it.
