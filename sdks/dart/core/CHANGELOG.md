## 0.1.0

First release of `stream_agents_core`, the Dart client for Stream Agents. Pure Dart, so it runs
in Flutter on every platform, on the Dart VM and in a browser.

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
