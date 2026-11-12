## 0.1.0

First release of `stream_agents_rtc`, for talking to a Stream agent out loud over Stream Video.

✅ Added

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
