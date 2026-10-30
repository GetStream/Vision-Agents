# vision_agents_rtc

Talk to a Stream Vision Agent out loud: joins the call the agent is on with Stream Video.

```dart
final voice = await VoiceSession.start(agents, agent: 'support');
VoiceCallView(voice: voice)
```

It joins with the agents' key, user and token, on one `StreamVideo` every voice session shares,
or on the app's own handed over with `agents.useVideo(client)`.

With the camera on (`voice.join(camera: true)` or `voice.setCameraEnabled(true)`, on the back
lens) the agent can look through it: when the agent's vision skill asks what the camera shows,
the session takes a picture of the call's own video then, or several half a second apart, and
answers with them as JPEGs at most 1024 px on their longest side. Nothing is taken until the agent
asks.

The host app declares the microphone, and the camera if it shows one: `RECORD_AUDIO`,
`MODIFY_AUDIO_SETTINGS`, `CAMERA`, `BLUETOOTH_CONNECT`, `INTERNET` and `ACCESS_NETWORK_STATE` on
Android; `NSMicrophoneUsageDescription` and `NSCameraUsageDescription` on iOS.

See [the Dart SDKs](https://github.com/GetStream/Vision-Agents/tree/accelerate/sdks/dart) for the rest.
