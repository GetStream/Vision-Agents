# vision_agents_rtc

Talk to a Stream Vision Agent out loud: joins the call the agent is on with Stream Video.

```dart
final voice = await VoiceSession.start(agents, agent: 'support');
VoiceCallView(voice: voice, credentials: yourBackend.callCredentials)
```

The host app declares the microphone, and the camera if it shows one: `RECORD_AUDIO`,
`MODIFY_AUDIO_SETTINGS`, `CAMERA`, `BLUETOOTH_CONNECT`, `INTERNET` and `ACCESS_NETWORK_STATE` on
Android; `NSMicrophoneUsageDescription` and `NSCameraUsageDescription` on iOS.

See [the Dart SDKs](https://github.com/GetStream/Vision-Agents/tree/main/sdks/dart) for the rest.
