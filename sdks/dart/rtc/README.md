# Official Flutter voice and video for Stream Agents

> Talk to a Stream agent out loud: joins the call the agent is on with Stream Video, renders
> the agent's video beside the device camera, and lets the agent look through the camera.

[![Pub](https://img.shields.io/pub/v/stream_agents_rtc.svg)](https://pub.dev/packages/stream_agents_rtc)

**Quick Links**

- [Register](https://dashboard.getstream.io/signup/) to get an API key for Stream

## Changelog

Check out the [changelog on pub.dev](https://pub.dev/packages/stream_agents_rtc/changelog) to
see the latest changes in the package.

## Overview

A voice session puts the agent on a Stream Video call. This package joins this device to the
same call, so the conversation is spoken, and gives a Flutter app the call controls and the
agent's video.

| Package | What it adds |
| --- | --- |
| [`stream_agents_core`](https://pub.dev/packages/stream_agents_core) | The client and the conversation state. Pure Dart. |
| [`stream_agents_chat`](https://pub.dev/packages/stream_agents_chat) | The Stream Chat channel a conversation is kept in. Pure Dart. |
| **`stream_agents_rtc`** | Talking to the agent out loud, over Stream Video. Flutter. |

To show the agent while it talks, use `AIVoiceGlowView` or `AIVoiceWaveView` from
[`stream_chat_flutter_ai`](https://pub.dev/packages/stream_chat_flutter_ai).

## Getting Started

### Add dependency

Add this to your `pubspec.yaml`, using the latest version
[![Pub](https://img.shields.io/pub/v/stream_agents_rtc.svg)](https://pub.dev/packages/stream_agents_rtc):

```yaml
dependencies:
  stream_agents_core: ^0.1.0
  stream_agents_rtc: ^0.1.0
```

Then run:

```shell
flutter pub get
```

### Platform setup

The app declares the microphone, and the camera if it lets the agent see.

**Android** (`AndroidManifest.xml`): `RECORD_AUDIO`, `MODIFY_AUDIO_SETTINGS`, `CAMERA`,
`BLUETOOTH_CONNECT`, `INTERNET` and `ACCESS_NETWORK_STATE`.

**iOS** (`Info.plist`): `NSMicrophoneUsageDescription` and `NSCameraUsageDescription`.

See [Stream Video's Flutter setup](https://getstream.io/video/docs/flutter/) for the rest of
what a Stream Video app needs.

### Start a call

```dart
import 'package:stream_agents_core/stream_agents_core.dart';
import 'package:stream_agents_rtc/stream_agents_rtc.dart';

final voice = await VoiceSession.start(agents, agent: 'support');
await voice.join();
```

`VoiceCallView` is the call's controls (mute, camera and hang up), and joins the call when it
is first built, so an app can show it instead of calling `join` itself:

```dart
VoiceCallView(voice: voice)
```

`voice.state` is a `LiveValue<VoiceState>`: whether the device is muted, whether the camera is
on, and why joining failed. `voice.session` is the `AgentSession`, with the same transcript as a
written conversation.

`voice.leave()` leaves the call, and ends the session only if this device started it.
`voice.end()` leaves the call and ends the session, whoever started it.

### Let the agent look through the camera

```dart
await voice.join(camera: true); // or later: await voice.setCameraEnabled(true);
```

With the camera on, when the agent's vision skill asks what the camera shows, the session
answers with pictures of the call's own video, taken then, as JPEGs. Nothing is taken until the
agent asks. `AgentVideoView(voice: voice)` shows the device camera with the agent's video inset
beside it.

### The Stream Video client

The call is joined on the app's own client when it hands one over, connected as the same user:

```dart
agents.useVideo(yourStreamVideo);
```

Otherwise the SDK connects a `StreamVideo` with the agents' API key, user and token the first
time a session joins, and reuses it for that user's calls on the same `StreamAgents`.
`agents.disconnect()` closes the client the SDK connected and leaves the app's own alone.

## Contributing

We welcome code changes that improve this library or fix a problem. Please follow the existing
conventions and add tests before submitting a Pull Request on
[GitHub](https://github.com/GetStream/Vision-Agents).

```shell
flutter test
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
