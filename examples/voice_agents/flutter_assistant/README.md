# Flutter assistant

The Flutter half of [swift_assistant](../swift_assistant): Juno, the same agent, talked to by a
Flutter app on iOS and Android. The agent, its instructions, and the two Go programs that store
it and sign the phone's Stream token are all swift_assistant's. Only the app is here, as with
[android_assistant](../android_assistant).

```
lib/       the phone half: Flutter over vision_agents_core/chat/rtc and Stream's AI components
ios/       Runner, with the calendar and reminders over EventKit (AppDelegate.swift)
android/   the app, with the calendar over the calendar provider (CalendarChannel.kt)
```

## Run it

Start the router in proxy mode, as in the [Swift demo](../swift_demo/README.md#run-it), so it
trusts the customer and user ids the phone sends. Then store the agent and leave its backend
running, both from swift_assistant, as [its README](../swift_assistant/README.md#run-it) says:

```bash
cd examples/voice_agents/swift_assistant
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
go run ./configure

STREAM_API_SECRET=<the router's STREAM_API_SECRET> \
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
go run ./backend
```

The backend listens on `:8098`. Run the app with the Stream key the router runs with:

```bash
cd examples/voice_agents/flutter_assistant
flutter run --dart-define=STREAM_API_KEY=<the router's STREAM_API_KEY>
```

The app reaches the router on `http://localhost:8080` and the backend on
`http://localhost:8098`, the phone's own localhost. The iOS simulator shares the Mac's. On
Android, a phone or an emulator, `adb reverse` hands it on to the Mac's, so run this first, and
again whenever the phone reconnects:

```bash
adb reverse tcp:8080 tcp:8080 && adb reverse tcp:8098 tcp:8098
```

To reach the Mac over the network instead, pass its address with
`--dart-define=ROUTER_URL=http://192.168.1.20:8080` and
`--dart-define=BACKEND_URL=http://192.168.1.20:8098`.

The chat asks for `xai/grok-4.7`, a model that reads photos and streams its reasoning, so the
router needs `XAI_API_KEY`. A call needs `GOOGLE_API_KEY`, `DEEPGRAM_API_KEY` and
`INWORLD_API_KEY`. Web search needs one of `EXA_API_KEY`, `TAVILY_API_KEY` or
`PERPLEXITY_API_KEY`. The AI components come from `stream-chat-flutter-ai`'s `main`, pinned,
until they are released.

## What the app shows

- **Conversations** are the router's, in the drawer the menu button opens: `sessions.query()`
  lists this user's, `sessions.search` finds them by title, and the router titles each one as it
  goes. They are under Today, Yesterday, This week and then their month, by when they started.
  Pull the list down to read it again; swipe one away to delete it. A new chat has no session
  until its first message is written, and one opened but never sent is deleted again on leaving
  it. One that ended carries on when it is next asked something.
- **Chat** is Stream's AI components (`stream_chat_flutter_ai`) over the session's channel.
  `AIConversationView` lays it out in turns: a question moves to the top and shows at once, and
  its reply writes itself out below it, with the model's reasoning as it thinks (reassembled
  from the live updates by `LiveReasoning`), each tool call as it runs, and the answer streaming
  in. Older messages load as you scroll up. Under a finished reply you can copy and rate it;
  ratings stay on the phone, as the router takes none yet. The composer floats over the bottom
  of the conversation, which fades away under it, and its options sheet has **Think harder**,
  which asks with a higher `reasoning_effort`.
- **Photos** from the composer's button go with the question, shrunk to 1280 pixels across,
  turned the way the camera held them, and inline as JPEG data URIs (`photos.dart`). The question
  shows them at once; the router uploads them to the conversation's channel and writes them on
  the person's message, where the chat shows them, and the model sees them with that question's
  reply.
- **Voice mode** is the waveform beside the composer. It starts voice on the open conversation,
  so the agent joins its call with the chat so far, and everything said on the call lands back in
  the chat. The chat's socket stays open and runs the tools, and a tool that asks first asks on a
  card over the call; the call is followed by a second session with none. The agent is Stream's
  voice glow, moving with its voice in Stream's colors and with yours in red, with a word for
  what it is doing and captions a sentence at a time, paced to its voice, as in the
  [Flutter demo](../flutter_demo). The microphone is asked for as voice mode opens. Hang up, the
  close button or Back take the agent off the call and go back to the chat.
- **The camera** button in voice mode shows Juno what the phone is pointing at, through the back
  lens, with a preview of it. Try it on a phone or the Android emulator, whose camera is a test
  scene; the iOS simulator has none.
- **Tools on the phone**, in `device_tools.dart`: `get_local_context` (date, time, timezone),
  `get_location` (asks first), `list_events` (the calendar), `add_reminder` (asks first) and
  `send_email` (asks first, showing who it goes to). A tool that asks first asks on its step in
  the reply. On iOS a reminder goes in Reminders; Android has no Reminders an app can add to, so
  there it is an event in the calendar new events go in, at the time it is due, with an alert
  then, and it takes a due time, as android_assistant's does. `send_email` has swift_assistant's
  backend send it through the router's proxy for the person's Gmail (`POST /gmail/send`).

The app is the user `juno-owner`, Alex, as the iOS and Android apps are, so all three phones
list the same conversations. `assistant.dart` holds the whole configuration: the router's and
the backend's URLs, the customer id, the Stream key, the agent's name and the user.

## What is not here yet

Each of these waits on Stream's Flutter AI components or on the Dart SDK, not on the app.

- **Signing in to Gmail and Linear.** A call that waits for a sign-in shows its step, with no
  Connect or Not now: the sign-in card is `AIToolAuthorizer` in GetStream/stream-chat-flutter-ai#41,
  not merged yet, and the Dart SDK has neither `SessionOptions.returnUrl`, which names where the
  sign-in comes back to, nor `AgentSession.declineLogin` for Not now (the Swift and Kotlin SDKs
  have both). Until then, connect Gmail and Linear from the iOS app, which is the same
  `juno-owner`, and this app uses those connections.
- **Looking through the camera.** Asked "what's this?", the agent's `vision` skill asks the call
  for what the camera showed. The Swift and Kotlin SDKs answer that themselves (`CameraFrames`);
  the Dart SDK does not yet, nor can it send images with a tool's result.
- **Sharing a reply**, which the iOS and Android components offer under every reply:
  GetStream/stream-chat-flutter-ai#44.
- **This install's id** on a tool's question (`AIClientIdentity.installId()`):
  GetStream/stream-chat-flutter-ai#43. Without it, a question is answered from any of the
  person's devices.
- The composer floats over the conversation, but the conversation does not know it is there:
  its scroll-to-latest button sits under the composer. `AIConversationView` has no bottom inset
  yet.
- The Think harder chip reads "Think harder" rather than iOS's shorter "Think", since
  `ChatOption` has no short title.
