# Flutter demo

The Flutter half of [swift_demo](../swift_demo): the same agent, talked to by a Flutter app on
iOS and Android. The agent, its skill and its knowledge, and the two Go programs that store it and
sign the phone's Stream token, are all swift_demo's. Only the app is here.

```
pubspec.yaml   builds the app against sdks/dart as it is in this checkout
lib/           the phone half: chat, voice, and tools that run on the device
```

## Run it

Start the router and store the agent as [swift_demo's README](../swift_demo/README.md) says:
`docker compose up --build` from the repo root with `ROUTER_AUTH_MODE=proxy`, then
`go run ./configure` and `go run ./backend` in `examples/voice_agents/swift_demo`.

Then run the app on a simulator or emulator, with the `STREAM_API_KEY` the router runs with:

```bash
flutter run --dart-define=STREAM_API_KEY=<the router's STREAM_API_KEY>
```

The iOS simulator reaches your Mac's localhost, and the Android emulator reaches it as
`10.0.2.2`, so it works as it stands. On a device, add `--dart-define=ROUTER_URL=...` and
`--dart-define=BACKEND_URL=...` with your Mac's address on the network, for example
`http://192.168.1.20:8080` and `http://192.168.1.20:8099`.

The app depends on `sdks/dart/core`, `chat` and `rtc` by path, so a change to the SDK shows up in
the app without publishing it. Stream Chat's AI components come from the `stream-chat-flutter-ai`
branch that adds the conversation view and the voice glow, until they are released.

The Chat tab asks for `xai/grok-4.7`, a model that streams its reasoning, so the router needs
`XAI_API_KEY`; a call keeps the agent's faster model.

## What the app shows

- **Chat** is a text session: no call is joined, nothing is transcribed or spoken, and the
  replies still come through the same model with the same instructions, skills and knowledge a
  call would have had. The session is kept in a Stream Chat channel, which the router writes and
  the app shows with Stream's AI components (`stream_chat_flutter_ai`):
  - `AIConversationView` lays the conversation out in turns. The reply streams in below the
    question without moving it, and `AIReplyActionsView` copies and rates each finished reply.
  - The model's reasoning shows as it thinks (`StreamingReasoningView`), reassembled from the
    live updates by `LiveReasoning`.
  - Each tool call shows as it runs (`AIToolCallView`).
  - What the agent is working on before it answers shows as `AITypingIndicatorView`.
  - `ChatComposer` asks, dictates and stops a reply, and `AISuggestionsView` starts a
    conversation.
- The app is drawn in Stream's design system (`stream_core_flutter`, which
  `stream_chat_flutter_ai` is built on and re-exports): `StreamTheme` in light and dark,
  `StreamAppBar`, `StreamBottomNavBar`, `StreamButton`, the sheet `showStreamSheet` opens with
  its `StreamSheetHeader`, and `StreamIcons`.
- **Voice** starts a session, which puts the agent on a call, and joins it over Stream Video as
  the same user, with the token `backend` signed.
  - The agent is Stream's voice glow (`AIVoiceGlowView`). It moves with the agent's voice in
    Stream's blue and with yours in red, from the call's audio levels.
  - Underneath it, a word says what the agent is doing: "Listening", "Thinking", "Working on
    refund decision", "Speaking".
  - The button at the top right picks the glow or Stream's voice wave (`AIVoiceWaveView`). The
    last thing said is captioned at the bottom.
  - The call carries on while the Chat tab is shown, and ends with the hang-up button or when
    the agent leaves.
- **`lookup_order`** is a tool the model calls that runs in `lib/demo.dart`. Its data never
  leaves the phone; the agent asks, and only sees the answer. Ask about order `A-1042` or
  `A-1043`.
- **`issue_refund`** asks first. Ask for a refund on `A-1042`: once the refund skill says it is
  owed, the agent calls it, and the call waits for you.
  - In Chat the question is on the reply's step (`AIToolApprovalView`, through an
    `AIToolApprover` that answers with `session.decide`).
  - On a call it is a card above the controls (`AIToolApprovalCard`).
  - Choosing Refund runs the tool; choosing Not now tells the agent you declined.
