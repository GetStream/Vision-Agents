# Android demo

The Android half of [swift_demo](../swift_demo): the same agent, talked to by an Android app.
The agent, its skill and its knowledge, and the two Go programs that store it and sign the
phone's Stream token, are all swift_demo's. Only the app is here.

```
settings.gradle.kts   builds the app against sdks/kotlin as it is in this checkout
app/                  the phone half: chat, voice, and tools that run on the device
```

## Run it

Start the router and store the agent as [swift_demo's README](../swift_demo/README.md) says:
`docker compose up --build` from the repo root with `ROUTER_AUTH_MODE=proxy`, then
`go run ./configure` and `go run ./backend` in `examples/voice_agents/swift_demo`.

Then tell the app the Stream key, in `local.properties` here, which git ignores:

```properties
sdk.dir=/path/to/Android/sdk
demo.streamApiKey=<the router's STREAM_API_KEY>
```

And run it on an emulator, then ask about an order in the Chat tab or tap talk in the Voice tab:

```bash
./gradlew :app:installDebug
```

The emulator reaches your Mac's localhost as `10.0.2.2`, so it works as it stands. On a
device, add `demo.routerUrl` and `demo.backendUrl` with your Mac's address on the network,
for example `http://192.168.1.20:8080` and `http://192.168.1.20:8099`. From Android 17 either
one is the local network, which the app asks for when it opens.

The app builds against `sdks/kotlin` through an included build, so a change to the SDK shows
up in the app without publishing it. Stream Chat 7.15.0 is on `https://stream-io-repo.com`
rather than Maven Central, which `settings.gradle.kts` adds.

The Chat tab asks for `xai/grok-4.7`, a model that streams its reasoning, so the router needs
`XAI_API_KEY`; a call keeps the agent's faster model.

## What the app shows

- **Chat** is a text session: no call is joined, nothing is transcribed or spoken, and the
  replies still come through the same model with the same instructions, skills and knowledge a
  call would have had. The session is kept in a Stream Chat channel, which the router writes
  and the app shows with Stream's AI components (`stream-chat-android-ai-compose`): the reply
  streaming in (`StreamingText`), the model's reasoning as it thinks (`StreamingReasoning`,
  reassembled from the live updates by `LiveReasoning`), each tool call as it runs
  (`AIMessageParts`), and what the agent is doing before it answers (`AITypingIndicator`).
  `ChatComposer` asks, dictates and stops a reply; `AISuggestions` starts a conversation.
- **Voice** starts a session, which puts the agent on a call, and joins it over Stream Video as
  the user `setUser` named, with the token `backend` signed. It asks for the microphone first,
  and only the microphone: the call runs Stream's audio-call service. The agent is Stream's
  voice glow (`AIVoiceGlow`), moving with the agent's voice in Stream's colors and with yours
  in red, with a word for what it is doing underneath: "Listening", "Thinking", "Working on
  refund decision", "Speaking". The style button at the top right opens a sheet showing the
  glow and Stream's voice wave (`AIVoiceWave`), each moving, to pick from. What is said is
  captioned at the bottom, two lines at a time. The agent speaking is heard on its audio track;
  you speaking is the call's own speaking flag, set from what the phone sends after echo
  cancellation, because the microphone itself also hears the agent from the speaker. The call
  carries on while the Chat tab is shown, and ends with the hang-up button or when the agent
  leaves.
- **`lookup_order`** is a tool the model calls that runs in `Demo.kt`. Its data never leaves
  the phone; the agent asks, and only sees the answer. Ask about order `A-1042` or `A-1043`.
- **`issue_refund`** asks first. Ask for a refund on `A-1042`: once the refund skill says it is
  owed, the agent calls it, and the call waits for you. In Chat the question is on the reply
  (`AIToolApprover`); on a call it is a card above the controls (`AIToolApprovalCard`). Refund
  and the tool runs; Not now and the agent is told you declined.

What the agent is doing before it writes, "Thinking" or "Working on refund decision", comes
off the session's socket rather than Stream's AI indicator events: Stream Chat 7.15.0 cannot
read those as the router sends them, with their state under `custom`, and drops them.

## No auth

`Demo.kt` holds the whole configuration: the router's URL, the backend's URL, the customer id,
the Stream key and the agent's name. The router is running in the mode where it trusts the
customer and user ids it is given, and the key is Stream Chat's and Stream Video's alone. In
front of a real deployment the router is reached by the key and the same token, and nothing
else in the app would change.

The app is the user `demo-caller`, and a session belongs to whoever opened it, so it reaches
its own sessions and nothing else. `usesCleartextTraffic` in the manifest is what lets plain
HTTP through to the Mac.
