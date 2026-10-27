# Android assistant

The Android half of [swift_assistant](../swift_assistant): Juno, the same agent, talked to by an
Android app. The agent, its instructions, and the two Go programs that store it and sign the
phone's Stream token are all swift_assistant's. Only the app is here.

```
settings.gradle.kts   builds the app against sdks/kotlin as it is in this checkout
app/                  the phone half: Compose over vision-agents-core and Stream's AI components
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

The backend listens on `:8098`. Tell the app the Stream key, in `local.properties` here, which
git ignores:

```properties
sdk.dir=/path/to/Android/sdk
assistant.streamApiKey=<the router's STREAM_API_KEY>
```

The app reaches the router on `http://localhost:8080` and the backend on
`http://localhost:8098`, the phone's own localhost, which `adb reverse` hands on to the Mac's.
That works the same on a device and an emulator:

```bash
adb reverse tcp:8080 tcp:8080
adb reverse tcp:8098 tcp:8098
ANDROID_HOME=/path/to/Android/sdk ./gradlew :app:installDebug
```

`ANDROID_HOME` is for `sdks/kotlin`, which the app builds through an included build, so a change
to the SDK shows up in the app without publishing it; that build does not read this directory's
`local.properties`. Run `adb reverse` again whenever the phone reconnects.

To reach the Mac over the network instead, add `assistant.routerUrl` and
`assistant.backendUrl` with its address, for example `http://192.168.1.20:8080` and
`http://192.168.1.20:8098`. From Android 17 that is the local network, which the app then asks
for when it opens.

The chat asks for `xai/grok-4.7`, a model that streams its reasoning, so the router needs
`XAI_API_KEY`. Web search needs one of `EXA_API_KEY`, `TAVILY_API_KEY` or `PERPLEXITY_API_KEY`.
Stream Chat is on `https://stream-io-repo.com` rather than Maven Central, which
`settings.gradle.kts` adds. The sign-in card and `AIConversation`'s `contentPadding` are only in
a snapshot of it so far, `7.16.1-202610271134-SNAPSHOT` from `https://snapshots.stream-io-repo.com`,
which is kept seven days; the app moves to the release that has them.

## What the app shows

- **Conversations** are the router's, in the drawer the menu button opens: `sessions.query()`
  lists this user's, `sessions.search` finds them by title, and the router titles each one as it
  goes. They are under Today, Yesterday, This week and then their month, by when they started.
  Pull the list down to read it again; hold one to delete it. A new chat has no session until
  its first message is written, and one opened but never sent is deleted again on leaving it.
  One that ended carries on when it is next asked something.
- **Chat** is Stream's AI components (`stream-chat-android-ai-compose`) over the session's
  channel. `AIConversation` lays it out in turns: a question moves to the top and shows at once,
  and its reply writes itself out below it, with the model's reasoning as it thinks
  (reassembled from the live updates by `LiveReasoning`), each tool call as it runs, and the
  answer streaming in. Older messages load as you scroll up. Under a finished reply you can
  copy, share and rate it; ratings stay on the phone, as the router takes none yet. The composer
  floats over the bottom of the conversation, which fades away under it.
- **Photos** from the composer's button, which opens the system photo picker, go with the
  question, shrunk to 1280 pixels across and inline as JPEG data URIs (`Photos.kt`). The question
  shows them at once; the router uploads them to the conversation's channel and writes them on
  the person's message, where the chat shows them, and the model sees them with that question's
  reply. From Android 9 a photo is turned the way the camera held it; before, it goes as stored.
- **Voice mode** is the waveform beside the composer. It starts voice on the open conversation,
  so the agent joins its call with the chat so far, and everything said on the call lands back in
  the chat. The chat's socket stays open and runs the tools, and a tool that asks first asks on a
  card over the call; the call is followed by a second session with none. The agent is Stream's
  voice glow (`AIVoiceGlow`), moving with its voice in Stream's colors and with yours in red, with
  a word for what it is doing and captions two lines at a time, as in the
  [Android demo](../android_demo/README.md#what-the-app-shows), whose `Captions.kt` and
  `VoiceMeter.kt` these are. The microphone is asked for as voice mode opens. Hang up, the close
  button or Back take the agent off the call and go back to the chat.
- **The camera** button in voice mode shows Juno what the phone is pointing at, through the back
  lens, with a preview of it: this phone's own video on the call, drawn with Stream Video's
  renderer. Ask "what's this?" and the agent's `vision` skill asks for what the camera showed at
  that moment, which the call's session answers with the last few frames. The camera is asked for
  the first time it is turned on. Try it on a phone: the emulator's camera is a test scene.
- **Tools on the phone**, in `DeviceTools.kt`: `get_local_context` (date, time, timezone),
  `get_location` (asks first; the platform's `LocationManager` and `Geocoder`), `list_events`
  (the calendar, through `CalendarContract`), `add_reminder` (asks first) and `send_email` (asks
  first, showing who it goes to). A tool that asks first asks on its step in the reply. The
  calendar and the location are each asked for the first time a tool needs them, and a tool the
  person refused tells the agent so. Android has no Reminders an app can add to, so a reminder is
  an event in the calendar new events go in, at the time it is due, with an alert then; it takes
  a due time, which iOS's may leave out. `send_email` has swift_assistant's backend send it
  through the router's proxy for the person's Gmail (`POST /gmail/send`).
- **Sign-ins go back to the app.** Every new conversation names
  `io.getstream.visionagents.androidassistant:/connected` as its session's `return_url`, the
  app's own scheme in the reverse domain form the router asks for. `SignIn.kt` is the browser
  half of a connector's sign-in: `authenticate(url)` opens `launch_url#handoff_token=<token>` in
  an Auth Tab (`androidx.browser` 1.10.0) and returns the `…:/connected` URL the router sent it
  back to, with `connection_id` and `status`, or null when the person closed it. Where the
  browser has no Auth Tab, it is a Custom Tab, and the redirect comes back through
  `SignInCallbackActivity`, which the app's scheme opens. `MainActivity` provides it as
  `LocalSignIn`.
- **Connect Gmail and Linear in the chat.** A call that waits for the person's account shows
  Stream's sign-in card on its step. Connect hands the card's page to `SignIn.authenticate`, and
  the account is connected when the page comes back with `status=connected`; the call then
  carries on at the router. Not now is `AgentSession.declineLogin`, which ends the call there.
  The composer floats over the conversation, which `contentPadding` keeps clear of it, so the
  latest message and the scroll-to-latest button are never under it.

The app is the user `juno-owner`, Alex, as the iOS app is, so both phones list the same
conversations. `Assistant.kt` holds the whole configuration: the router's and the backend's
URLs, the customer id, the Stream key, the agent's name and the user.

## What is not here yet

- Think harder, which iOS offers in the composer's options: Stream's Android composer has no
  options sheet.
