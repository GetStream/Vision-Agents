# Flutter assistant

The Flutter half of [swift_assistant](../swift_assistant): Juno, the same agent, talked to by a
Flutter app on iOS and Android. The agent, its instructions, and the two Go programs that store
it and sign the phone's Stream token are all swift_assistant's. Only the app is here, as with
[android_assistant](../android_assistant).

```
lib/       the phone half: Flutter over stream_agents and Stream's AI components
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

Stream's hosted router is reached by the app's key and the person's token rather than a
customer id, so pass an empty one with its address, and the key and the backend of the app the
agent is stored in: `--dart-define=CUSTOMER_ID=` and
`--dart-define=ROUTER_URL=https://accelerate-us-east4.gcp.stream-io-api.com`. Store the agent
there with `configure` and run the backend with that app's `STREAM_API_KEY` and
`STREAM_API_SECRET` instead of a customer id.

The chat asks for `xai/grok-4.7`, a model that reads photos and streams its reasoning, so the
router needs `XAI_API_KEY`. A call needs `GOOGLE_API_KEY`, `DEEPGRAM_API_KEY` and
`INWORLD_API_KEY`. Web search needs one of `EXA_API_KEY`, `TAVILY_API_KEY` or
`PERPLEXITY_API_KEY`.

The AI components and the A2UI renderer are `stream_chat_flutter_ai` and
`stream_chat_flutter_ai_a2ui` 0.2.0 from pub.dev. The OpenUI renderer is the app's own
(`lib/openui_renderer.dart`) until Stream's is published.

## Gmail and Linear

The agent binds `gmail` and `linear` as each person's own (`connectors:` in swift_assistant's
`agent.yaml`). Ask "what are my most important Linear tickets?" before Linear is connected, and
the agent's Linear call waits for you to sign in. Its step in the reply asks you to connect
Linear, with Connect and Not now.

Connect opens the router's sign-in page with `flutter_web_auth_2`: in a web authentication sheet
on iOS, and on Android in an Auth Tab, or a Custom Tab where the browser has none, with the
handoff token in its fragment. You sign in at Linear, the router stores the grant, and it hands
the page back to the app's own scheme, `io.getstream.visionagents.flutterassistant:/connected`.
Every new conversation names that as its session's `return_url`, and the router takes one only in
reverse domain form (RFC 8252 section 7.1). The call that waited then carries on in the same
reply, on your own account, and the agent answers with the list. Later conversations use the
account with no sign-in, as do the iOS and Android apps, which are the same `juno-owner`.

Not now ends the call without it, and the agent says so. A sign-in refused at Linear, or one not
finished within ten minutes, ends it the same way, and stopping the reply ends it too.

The credentials stay sealed in the router, and the app never sees them. Every connector signs in
this way, Gmail included. The router writes the sign-in on the call's `ai_tool_call` step
(`awaiting_authorization`, with an `authorization`), `stream_chat_flutter_ai`'s
`AIToolAuthorizationView` shows it when the `AIToolApprover` has an `AIToolAuthorizer`
(`chat_screen.dart`), and Not now calls `AgentSession.declineLogin`. On Android the page comes
back from a Custom Tab through `flutter_web_auth_2`'s `CallbackActivity`, registered for the
app's scheme in `AndroidManifest.xml`; iOS needs nothing registered.

The sign-in page must be one the phone can open. The card offers only an `https` page, or an
`http` one on the phone's own localhost: the iOS simulator, or Android with `adb reverse`, with
`ROUTER_PUBLIC_URL=http://localhost:8080`. A phone reaching the Mac over the network needs the
router behind `https`, such as a tunnel, as its `ROUTER_PUBLIC_URL`, since the card will not open
a cleartext page elsewhere and Linear takes a plain `http` redirect only on localhost.

To try the sign-in again, delete the person's connection in the Volt dashboard, under Library ›
Connectors › Connections, or with `DELETE /v1/agents/connections/{id}`.

What it takes, once, besides the router's `ROUTER_CONNECTORS_ENABLED=true`, `ROUTER_AUTH_KEK`
and `ROUTER_PUBLIC_URL` in the repo-root `.env` (restart the router after a change), and
swift_assistant's `go run ./configure` again, which binds both:

- Linear needs nothing more, since its MCP server registers the router's client by itself
  (dynamic client registration).
- Gmail needs a Google OAuth client, set up as
  [swift_assistant's README](../swift_assistant/README.md#gmail-and-linear) says, with
  `<ROUTER_PUBLIC_URL>/v1/agents/connectors/oauth/callback` as an authorized redirect URI and
  given to the router as `GOOGLE_MCP_CLIENT_ID` and `GOOGLE_MCP_CLIENT_SECRET`. Without one, the
  agent says Gmail cannot be connected here.

## What the app shows

- **Conversations** are the router's, in the drawer the menu button opens: `sessions.query()`
  lists this user's, `sessions.search` finds them by title, and the router titles each one as it
  goes. They are under Today, Yesterday, This week and then their month, by when they started.
  Pull the list down to read it again; swipe one away to delete it. A new chat opens its session
  as its first message is typed, since opening one takes seconds, and one left unsent is deleted
  again on leaving it. One that ended carries on when it is next asked something, with the
  phone's tools.
- **Chat** is Stream's AI components (`stream_chat_flutter_ai`) over the session's channel.
  `AIConversationView` lays it out in turns: a question moves to the top and shows at once, and
  its reply writes itself out below it, with the model's reasoning as it thinks (reassembled
  from the live updates by `LiveReasoning`), each tool call as it runs, and the answer streaming
  in. Older messages load as you scroll up. Under a finished reply you can copy, share and rate
  it; sharing opens the platform's share sheet, and ratings stay on the phone, as the router
  takes none yet. The composer floats over the bottom of the conversation, which fades away
  under it, and its options sheet has **Think harder**, which asks with a higher
  `reasoning_effort`.
- The app is drawn in Stream's design system (`stream_core_flutter`, which
  `stream_chat_flutter_ai` is built on and re-exports): `StreamTheme` in light and dark,
  `StreamAppBar`, `StreamButton`, `StreamListTile` and `StreamTextInput` in the drawer, and
  `StreamIcons`. The composer's slots and the agent's messages are registered with
  `streamAIComponentBuilders`.
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
  what it is doing and captions a sentence at a time, paced to its voice, as swift_assistant's
  are. The microphone is asked for as voice mode opens. Hang up, the close button or Back take
  the agent off the call and go back to the chat.
- **The camera** button in voice mode shows Juno what the phone is pointing at, through the back
  lens, with a preview of it. Ask "what's this?" and the agent's `vision` skill asks the call's
  session to look (`get_video_frames`), which answers with pictures the camera takes then: unlike
  the Swift and Kotlin SDKs, which keep the last few frames, the Dart SDK takes them only when
  asked. Try it on a phone or the Android emulator, whose camera is a test scene; the iOS
  simulator has none.
- **Tools on the phone**, in `device_tools.dart`: `get_local_context` (date, time, timezone),
  `get_location` (asks first), `list_events` (the calendar), `add_reminder` (asks first) and
  `send_email` (asks first, showing who it goes to). A tool that asks first asks on its step in
  the reply. On iOS a reminder goes in Reminders; Android has no Reminders an app can add to, so
  there it is an event in the calendar new events go in, at the time it is due, with an alert
  then, and it takes a due time, as android_assistant's does. `send_email` has swift_assistant's
  backend send it through the router's proxy for the person's Gmail (`POST /gmail/send`). A seat
  selector card adds `reserve_table` while it is shown (see [Interactive UI](#interactive-ui)).

The app is the user `juno-owner`, Alex, as the iOS and Android apps are, so all three phones
list the same conversations. `assistant.dart` holds the whole configuration: the router's and
the backend's URLs, the customer id, the Stream key, the agent's name and the user.

## Interactive UI

The gear beside New chat in the drawer opens Settings, where **Replies** chooses how the agent
answers: in Markdown, or with interactive UI in A2UI or OpenUI. swift_assistant's backend keeps
the choice as what the agent is told (`GET` and `PUT /reply-format`), so the app has nothing to
switch: its `AIUIHost` has a renderer for each format, and a reply shows whichever UI it carries
in `custom.ai_ui`. A conversation keeps the instructions it started with, so the next new one is
the first to answer the new way, and earlier replies keep theirs. How the agent learns to write
each format, and the prompts generated for it, are in
[swift_assistant's README](../swift_assistant/README.md#interactive-ui).

With A2UI or OpenUI chosen, ask "Plan a weekend in Ohrid" and the reply is UI rather than text: a
map of the places, the person's own calendar for those days, place cards with their photos, and
follow-ups. The map, the place cards and the calendar are the app's own widgets
(`lib/trip_components.dart`), described in `assets/UIComponents.json`, a copy of
swift_assistant's, and both formats place them. They load what they show on the phone: the photos
from Wikipedia and the calendar from the device, which the agent never sees. Tapping a place, a
follow-up or a form's button asks the agent again: the person's message says what they did in
words and carries the action as `custom.ai_ui_action` (`responses.create(uiAction:)`).

Ask "Where should we eat in Ohrid tonight?" for the other kind of card: a `RestaurantCard` per
venue, an `AvailabilityGauge` of how full the first one is and a `SeatSelector` of its tables,
also the app's own. Tapping a free table asks to reserve it.

- **A card brings its own tool.** The agent cannot book a table until a seat selector is on the
  screen: once a card is written, it offers its tables to the conversation, which gives its
  session `reserve_table` (`SeatSelectorComponent.reserveTable`) with `AgentSession.addTools`.
  Tapping a free table has the agent call it, which asks first ("Reserve this table?", with the
  table it names) and books only a free table a card on the screen offers. Once you allow it
  that card turns into the booking: the table, the time and a confirmation code. Each card keeps
  its own booking, so a conversation that showed two venues can book at both. Once every card
  is booked the tool is taken back with `AgentSession.removeTool`. The router's
  `GET /v1/agents/sessions/{id}/tools`, with the backend's credentials, lists `reserve_table`
  as `session` while it is given.
- **The UI streams in over the socket.** The router sends a reply's UI on the session's socket
  as the model writes it (`ui_delta`, then `ui_completed`), and writes it on the reply's message
  in Chat later, on a slower tick. `AgentChat` reads each turn's UI into the SDK's `LiveUI`, which
  shows it on the reply until the message's own `ai_ui` is complete. Each part of the card
  animates in as it arrives, and the reply's text gives way to it rather than jumping.
- **A change to the last card is a patch.** Asked to change the card it just wrote, the agent
  writes only the statements that change, in an ```` ```openui-patch ```` block. The router
  merges them into that card before it writes the message, and the SDK does the same with what
  streams in, so the reply shows the whole card, changed.
- **The UI shows on a call too.** In voice mode the agent says a sentence or two and writes the
  same UI, which the router keeps out of what is spoken. Voice mode shows the newest of it under
  the glow, from the call's session: it comes up from below and grows as it is written, moving
  the glow, and tapping it asks the agent as in the chat. The animations follow the phone's
  reduce-motion setting.
- **Settings › Brand** chooses the accent and corners the UI wears: Stream Default, Ember
  Hospitality or Nordic Mobility (`brand_theme.dart`), kept between launches. Cards already on
  screen take it on at once: the A2UI and OpenUI renderers and the app's widgets read it from
  the widget tree. The agent writes the same UI whichever is chosen.

## What is not here yet

Each of these waits on Stream's Flutter AI components, the Dart SDK or the router, not on the
app.

- **Sources.** A reply lists what it read in `support_message.sources` only when a shown tool's
  result is a citations object, and the built-in search answers in prose, so a searched reply
  never carries any. Stream's Flutter AI components have no view for them yet either.
- **A tool's question asked of one device.** `AIClientIdentity` takes the install's id, but no
  SDK sends one to the router, so a question waiting for the person can be answered from any
  of their devices, on iOS and Android too.
- A sign-in is not asked for on a call: voice mode shows only the questions of the phone's
  tools, and a connector's sign-in waits on the reply's step in the chat. Connect it in writing
  first.
- Closing the sign-in page midway leaves its Connect button unable to open it again, because
  the router hands a sign-in to the first browser that opens it. Not now and asking again begin
  a fresh one.
- A sign-in goes back to the app that started the conversation, the `return_url` of its
  session: one started on iOS or Android goes back to that app, and one from before
  `return_url` to the dashboard, so the page never comes back here. Start a new chat.
- A card's UI makes room for itself in the chat a part at a time: Stream's components animate
  each part in, but not the reply growing around it, which the app cannot reach.
- The Think harder chip reads "Think harder" rather than iOS's shorter "Think", since
  `ChatOption` has no short title.
