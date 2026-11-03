# React assistant

The web half of [swift_assistant](../swift_assistant): Juno, the same agent, talked to from a
browser. The agent, its instructions, and the two Go programs that store it and sign the page's
Stream token are all swift_assistant's, as they are for the [Android app](../android_assistant).
Only the page is here.

```
package.json     React and Vite over @stream-io/calypso (sdks/js, as it is in this checkout)
                 and stream-chat-react's AI components
connected.html   where a connector's sign-in comes back to
src/             the page
tests/           its pure parts: captions, the glow's phases, photos, days
```

## Run it

Start the router in proxy mode, as in the [Swift demo](../swift_demo/README.md#run-it), so it
trusts the customer and user ids the page sends. Then store the agent and leave its backend
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

The backend listens on `:8098`, which the page reaches through Vite's dev server at `/backend`:
it answers no browser of another origin. Build the SDK once, then tell the page the Stream key,
in `.env.local` here, which git ignores:

```bash
(cd ../../../sdks/js && npm ci && npm run build)
echo "VITE_STREAM_API_KEY=<the router's STREAM_API_KEY>" > .env.local
npm install
npm run dev
```

and open <http://localhost:5173>. The page is served on `localhost` and nowhere else: a
connector's sign-in comes back only to the origin a conversation named, and the router takes an
`http` one only on a loopback host. `VITE_ROUTER_URL` points it at another router.

The AI components it uses (`AIConversation`, the reply's steps, the approval and sign-in cards,
suggestions, reply actions and the composer's chat options) are not in a released
`stream-chat-react` yet. Until 14.x has them, build the package from its
`feat/ai-components-juno` branch (`yarn install && yarn build && npm pack`) and put the tarball
here as `vendor/stream-chat-react.tgz`, which git ignores, then run
`npm install ./vendor/stream-chat-react.tgz`: a plain `npm install` keeps the copy it cached for
the old tarball. The page moves to the release that has them.

The chat asks for `xai/grok-4.7`, a model that streams its reasoning, so the router needs
`XAI_API_KEY`, and `GOOGLE_API_KEY`, `DEEPGRAM_API_KEY` and `INWORLD_API_KEY` for a call. Web
search needs one of `EXA_API_KEY`, `TAVILY_API_KEY` or `PERPLEXITY_API_KEY`.

```bash
npm test          # the pure parts
npm run build     # types, then the bundle
```

## What the page shows

- **Conversations** are the router's, in a sidebar beside the chat that folds to a rail, or on a
  window narrower than 768 px in the drawer the menu button opens: `sessions.query()` lists this
  user's, `sessions.search` finds them by title, and the router titles each one as it goes. They
  are under Today, Yesterday, This week and then their month, by when they started, with who is
  signed in and a light or dark switch at the bottom, which follows the system until it is used. The list is read again when it is shown, when a new chat gets its
  session and after each reply. A new chat has no session until its first
  message is written, and one opened but never sent is deleted again on leaving it. One that
  ended carries on when it is next asked something.
- **Chat** is Stream's AI components (`stream-chat-react/ai-components`) over the session's
  channel. `AIConversation` lays it out in turns: a question moves to the top and shows at once,
  and its reply writes itself out below it, with the model's reasoning as it thinks
  (reassembled from the live updates by the SDK's `LiveReasoning`), each tool call as it runs,
  and the answer streaming in. Older messages load as you scroll up. Under a finished reply you
  can copy, share and rate it; ratings stay in the page, as the router takes none yet. The
  composer floats over the bottom of the conversation, which fades away under it, and its
  options offer Think harder, which asks with a higher `reasoning_effort`.
- **Photos** from the composer's button go with the question, shrunk to 1280 pixels across and
  inline as JPEG data URIs (`photos.ts`), turned the way the camera held them. The question shows
  them at once; the router uploads them to the conversation's channel and writes them on the
  person's message, and the model sees them with that question's reply.
- **Voice mode** is the waveform beside the composer. It starts voice on the open conversation,
  so the agent joins its call with the chat so far, and everything said on the call lands back in
  the chat. The session's one socket stays open and runs the tools, and a tool that asks first
  asks on a card over the call. The agent is Stream's voice glow (`VoiceGlow`), moving with its
  voice in Stream's colors and with yours in red, with a word for what it is doing, captions two
  lines at a time, and a line under it when the agent fails on the call (speech recognition,
  model or voice), as on Android. The microphone is asked for as voice mode opens. Hang up, the
  close button or Escape take the agent off the call and go back to the chat.
- **The camera** button in voice mode shows Juno what the camera is pointing at, with a preview of
  it. Ask "what's this?" and the agent's `vision` skill asks for what the camera showed at that
  moment, which the session answers with the last few frames. The camera is asked for the first
  time it is turned on.
- **Tools in the page**, in `deviceTools.ts`, by the same names as on the phones:
  `get_local_context` (date, time, timezone, language), `get_location` (asks first; the
  browser's geolocation, and OpenStreetMap's Nominatim for the address, since a browser has no
  geocoder), `list_events`, `add_reminder` (asks first) and `send_email` (asks first, showing
  who it goes to). A browser has no calendar or reminders a page can read or add to, so Juno
  keeps its own in the browser (`reminders.ts`): `add_reminder` puts a reminder there, with a
  notification at its time, and `list_events` reads it. Notifications are asked for when the
  person allows the first reminder. A tool that asks first asks on its step in the reply.
  `send_email` has swift_assistant's backend send it through the router's proxy for the
  person's Gmail (`POST /gmail/send`).
- **Connect Gmail and Linear in the chat.** Every new conversation names
  `http://localhost:5173/connected.html` as its session's `return_url`. A call that waits for the
  person's account shows Stream's sign-in card on its step. Connect opens the router's sign-in
  page in a popup and hands it the token it asks for (`createPopupAuthorizer`); the provider
  sends the popup back to `connected.html`, which tells the page how it went and closes. The call
  then carries on at the router. Not now is `session.declineLogin`, which ends the call there.
  The router needs connectors on (`ROUTER_CONNECTORS_ENABLED=true` and a `ROUTER_PUBLIC_URL`).
  Linear registers its own OAuth client; Gmail needs a Google OAuth client, as
  [swift_assistant's README](../swift_assistant/README.md#gmail-and-linear) says.

The page is the user `juno-owner`, Alex, as the phones are, so all three list the same
conversations. `assistant.ts` holds the whole configuration: the router's and the backend's
URLs, the customer id, the Stream key, the agent's name and the user.

## What does not work yet

- **Sign-ins in a phone's conversation.** The router keeps each session's `return_url`, and a
  conversation started on a phone names the phone's own scheme, so its sign-ins go back to the
  phone and a page cannot finish one. Connect it in a conversation started here; the account then
  works everywhere. The same holds the other way round.
- **Pop-up blockers** can stop the sign-in window. The card says so, and Connect works once the
  page may open pop-ups.
- **Reminders alert only while a Juno tab is open**, and live in this browser alone.
- **Sources.** A reply lists what it read only when a shown tool's result is a citations object,
  and the built-in search answers in prose, so a searched reply never carries any.
- A sign-in is not asked for on a call, as on the phones. Connect an account in writing first.
- **Memory across conversations** is not used here, as on the phones.
