# Swift assistant

Juno, a personal assistant in the shape of the Grok and Meta AI apps: conversations kept in a
sidebar, a chat that shows its thinking and its steps, a voice mode you switch into and back
out of, tools that run on the phone for the things only the phone knows, and the person's own
Gmail and Linear, connected in the chat the first time the agent needs them.

```
agent.yaml           what runs the agent: models, voice, search, greeting
instructions.md      what the agent is told
configure/main.go    the backend half that runs once: syncs this directory to the router
backend/main.go      the backend half that stays up: Stream tokens, sending from Gmail, and how
                     the agent answers
a2ui/                how the agent writes interactive UI in A2UI, generated from the app's components
openui/              the same in OpenUI
app/                 the phone half: SwiftUI over VisionAgentsCore and StreamChatAI
```

## Run it

Start the router in proxy mode, as in the [Swift demo](../swift_demo/README.md#run-it), so it
trusts the customer and user ids the phone sends.

Store the agent:

```bash
cd examples/voice_agents/swift_assistant
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
go run ./configure
```

`configure` syncs the directory with the Go SDK, then patches on `visible_tools` and `skills`,
which `agent.yaml` has no field for: the first shows a step for the phone's own tools on a reply,
and the second adds `vision`, which looks through the phone's camera on a call. It leaves the
agent answering in Markdown, until the app's settings choose [interactive UI](#interactive-ui).

Leave the backend running. It signs the Stream token the phone is its user with, as the Swift
demo's does, does what only a server may with the person's Gmail connection: send a mail
through it, and changes what the agent is told when the app's settings choose how it answers.

```bash
STREAM_API_SECRET=<the router's STREAM_API_SECRET> \
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
go run ./backend
```

It listens on `:8098`, beside the Swift demo's on `:8099`.

Put the `STREAM_API_KEY` the router runs with in `Assistant.streamAPIKey`, in
`app/SwiftAssistant/Assistant.swift`, then open `app/SwiftAssistant.xcodeproj` in Xcode 27 and
run it on a simulator. The app
takes `stream-chat-swift` from its `add/generative-ui` branch, because the generative UI it shows
is not in a released `StreamChatAI` yet, and GenUI and OpenUI, which its renderers of A2UI and
OpenUI draw with, from their `main` branches. The renderers themselves are the app's own, in
`app/SwiftAssistant/A2UI` and `app/SwiftAssistant/OpenUI`, for now.
Close `SwiftDemo.xcodeproj` first: it opens the same local package, the repository's root, and Xcode
gives a local package to one window at a time, so the other reports "Missing package product".

The router needs `XAI_API_KEY` for the chat's model and `GOOGLE_API_KEY`, `DEEPGRAM_API_KEY`
and `INWORLD_API_KEY` for a call. Web search needs one of `EXA_API_KEY`, `TAVILY_API_KEY` or
`PERPLEXITY_API_KEY`; without one the agent is never offered the tool, and only the router's
log says so.

## Gmail and Linear

The agent binds `gmail` and `linear` as each person's own (`connectors:` in `agent.yaml`).
Ask "what are my most important Linear tickets?" before Linear is connected, and the agent's
Linear call waits for you to sign in. Its step in the reply asks you to connect Linear, with
Connect and Not now.

Connect opens the router's sign-in page in a web authentication sheet, with the handoff token
in its fragment. You sign in at Linear, the router stores the grant, and it hands the sheet back
to the app's own scheme, `io.getstream.visionagents.swiftassistant:/connected`. Every new
conversation names that as its session's `return_url`, and the router takes one only in reverse
domain form (RFC 8252 section 7.1). The call that waited then carries on in the same reply, on
your own account, and the agent answers with the list. Later conversations use the account with
no sign-in.

Not now ends the call without it, and the agent says so. A sign-in refused at Linear, or one not
finished within ten minutes, ends it the same way, and stopping the reply ends it too.

The credentials stay sealed in the router, and the app never sees them. Every connector signs in
this way, Gmail included. The router writes the sign-in on the call's `ai_tool_call` step
(`awaiting_authorization`, with an `authorization`), StreamChatAI's `AIToolAuthorizationView`
shows it when the `AIToolApprover` has an `AIToolAuthorizer`, and Not now calls
`AgentSession.declineLogin`.

Google's Gmail MCP server searches, reads, labels and drafts, but cannot send, so sending is
`send_email`, a tool on the phone that asks first and has the backend post to Gmail's REST API
through the router's proxy for the same connection. Asked to send before Gmail is connected, it
tells the agent to call a Gmail tool first, which asks for the sign-in.

To try the sign-in again, delete the person's connection in the Volt dashboard, under Library ›
Connectors › Connections, or with `DELETE /v1/agents/connections/{id}`.

What it takes, once, besides the router's `ROUTER_CONNECTORS_ENABLED=true`, `ROUTER_AUTH_KEK`
and `ROUTER_PUBLIC_URL=http://localhost:8080` in the repo-root `.env` (restart the router after
a change), and `go run ./configure` again, which binds both:

- Linear needs nothing more, since its MCP server registers the router's client by itself
  (dynamic client registration).
- Gmail needs a Google OAuth client. In a Google Cloud project, enable
  `gmail.googleapis.com` and `gmailmcp.googleapis.com`, and enroll the project in the
  [Workspace Developer Preview](https://developers.google.com/workspace/preview), which the
  Gmail MCP server is part of. Set up the consent screen with the `gmail.readonly` and
  `gmail.compose` scopes, then create a client of type Web application, marked for an
  AI-powered agent, with `http://localhost:8080/v1/agents/connectors/oauth/callback` as an
  authorized redirect URI. Give it to the router as `GOOGLE_MCP_CLIENT_ID` and
  `GOOGLE_MCP_CLIENT_SECRET`, or as the app's own with
  `PUT /v1/agents/connectors/gmail/oauth-client`. Without one, the agent says Gmail cannot be
  connected here.

## What the app shows

- **Conversations** are the router's: `sessions.query()` lists this user's, `sessions.search`
  finds them by title, and the router titles each one as it goes. A new chat has no session
  until its first message. One that ended carries on when it is next asked something.
- **Chat** is Stream's AI components over the session's channel: the reply streaming in, its
  reasoning, each tool call and the approval a tool asks for. The composer's options sheet has
  Think harder, which asks with a higher `reasoning_effort`.
- **Photos** from the composer go with the question, shrunk and inline as data URIs. The
  router uploads them to the conversation's channel and writes them on the person's message,
  where the chat shows them; the model sees them with that question's reply.
- **Voice mode** starts voice on the open conversation, so the agent joins its call with the
  chat so far, and everything said on the call lands back in the chat. The chat's socket stays
  open and runs the tools; the call is followed by a second session with none.
- **The camera** button in voice mode shows Juno what the phone is pointing at, with a preview
  of it. Ask "what's this?" and the agent's `vision` skill asks for what the camera showed at
  that moment (`get_video_frames`), which the call's session answers with the last few frames.
  Try it on a phone: the simulator has no camera.
- **Tools on the phone**: `get_local_context` (date, time, timezone), `get_location` (asks
  first), `list_events` (EventKit), `add_reminder` (asks first), `send_email` (asks first,
  showing who it goes to). Calendar, reminders and location each need the phone's permission
  too.

## Interactive UI

The gear in the conversations list opens the settings, where Replies chooses how the agent
answers: in Markdown, or with interactive UI in A2UI or OpenUI. The backend keeps the choice as
what the agent is told (`GET` and `PUT /reply-format`), so the app has nothing to switch: its
`AIUIHost` has a renderer for each format, and a reply shows whichever UI it carries. A
conversation keeps the instructions it started with, so the next new one is the first to answer
the new way, and earlier replies keep theirs.

With A2UI or OpenUI chosen, ask "Plan a weekend in Ohrid" and the reply is UI rather than text: a
map of the places, the person's own calendar for those days, a tab per day with each place's photo
and summary, a form to change the pace and the interests, and follow-ups. The model writes it
after one sentence saying what it shows: as [A2UI](https://github.com/a2ui-project/a2ui) 1.0
messages in a ```` ```ui:a2ui@1.0 ```` block, or in [OpenUI](https://github.com/thesysdev/openui)
Lang in an ```` ```openui-lang ```` block. The router keeps that sentence as the message's text,
which push notifications and the Android app show, and moves the block into the message's
`custom.ai_ui` with its format, which StreamChatAI's `AIUIHost` shows in place of the text with
the app's `A2UIRenderer`, on [GenUI](https://github.com/martinmitrevski/GenUI), or its
`OpenUIRenderer`, on [OpenUI](https://github.com/martinmitrevski/OpenUI).

The map, the place cards and the calendar are the app's own views (`TripComponents.swift`),
described once in `app/SwiftAssistant/UIComponents.json`, and both formats place them. They
load what they show on the phone: the photos from Wikipedia and the calendar from EventKit, which
the agent never sees. The agent learns them with the format, from `a2ui/instructions.md` or
`openui/instructions.md`, which the backend adds to what it is told, and which are generated
from the same file by GenUI's `a2ui-prompt` and OpenUI's `openui-prompt`, with both cloned beside
this repository:

```bash
swift run --package-path ../../../../GenUI a2ui-prompt a2ui/prompt.json > a2ui/instructions.md
swift run --package-path ../../../../OpenUI openui-prompt openui/prompt.json > openui/instructions.md
```

Each `prompt.json` picks the components the agent may use and says when to write UI, and its
example has to render, or nothing is written. Choose the format again in the settings after
regenerating.

Tapping a place, a pin, a follow-up or the form's button asks the agent again: the person's
message says what they did in words, such as "Update the plan", and carries the action, with
the form's values, as `custom.ai_ui_action` (`session.responses.create(_:uiAction:)`). The
model is told it after their words.

## What does not work yet

- **Sources.** A reply lists what it read in `support_message.sources` only when a shown
  tool's result is a citations object, and the built-in search answers in prose, so a searched
  reply never carries any. Neither SDK nor `StreamChatAI` has a view for them yet either.
- A sign-in is not asked for on a call. A call's words are written by its transcript, so there
  is no reply to show it on, and an account not connected yet is not available by voice.
  Connect it in writing first.
- Closing the sign-in sheet midway leaves its Connect button unable to open it again, because
  the router hands a sign-in to the first browser that opens it. Not now and asking again
  begins a fresh one.
- A conversation from before this has no `return_url`, so its sign-ins go back to the dashboard
  and the sheet never comes back. Start a new chat.
- **Memory across conversations** needs `MEM0_API_KEY` on the router and a `memory` on the
  session, which `SessionOptions` has no field for.
