# Swift assistant

Juno, a personal assistant in the shape of the Grok and Meta AI apps: conversations kept in a
sidebar, a chat that shows its thinking and its steps, a voice mode you switch into and back
out of, tools that run on the phone for the things only the phone knows, and the person's own
Gmail, connected from a button at the top of the chat.

```
agent.yaml           what runs the agent: models, voice, search, greeting
instructions.md      what the agent is told
configure/main.go    the backend half that runs once: syncs this directory to the router
backend/main.go      the backend half that stays up: Stream tokens, and the person's Gmail
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
and the second adds `vision`, which looks at the picture `look_at_latest_photo` returns.

Leave the backend running. It signs the Stream token the phone is its user with, as the Swift
demo's does, and does what only a server may with the person's Gmail connection: start its
consent, read whether it is connected, delete it, and send a mail through it.

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
takes `stream-chat-swift` from its `develop` branch, because the voice glow (`AIVoiceGlowView`)
is not in a released `StreamChatAI` yet.
Close `SwiftDemo.xcodeproj` first: it opens the same local package, `sdks/swift/core`, and Xcode
gives a local package to one window at a time, so the other reports "Missing package product".

The router needs `XAI_API_KEY` for the chat's model and `GOOGLE_API_KEY`, `DEEPGRAM_API_KEY`
and `INWORLD_API_KEY` for a call. Web search needs one of `EXA_API_KEY`, `TAVILY_API_KEY` or
`PERPLEXITY_API_KEY`; without one the agent is never offered the tool, and only the router's
log says so.

## Gmail

The envelope at the top right of a conversation connects the person's Gmail. The backend starts
the router's consent for their own Gmail connection, with `return_url` set to the app's own
scheme, `io.getstream.visionagents.swiftassistant:/gmail` (the router takes one only in reverse
domain form, RFC 8252 section 7.1); the app opens the launch page in a web authentication sheet
with the handoff token in its fragment, the person answers Google, and the router stores the
grant and sends the sheet back to the app. The Google credential stays sealed in the router.

The agent binds `gmail` as each person's own (`connectors:` in `agent.yaml`). Google's Gmail MCP
server searches, reads, labels and drafts, but cannot send, so sending is `send_email`, a tool
on the phone that asks first and has the backend post to Gmail's REST API through the router's
proxy for the same connection. A conversation open before Gmail was connected picks it up on
its next Gmail call.

What it takes, once:

1. **A Google OAuth client.** In a Google Cloud project, enable `gmail.googleapis.com` and
   `gmailmcp.googleapis.com`, and enroll the project in the
   [Workspace Developer Preview](https://developers.google.com/workspace/preview), which the
   Gmail MCP server is part of. Set up the consent screen with the `gmail.readonly` and
   `gmail.compose` scopes, then create a client of type Web application, marked for an
   AI-powered agent, with `http://localhost:8080/v1/agents/connectors/oauth/callback` as an
   authorized redirect URI.
2. **The router's settings,** in the repo-root `.env`: `ROUTER_CONNECTORS_ENABLED=true`,
   `ROUTER_AUTH_KEK`, `ROUTER_PUBLIC_URL=http://localhost:8080`, a `DASHBOARD_BASE_URL` (the
   launch page will not render without one), and the client as `GOOGLE_MCP_CLIENT_ID` and
   `GOOGLE_MCP_CLIENT_SECRET`. Or give the app its own client instead of the environment's:
   `PUT /v1/agents/connectors/gmail/oauth-client`. Restart the router after either.
3. **`go run ./configure` again,** which binds `gmail` to the agent.

Without the client, the button says so: the router answers that it has no OAuth client for
`gmail` and names both ways to give it one.

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
- **Tools on the phone**: `get_local_context` (date, time, timezone), `get_location` (asks
  first), `list_events` (EventKit), `add_reminder` (asks first), `send_email` (asks first,
  showing who it goes to), `look_at_latest_photo` (asks first, then shows the agent the newest
  picture in the library). Calendar, reminders, location and photos each need the phone's
  permission too. The photo tool works in voice mode: the router hands a tool's pictures to the
  `vision` skill, which runs on a call, and a chat has nowhere to send them yet.

## What does not work yet

- **The camera on a call.** An agent sees video only by asking for frames with
  `get_video_frames`. A Swift tool can answer with pictures now (`ToolOutput`), but nothing on
  the phone takes frames from the call's camera to answer it with, so voice mode has no camera.
- **Sources.** A reply lists what it read in `support_message.sources` only when a shown
  tool's result is a citations object, and the built-in search answers in prose, so a searched
  reply never carries any. Neither SDK nor `StreamChatAI` has a view for them yet either.
- **Connecting Gmail from the chat.** When the agent reaches for Gmail before it is connected,
  the router writes a `connector_authorization` card on the reply, which `StreamChatAI` does not
  render, since it reads only `ai_` attachments. The instructions send the person to the
  Gmail button instead.
- **Memory across conversations** needs `MEM0_API_KEY` on the router and a `memory` on the
  session, which `SessionOptions` has no field for.
