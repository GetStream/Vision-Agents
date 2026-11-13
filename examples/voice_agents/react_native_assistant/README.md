# React Native assistant

The React Native half of [swift_assistant](../swift_assistant): Juno, the same agent, talked to
from an Expo app on iOS and Android. The agent, its instructions, and the two Go programs that
store it and sign the phone's Stream token are all swift_assistant's. Only the app is here. The
other apps are [android_assistant](../android_assistant), [flutter_assistant](../flutter_assistant)
and, in a browser, [react_assistant](../react_assistant).

```
app/          the screens, with expo-router
agent/        the router: sessions, the conversation over AgentChat, the phone's tools, photos
contexts/     AppContext: the open conversation and its AIConversationController
screens/      the chat and the drawer of conversations
components/   voice mode, and the generated UI's own components
tests/        Jest tests of agent/
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

Copy `.env.example` to `.env.local`, which git ignores, and set `EXPO_PUBLIC_STREAM_API_KEY` to
the router's `STREAM_API_KEY`. The app reaches the router on `http://localhost:8080` and the
backend on `http://localhost:8098`. On Android, `adb reverse` hands the phone's localhost on to
the Mac's:

```bash
adb reverse tcp:8080 tcp:8080
adb reverse tcp:8098 tcp:8098
```

Then build and run the app. `expo run` generates the native `ios` and `android` folders, which
git ignores:

```bash
cd examples/voice_agents/react_native_assistant
npm install
npx expo run:ios       # or: npx expo run:android
```

`npm test` runs the tests and `npm run typecheck` checks the types.

To run it against a local checkout of the chat SDK, start Metro with `STREAM_CHAT_RN_DIR` set to
that checkout: `STREAM_CHAT_RN_DIR=~/Projects/stream-chat-react-native npx expo start`. Metro then
reads `stream-chat-expo` and `stream-chat-react-native-core` from its source, and SDK changes
reload in the app. Native changes to the SDK still need an install and a rebuild.

The chat asks for `xai/grok-4.7`, a model that streams its reasoning, so the router needs
`XAI_API_KEY`. The app needs `stream-chat-expo` 10.1.0 or newer, whose `ai-components` entry has
`AIConversation` and the rest of the AI components.

On iOS, if the app crashes at launch in `dyld` with a missing symbol, reinstall the pods without
Expo's precompiled modules: `cd ios && EXPO_USE_PRECOMPILED_MODULES=0 pod install`, then build
again.

## How it works

The app talks to the router through
[`@stream-io/stream-agents`](https://www.npmjs.com/package/@stream-io/stream-agents) (`agent/`).

- Each conversation is a router session, kept in an `agent` channel. The drawer lists the user's
  `agent` channels. `+` starts a new session, and one that is never asked anything is deleted
  when you leave it.
- The chat is `AIConversation` from `stream-chat-expo/ai-components`, over an
  `AIConversationController` that `contexts/AppContext.tsx` builds from the SDK's `AgentChat` and
  `agentChatTransport` (`@stream-io/stream-agents/chat`). `AgentChat` watches the channel, loads
  older pages, keeps each reply's full reasoning from its live updates, and follows what the agent
  is doing on the session's socket. The controller offers no edit or regenerate: the router can't
  rewind a conversation.
- Asking goes through the session, not the chat composer. The router writes the question and the
  streamed reply into the channel, the reply's steps as `ai_reasoning` and `ai_tool_call`
  attachments, and titles the conversation itself. Photos go as data URIs, and "Think harder" as
  a high reasoning effort. The phone's tool, `get_local_context` (`agent/deviceTools.ts`), is
  declared on the session, which runs it.
- The keyboard is handled by the AI components through `react-native-keyboard-controller`; the
  app only wraps itself in `KeyboardProvider`. The starters and the voice button sit in the
  composer's `header`, so they ride the keyboard.
