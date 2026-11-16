# Crew assistant (Expo)

An AI assistant for a restaurant's team members, built on the AI components and the Vision-Agents
router. A team member opens it before, during or after a shift. The home screen lists what it
helps with, and each flow starts a conversation with the agent: `AIConversation` for the chat,
A2UI for checklists, quizzes and forms, and approval cards for what the agent does in your name.

The backend is [crew_assistant](..), the folder above this one, forked from
[swift_assistant](../../swift_assistant). It stands in for the restaurant's schedule, handbook, equipment and leaders'
inbox, hosts their tools on the router, signs each person's Stream token, and makes the agent
speak first when something happens, such as a clock-in. Everything the app asks it carries the
user's Stream token, and the backend acts for that user alone.

## What it shows

| Flow                   | How it starts                 | What to look at                                                                                                                                                     |
| ---------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Clock in**           | the agent speaks first        | A fit-check checklist and the focus for the shift, as interactive UI. Tick the boxes and tap "I'm ready".                                                           |
| **Call out sick**      | a question                    | Schedule and call-out policy as tool steps. With a cold it offers cover, the swap asks you first, and the shift leader gets a note. With a fever it says stay home. |
| **Ask the handbook**   | a question                    | The answer comes from `search_handbook`, and the passages show as **Sources** under the reply.                                                                      |
| **Equipment problem**  | a question, a photo or a scan | The quick fix for the asset and code. "Still broken" drafts a repair ticket, which asks you first. Send a photo with "+" to have it look first.                     |
| **Back from time off** | the agent speaks first        | What you missed since your time off started, grouped.                                                                                                               |
| **I'm new here**       | the agent speaks first        | Who's who, a menu quiz as interactive UI, and booking a check-in with the trainer.                                                                                  |
| **Interview practice** | a question                    | A mock shift leader interview, one question at a time, with feedback. Try it in voice mode (the button beside the composer).                                        |
| **Clock out**          | the agent speaks first        | An end-of-shift reflection form. What it saves shows in the team lead view, without your name.                                                                      |

**Team lead** (top right) is what the shift leaders and the owner see: call-out notes, swap
requests, drafted tickets, booked check-ins and the team's pulse from reflections. It signs in as
a leader (`EXPO_PUBLIC_AGENT_LEADER_ID`, Maya by default), since the backend answers the leaders'
inbox only to a user whose Stream user has `crew_role: "leader"`. It refreshes every few seconds,
so keep it open next to a flow.

**Scanning a machine.** A QR code on a machine holds a link such as
`crewassistant://equipment?asset=FRY-02&code=E-41`. Opening it starts a conversation about that
machine. On an emulator, open it with:

```bash
adb shell am start -a android.intent.action.VIEW -d "crewassistant://equipment?asset=FRY-02\&code=E-41"
```

Known assets and codes: `FRY-02` (`E-41`, `E-07`), `POS-1` (`E-NET`), `ICE-1` (`BIN`).

## Run it

The agent runs on Stream's hosted Agents.

1. Sync the agent and start its backend, as [crew_assistant's README](../README.md) says: with the
   beta Stream CLI (`curl -fsSL https://getstream.io/cli/beta/install.sh | bash`) and your Stream
   app's key and secret, from the Vision-Agents root:

   ```bash
   export STREAM_API_KEY=<your Stream app's key> STREAM_API_SECRET=<its secret>
   getstream agents sync ./examples/voice_agents/crew_assistant
   (cd examples/voice_agents/crew_assistant/configure && GOWORK=off go run . -dir ..)
   (cd examples/voice_agents/crew_assistant/backend && GOWORK=off go run .)
   ```

2. Copy `.env.example` to `.env.local` and set `EXPO_PUBLIC_STREAM_API_KEY` to the same Stream app.

3. Build and run the app:

   ```bash
   cd examples/voice_agents/crew_assistant/app
   npm install
   npx expo run:android   # or: npx expo run:ios
   ```

   `expo run` generates the native `ios` and `android` folders, which git ignores. `npm test`
   runs the tests and `npm run typecheck` checks the types. The app needs `stream-chat-expo`
   10.1.0 or newer, whose `ai-components` entry has the AI components.

   On an Android emulator, `adb reverse tcp:8097 tcp:8097` lets the app reach the backend on
   `localhost`. To use a router of your own instead of the hosted one, set
   `EXPO_PUBLIC_AGENT_ROUTER_URL`, as `.env.example` shows.

   To run it against a local checkout of the chat SDK, start Metro with `STREAM_CHAT_RN_DIR` set
   to that checkout: `STREAM_CHAT_RN_DIR=~/Projects/stream-chat-react-native npx expo start`.
   Metro then reads `stream-chat-expo` and `stream-chat-react-native-core` from its source, and
   SDK changes reload in the app. Native changes to the SDK still need an install and a rebuild.

## How it's built

- `contexts/AppContext.tsx` opens conversations: new ones, ones from the drawer, ones that start
  with a question (`startConversationWith`), and ones the agent opens about an event
  (`startConversationAbout`), which asks the backend's `POST /events`.
- `agent/crewTools.ts` declares the phone's tools: its clock (`get_local_context`), and
  `request_shift_swap`, `create_support_ticket` and `book_check_in`, which ask first with the AI
  components' approval card, then call the backend. The router asks for a yes only on a tool the
  phone declares, so those stay here; the restaurant's other tools run on the backend.
  `DISPLAY_TITLES` titles every tool's step, the backend's too (`agent/conversation.ts`).
- `agent/agents.ts` signs in with the backend and asks it with the user's Stream token
  (`crewBackend`).
- The handbook is the agent's knowledge base on the router, filled from `crew_assistant/knowledge/`:
  Markdown files and documents such as PDFs. `search_handbook` answers with the passages that
  match as `{"status": "answered", "citations": [...]}`, which the router adds to the reply's
  `support_message.sources`. `screens/ChatContent.tsx` shows them under the reply.
- `app/leader.tsx` is the team lead view, and `app/equipment.tsx` the scanned-code link.

**Your calendar.** Booking a check-in ("Book a check-in with Priya on Thursday afternoon when I'm
free") first reads your own Cal.com. The first time, the reply asks you to **Connect Cal.com**, the
same sign-in the AI components show for any connector: Connect opens Cal.com's sign-in, and the
booking carries on once you're connected. Not now books the time you asked for without it.

**Voice memos.** The red mic beside the composer records a memo with the chat SDK's own recorder
(`AudioRecorderManager` and `AudioRecordingInProgress`). Send uploads it to the conversation's
channel with `channel.uploadChannelFile`, then asks the router with `audio: { url }`: the router transcribes
it, answers it and shows it on your message as a voice recording, which plays with the chat SDK's
`AudioAttachment`. At clock-out, a memo is your reflection. The router transcribes recordings with
Deepgram's batch model only, so it needs a `DEEPGRAM_API_KEY` with credit.

Not here yet: approvals answered by someone other than the person chatting.
