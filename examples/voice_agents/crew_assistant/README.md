# Crew assistant

Crew, the assistant a restaurant's team members talk to from their phone, before, during and
after a shift. It is forked from [swift_assistant](../swift_assistant) and talked to from the
React Native app `examples/ExpoCrewAssistant` in stream-chat-react-native.

```
agent.yaml           what runs the agent
instructions.md      what the agent is told: the restaurant, its tools and each flow
a2ui/instructions.md how it writes checklists, quizzes and forms in A2UI
configure/main.go    stores the agent on the router, with A2UI, its visible tools and skills
backend/main.go      Stream tokens, and the agent speaking first about an event
backend/crew.go      the restaurant: schedule, team, handbook, equipment, tickets, the leaders' inbox
```

## What it does

- **Tools on the phone.** The app declares the tools (`get_my_schedule`, `check_callout`,
  `find_coverage`, `request_shift_swap`, `notify_shift_leader`, `search_handbook`,
  `get_updates_since`, `get_equipment_guide`, `create_support_ticket`, `get_team`,
  `book_check_in`, `get_menu`, `save_shift_reflection`), and each reads or writes this
  backend under `/crew/`, which stands in for the restaurant's own systems. The swap, the
  ticket and the check-in ask the person first. Every one of them shows as a step on the reply
  (`visible_tools`).
- **Cal.com.** `agent.yaml` binds `calcom` as each person's own, with only tools that read. The
  first time the agent looks at their calendar, the call waits for them to connect it, shown on
  its step in the chat, as swift_assistant does with Linear. Cal.com registers the router as its
  OAuth client by itself, so it needs nothing besides the router's `ROUTER_CONNECTORS_ENABLED=true`,
  `ROUTER_AUTH_KEK` and `ROUTER_PUBLIC_URL`. Google Calendar would need an OAuth client of your
  own on the router first.
- **Sources.** `search_handbook` answers exactly `{"status": "answered", "citations": [...]}`,
  so the router adds the handbook passages to the reply's `support_message.sources`, which the
  app lists under the answer.
- **Speaking first.** `POST /events` with a `user_id`, a `session_id`, a `kind` (`clock_in`,
  `clock_out`, `back_from_time_off` or `onboarding`) and the phone's `timezone` creates a
  scheduled task in that session's conversation and runs it at once, so the agent opens the
  conversation: the fit check at clock-in, the reflection at clock-out, what changed after
  time off, and a welcome for a new hire. The task is deleted two minutes later.
- **The leaders' inbox.** `GET /crew/leader` is what the app's team lead view shows: call-out
  notes, swap requests, drafted tickets, check-ins, and the pulse of end-of-shift reflections,
  kept without names.

Everything lives in memory and starts from the same restaurant each time the backend starts.

## Run it

Run the router as in the [Swift demo](../swift_demo/README.md#run-it). Then:

```bash
cd examples/voice_agents/crew_assistant/configure
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
GOWORK=off go run . -dir ..

cd ../backend
STREAM_API_SECRET=<the router's STREAM_API_SECRET> \
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
GOWORK=off go run .
```

It listens on `:8097`, beside the Swift assistant's on `:8098`. Both modules point at the SDK
next door with a `replace`, so `GOWORK=off` runs them without adding them to `go.work`.

The agent runs on `xai/grok-4.7`, which reads photos and thinks before it answers, so replies
show reasoning steps. The router needs `XAI_API_KEY`, and `DEEPGRAM_API_KEY` and
`INWORLD_API_KEY` for voice mode.
