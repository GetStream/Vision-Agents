# Crew assistant

Crew, the assistant a restaurant's team members talk to from their phone, before, during and
after a shift. It is forked from [swift_assistant](../swift_assistant) and talked to from the
React Native app `examples/ExpoCrewAssistant` in stream-chat-react-native.

```
agent.yaml           what runs the agent
knowledge/           the team handbook: Markdown, and documents such as PDFs
instructions.md      what the agent is told: the restaurant, its ground rules and each flow
a2ui/instructions.md how it writes checklists, quizzes and forms in A2UI
simulations/         the conversations each flow is judged on
configure/main.go    stores the agent on the router, with A2UI, its visible tools and skills
backend/main.go      Stream tokens, and the agent speaking first about an event
backend/crew.go      the restaurant: schedule, team, equipment, tickets, the leaders' inbox
backend/handbook.go  the handbook's passages, read from the router's knowledge base
evals/main.go        runs the simulations, with the phone's tools hosted against the backend
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
- **The handbook.** It is the agent's knowledge base on the router. `knowledge/` holds its
  documents: the Markdown files the sync pushes, and PDF, Word or Excel files `configure` uploads,
  which the router reads into Markdown itself, page by page. `/crew/handbook` searches the passages
  the router cut them into, by heading, and answers `{"status": "answered", "citations": [...]}`,
  so the passages it found show as the reply's sources. Add or replace a document and run
  `configure` again: the backend reads the passages again within a minute.
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

It listens on `:8097`, beside the Swift assistant's on `:8098`.

Uploading a PDF, Word or Excel file needs somewhere for it to wait until the router reads it:
`ROUTER_KNOWLEDGE_BUCKET_URL` in the repo-root `.env`, such as `file:///tmp` for a local router
(restart the router after setting it). Without it `configure` stops at the first such file, and
the Markdown documents still work.

The backend, `configure` and `evals` each point at the SDK next door with a `replace`, so
`GOWORK=off` runs them without adding them to `go.work`.

The agent runs on `xai/grok-4.7`, which reads photos and thinks before it answers, so replies
show reasoning steps. The router needs `XAI_API_KEY`, and `DEEPGRAM_API_KEY` and
`INWORLD_API_KEY` for voice mode.

## Evals

`simulations/flows.yaml` holds a conversation for each flow: the handbook (an answer, a gap it
mustn't fill, an allergen), calling out with and without cover, a fryer error and its ticket, the
fit check, the menu quiz, coming back from time off, the clock-out reflection and interview
practice. The router holds each three times, worded differently, with a model playing the team
member, and another model judges each conversation against the assertion, tool calls included.
`configure` syncs them.

A simulation has no phone, so `evals` hosts the phone's tools under the agent's name while it
runs them, each calling the backend the way the app does. A phone declares its own tools, which
win, so the app is unaffected. Give the evals a backend of their own, so their swaps and tickets
stay out of the team lead view:

```bash
cd examples/voice_agents/crew_assistant/backend
STREAM_API_SECRET=<the router's STREAM_API_SECRET> \
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
GOWORK=off go run . -addr :8096

cd ../evals
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
GOWORK=off go run . -backend http://localhost:8096
```

It prints each simulation's score and the judge's reason for every conversation that failed, and
exits non-zero unless all of them passed. `-only <name>` runs the matching simulations alone, and
`-transcripts` prints what was said in the failed ones. The hosted tools mirror
`agent/crewTools.ts` in the app: change both together.
