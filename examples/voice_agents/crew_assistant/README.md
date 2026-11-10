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
backend/main.go      runs the backend, and the agent speaking first about an event
backend/crew.go      the restaurant: team, schedule, equipment, tickets, the leaders' inbox
backend/hosted.go    the restaurant's tools, hosted on the router
backend/http.go      what the phone calls: its token, events, the asked-first tools, the inbox
backend/roles.go     each person's crew role, kept in their Stream user's custom data
backend/handbook.go  the handbook's passages, read from the router's knowledge base
evals/main.go        runs the simulations, standing in for the phone's asked-first tools
```

## What it does

- **Tools on the backend.** The backend stands in for the restaurant's own systems, and hosts
  their tools on the router: `get_my_schedule`, `check_callout`, `find_coverage`,
  `notify_shift_leader`, `search_handbook`, `get_updates_since`, `get_equipment_guide`,
  `get_team`, `get_menu` and `save_shift_reflection`. The router sends it their calls with the
  user who asked, and each acts on that person's shifts and requests alone.
- **Tools on the phone.** The phone keeps what only it can do: its clock (`get_local_context`),
  and the three actions the person says yes to first, `request_shift_swap`,
  `create_support_ticket` and `book_check_in`, which call the backend once they have. The router
  asks for a yes only on a tool the phone declares, so those three stay there. The backend hosts a
  `get_local_context` with its own clock too, for a conversation with no phone; a phone's wins.
  Every tool shows as a step on the reply (`visible_tools`), and the app titles the steps.
- **Who is who.** Every request but the token needs `Authorization: Bearer <Stream token>`, signed
  with the Stream app's secret, and acts for the token's user. `POST /stream-token` signs one for
  whichever user the phone names, which stands in for the restaurant's sign-in; someone it hasn't
  met is enrolled as a new hire with a starter schedule. Each person's crew role (`leader`,
  `trainer` or `team_member`) is kept in their Stream user's custom data as `crew_role`.
- **Cal.com.** `agent.yaml` binds `calcom` as each person's own, with only tools that read. The
  first time the agent looks at their calendar, the call waits for them to connect it, shown on
  its step in the chat, as swift_assistant does with Linear. Cal.com registers the router as its
  OAuth client by itself, so it needs nothing besides the router's `ROUTER_CONNECTORS_ENABLED=true`,
  `ROUTER_AUTH_KEK` and `ROUTER_PUBLIC_URL`. Google Calendar would need an OAuth client of your
  own on the router first.
- **The handbook.** It is the agent's knowledge base on the router. `knowledge/` holds its
  documents: the Markdown files the sync pushes, and PDF, Word or Excel files `configure` uploads,
  which the router reads into Markdown itself, page by page. `search_handbook` searches the passages
  the router cut them into, by heading, and answers `{"status": "answered", "citations": [...]}`,
  so the passages it found show as the reply's sources. Add or replace a document and run
  `configure` again: the backend reads the passages again within a minute.
- **Speaking first.** `POST /events` with a `session_id`, a `kind` (`clock_in`,
  `clock_out`, `back_from_time_off` or `onboarding`) and the phone's `timezone` creates a
  scheduled task in the user's conversation and runs it at once, so the agent opens the
  conversation: the fit check at clock-in, the reflection at clock-out, what changed after
  time off, and a welcome for a new hire. The task is deleted two minutes later.
- **The leaders' inbox.** `GET /crew/leader` is what the app's team lead view shows: call-out
  notes, swap requests, drafted tickets and check-ins, each with who sent it, and the pulse of
  end-of-shift reflections, kept without names. It answers only a user whose `crew_role` is
  `leader`; the app's view signs in as one.

Everything lives in memory and starts from the same restaurant each time the backend starts.

## Run it

Run the router as in the [Swift demo](../swift_demo/README.md#run-it). Then:

```bash
cd examples/voice_agents/crew_assistant/configure
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
GOWORK=off go run . -dir ..

cd ../backend
STREAM_API_KEY=<the router's STREAM_API_KEY> \
STREAM_API_SECRET=<the router's STREAM_API_SECRET> \
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
GOWORK=off go run .
```

It listens on `:8097`, beside the Swift assistant's on `:8098`, and connects to the router to host
the tools. Run one backend per router: the router shares an agent's tool calls between every
process hosting them. `GOWORK=off go test ./...` in `backend/` checks the tokens, the leaders'
inbox, enrolment and that each person only acts on their own shifts.

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

A simulation's conversation has no user and no phone. The backend acts for the user its
`-simulation-user` names, and refuses a call that names no user without it. `evals` stands in for
the phone's three asked-first tools while it runs the simulations, calling the backend as that
user, with every call allowed; a phone declares its own, which win. What the simulations write
lands in that user's requests and the leaders' inbox.

```bash
cd examples/voice_agents/crew_assistant/backend
STREAM_API_KEY=<the router's STREAM_API_KEY> \
STREAM_API_SECRET=<the router's STREAM_API_SECRET> \
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
GOWORK=off go run . -simulation-user crew-jordan

cd ../evals
STREAM_ACCELERATION_URL=http://localhost:8080 \
STREAM_ACCELERATION_CUSTOMER_ID=examples \
GOWORK=off go run .
```

It prints each simulation's score and the judge's reason for every conversation that failed, and
exits non-zero unless all of them passed. `-only <name>` runs the matching simulations alone, and
`-transcripts` prints what was said in the failed ones. The three stand-ins mirror
`agent/crewTools.ts` in the app: change both together.
