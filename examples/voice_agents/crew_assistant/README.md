# Crew assistant

Crew, the assistant a restaurant's team members talk to from their phone, before, during and
after a shift. It is forked from [swift_assistant](../swift_assistant) and talked to from the
React Native app `examples/ExpoCrewAssistant` in stream-chat-react-native.

```
agent.yaml           what runs the agent
knowledge/           the team handbook: Markdown, and documents such as PDFs
instructions.md      what the agent is told: the restaurant, its ground rules, each flow, and
                     how it writes checklists, quizzes and forms in A2UI
simulations/         the conversations each flow is judged on
configure/main.go    what the CLI's sync leaves out: PDFs and the like, visible tools, skills
backend/main.go      runs the backend, and the agent speaking first about an event
backend/crew.go      the restaurant: team, schedule, equipment, tickets, the leaders' inbox
backend/hosted.go    the restaurant's tools, hosted on the router
backend/http.go      what the phone calls: its token, events, the asked-first tools, the inbox
backend/roles.go     each person's crew role, kept in their Stream user's custom data
backend/handbook.go  the handbook's passages, read from the router's knowledge base
```

It runs on Stream's hosted Agents, and is synced and tested with the beta Stream CLI
(`getstream agents`).

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
  its step in the chat, as swift_assistant does with Linear. Cal.com registers Stream as its OAuth
  client at each sign-in, so it needs no setup.
- **The handbook.** It is the agent's knowledge base on the router. `knowledge/` holds its
  documents: the Markdown files the CLI syncs, and PDF, Word or Excel files `configure` uploads,
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

With the beta Stream CLI (`curl -fsSL https://getstream.io/cli/beta/install.sh | bash`), and the
key and secret of the Stream app the app runs in, from the repo root:

```bash
export STREAM_API_KEY=<your Stream app's key> STREAM_API_SECRET=<its secret>

getstream agents sync ./examples/voice_agents/crew_assistant
(cd examples/voice_agents/crew_assistant/configure && GOWORK=off go run . -dir ..)
(cd examples/voice_agents/crew_assistant/backend && GOWORK=off go run .)
```

`configure` runs after the sync, every time: it uploads the documents in `knowledge/` the CLI
skips, and sets the visible tools and the skills (`think`, `recall`, `explain`, `vision`), which
no file in the folder holds. A sync over them, which the CLI asks for once `configure` has run
(`--overwrite`), drops all three, so `configure` puts them back.

The backend listens on `:8097` and connects out to the router to host the tools, so it needs no
public address for them. The phone reaches it for its token, events, the asked-first tools and the
leaders' inbox, so a deployed app needs it somewhere public. Run one backend per agent: the router
shares an agent's tool calls between every process hosting them. `GOWORK=off go test ./...` in
`backend/` checks the tokens, the leaders' inbox, enrolment and that each person only acts on their
own shifts.

The team is made up, invites to `@example.com` addresses included. `CREW_SUPERVISOR_NAME` and
`CREW_SUPERVISOR_EMAIL` name a real direct supervisor instead, so a check-in booked with them
reaches their inbox through Cal.com.

The backend and `configure` reach Stream's hosted router. `STREAM_ACCELERATION_URL`, with
`STREAM_ACCELERATION_CUSTOMER_ID` for one without keys, points them at a router of your own, as
in the [Swift demo](../swift_demo/README.md#run-it); the CLI only reaches the hosted one. Both
point at the SDK next door with a `replace`, so `GOWORK=off` runs them without adding them to
`go.work`.

The agent runs on `xai/grok-4.7`, which reads photos and thinks before it answers, so replies
show reasoning steps.

## Evals

`simulations/flows.yaml` holds a conversation for each flow: the handbook (an answer, a gap it
mustn't fill, an allergen), calling out with and without cover, a fryer error and its ticket, the
fit check, the menu quiz, coming back from time off, the clock-out reflection and interview
practice. The router holds each three times, worded differently, with a model playing the team
member, and another model judges each conversation against the assertion, tool calls included.
The CLI syncs and runs them.

A simulation's conversation has no user and no phone. Started with `-simulation-user`, the backend
acts for that user when a call names none, and hosts the phone's three asked-first tools too, run
without asking; a phone declares its own, which win, so they only run where there is no phone.
What the simulations write lands in that user's requests and the leaders' inbox.

```bash
(cd examples/voice_agents/crew_assistant/backend && GOWORK=off go run . -simulation-user crew-jordan)
getstream agents simulate ./examples/voice_agents/crew_assistant --force
```

`--force` because `configure`'s settings make the CLI count the agent as changed since its sync.
Name a simulation, such as `"Clock-in fit check"`, to run it alone. The CLI prints each transcript
and the judge's ruling, and fails on any failed conversation. The three asked-first tools mirror
`agent/crewTools.ts` in the app: change both together.
