# Scheduled tasks

Asked for directly, against the same gap Pipecat, Retell, OpenAI Agents and LiveKit all
leave: every timer those frameworks ship is an in-session idle timeout, so "remind me
tomorrow" needs the developer to build the store, the cron, the retries and the delivery
(`unverified`: not re-checked against those frameworks' docs for this file).

How the code works, its terms and the tests that hold each rule:
[`acceleration/internal/scheduling/AGENTS.md`](../../acceleration/internal/scheduling/AGENTS.md).
How to turn it on, schedule one and debug one: the `scheduled-tasks` skill
(`.claude/skills/scheduled-tasks/SKILL.md`). Paths below are under `acceleration/`.

Example: Sam says "remind me to stretch every weekday at 09:00". The agent calls
`schedule_task` with `at: "09:00"`, `repeat: "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"`. The task
reads back as "Every weekday at 09:00 (Europe/Amsterdam)". At 09:00 Amsterdam time the
agent is woken in Sam's conversation and writes the reminder there.

## What exists

A user asks the agent to do something later, in conversation, and it happens in that same
conversation. Four tools (`schedule_task`, `list_scheduled_tasks`,
`update_scheduled_task`, `cancel_scheduled_task`) are offered when the agent config's
`scheduling.enabled` is on, the router has a store, the session is not incognito and has a
config, and the caller is a verified end user or a backend acting for one; guests and
anonymous callers are not offered them (`internal/session/schedule_tools.go:106`,
`verifiedCaller` in `internal/session/connector_tools.go:569`). A caller that declared a
tool under one of the four names keeps its own.
A schedule is a wall-clock time in the user's IANA zone, once or repeating by an RRULE
subset (`internal/scheduling/scheduling.go:436`: FREQ DAILY, WEEKLY, MONTHLY or YEARLY,
INTERVAL, BYDAY, BYMONTHDAY, COUNT up to 1000, and an inclusive `until` date), with DST
resolved explicitly — a 09:00 task stays at 09:00 across a clock change; a time in the
spring gap moves forward by the gap, and a fall-back time takes its first pass
(`internal/scheduling/resolve.go:5`). A one-shot is at most a year ahead
(`scheduling.go:249`). A repeat whose sentence `Describe` would misstate is refused with
`invalid_repeat`, and the refusal lists the shapes that are supported
(`internal/scheduling/describe.go:37`); a task stored before a shape was refused keeps
running.
The zone resolves, first match wins, from the tool call, the session, `users.timezone`,
then the config's `default_timezone`; REST create has no session step
(`internal/api/scheduled_tasks.go:219`). None known is a `timezone_unknown` refusal
rather than a guess (`scheduling.go:153`). A zone the user names in `schedule_task` becomes
theirs (`schedule_tools.go:235`). A session opened with a zone sets `users.timezone` when
the zone differs, except for an anonymous caller or an incognito session
(`internal/api/session_create.go:76`). Each user may hold 25 active tasks per agent config
(`scheduling.MaxActivePerUser`, `scheduling.go:72`), or the config's
`scheduling.max_tasks_per_user` (1 to 100, `internal/api/configs.go:2030`); the count is
per config (`internal/store/scheduled_tasks.go:194`). Deleting a config pauses its active
tasks (`configs.go:298`). Deleting a session deletes the tasks made in it or following up
in it (`internal/api/session_delete.go:47`). A data move (`router replicate`) carries the
tasks and their runs (`internal/store/datamove.go:99`).

Every router with a store and sessions runs the worker (`internal/scheduling.Scheduler`,
`cmd/router/main.go:1264`). Claims are one transaction under `FOR UPDATE SKIP LOCKED`
with a 1-minute lease and a `claim_id` fence, so each occurrence is claimed once across
the deployment (`internal/store/scheduled_tasks.go:455`); the run row is the prompt
snapshot, so a retry says the same thing. The worker renews the lease every 10 s while a
delivery runs (`internal/scheduling/scheduler.go:22`); a pause, edit or delete takes the
claim, and the delivery and its retries stop at the next renewal. A failed attempt is
retried after 30 s, 2 min and 10 min (`scheduler.go:62`). One occurrence holds a worker
slot for at most 20 minutes, retries included (`scheduler.go:34`); a router holds 16 slots
(`scheduler.go:25`). An occurrence whose worker died three times is failed with
`delivery_failed` (`internal/store/scheduled_tasks.go:59`). Five failed occurrences in a row set the task
to `failed` (`scheduler.go:57`). A stopping router claims nothing more, and its drain waits
for the deliveries in flight. The worker sleeps until the soonest a task may be claimed,
read off `scheduled_tasks_wake_idx` (migration `20261105130000_scheduled_tasks_wake.sql`;
the tables are `20261105120000_scheduled_tasks.sql`).
Delivery (`internal/api/scheduled_delivery.go`) follows up in the task's own
conversation: live on this node, followed up in place; live on another, handed over the
peer forwarder, where that node walks the task's channels and answers with the outcome
(`scheduled_delivery.go:536`); ended, reopened from its row and held until it answers. A
first occurrence with no session opens a text session on the task's config. A session on a
call is waited out, never interrupted. The follow-up carries the command id `st_<run id>`
(`scheduled_delivery.go:161`), and the conversation refuses a second copy of it
(`Session.FollowUp`, `internal/session/session.go:623`).

The same resource is REST: `/v1/agents/scheduled-tasks` (create for the user
`X-Stream-User-Id` names, list, get, PATCH, pause, resume counted from now, run by hand,
delete, runs), reached the way sessions are — the owner's device sees its backend-made
tasks, anybody else's answer 404. Create and run are server-side only; the other seven are
`x-client-accessible`. Create does not need `scheduling.enabled`: that flag only offers the
tools (`internal/api/scheduled_tasks.go:752`). A device may not change `instructions` (403). Moving a run
earlier than the current one is refused when it lands under 15 minutes from now
(`interval_too_short`, `scheduling.MinMove`, `scheduling.go:52`).
A task's `last_run_at` is when a run last finished, in UTC: the time a run settled
(`max(finished_at)`, `internal/store/scheduled_tasks.go:845`). A run's `scheduled_for` is when the
occurrence was due, in the task's zone (`internal/api/scheduled_tasks.go:712`). Example: a
one-shot due Mon 09:00, paused, and resumed Tue 10:00 has one `missed` run with
`scheduled_for` Mon 09:00 and `last_run_at` Tue 10:00.
Scheduled turns are customer spend: they never consume the end user's daily caps. That is
the follow-up turn and the replies that finish it; what the user types after it is theirs
again.

Delivery goes where the task says, in order (`deliver_to`, default the conversation
alone). The agent's reply is written once in its conversation, under the run's command
id; a text channel carries those same words out (`channels.Service.Notify`: the user's
identity on the channel names where they go, the agent's line carries them, the 10DLC
gate refuses who opted out, and WhatsApp is written only inside the day after the person
last wrote — outside it the next channel is tried, templates not being supported;
`internal/channels/service.go:202`). A text channel the user cannot be reached on is
refused before the turn is paid for (`channel_unreachable`, `scheduled_delivery.go:288`).
`phone` rings the user the way a campaign does (`phone.Service.Call`, a session opened
over the task's conversation) and the agent speaking is the delivery. The number rung is
the one the user texts this agent from (SMS, then WhatsApp); the number rung from is the
agent's SMS line, else its WhatsApp line (`scheduled_delivery.go:717`). `quiet_hours`
holds an occurrence that comes due in them until they end, on every channel: the worker
keeps the claim, the run stays running, and its age is counted from the hold's end. A
run made by hand is never held. The tools don't take `deliver_to` or `quiet_hours` yet;
they are REST-only.

## Decisions

- **Customer spend, not user allowance.** A scheduled run is charged server-side
  (`routing.WithScheduledTurn`). The agent keeps the mark with the turn, as it keeps the
  turn's sender, so the model requests that answer it carry it to the daily limit
  (`llmrouter.Session.charged`); usage recording and policy budgets are untouched.
- **Fire-time tools are the user's full tools.** The woken agent runs in the real
  conversation with its connectors, which is what makes "should I keep reminding
  you?" work.
- **Mid-call firing waits.** The reminder lands in writing after the call ends; if the
  call outlasts the delivery window (10 min, `scheduledWindow`), the attempt fails with
  `user_on_call` and is retried later.
- **Delivered means accepted.** A run is delivered once the session has taken the
  follow-up, never once the turn is over. A channel delivery answers later — once
  the words are sent — and the peer hand-off answers with where it landed.
- **Missed occurrences are recorded, not replayed.** A one-shot delivers up to 24h
  late, saying so; a recurrence skips occurrences over an hour old, each with a
  `missed` run row (`scheduler.go:51`). A repeating task's pause skips what it sleeps
  through, with no row.
- **A paused one-shot resumed after its time is closed, not run (AI-1602).** Resume over
  the API or the `update_scheduled_task` `resume` action writes one `missed` run with
  error `task_paused` and completes the task (`scheduling.SleptThrough`,
  `scheduler.go:129`). A second resume writes no second run: the occurrence's unique key
  keeps the first. The tool result says which time was missed, in the task's zone
  (`schedule_tools.go:435`).
- **A held run becomes `missed` when the user pauses or edits the task (S4).** The edit
  decides what happens next: a repeating task goes on to its next occurrence, and an
  active one-shot comes due again at the save, as it now reads
  (`SaveScheduledTask`, `internal/store/scheduled_tasks.go:270`). A one-shot a worker is delivering, not
  held, keeps its claim and run, so it is not delivered twice. Pausing through a config
  delete does not settle a held run yet (AI-1600).
- **A call more than 15 minutes late is `missed`, and an unanswered call is not retried
  (S5).** The call is not placed (`scheduledCallGrace`, `scheduled_delivery.go:65`).
  Nobody picking up within the 90 s ring is `not_answered`, final: the next channel in
  `deliver_to` is still tried, but the occurrence is not retried
  (`scheduled_delivery.go:821`). The call's session is taken on the run before it is
  placed, so no retry and no worker taking the run over rings the user twice
  (`already_called`). A vendor refusing the dial gives the call back and fails the attempt
  at once; a later attempt may ring. The router keeps no vendor call dispositions and has
  no answering-machine detection, so busy collapses into the dial's answer, and a mailbox
  that picks up hears the reminder and counts as delivered, the way a text in an inbox
  does.
- **Refuse repeat rules `Describe` can't say (S18).** What a task says it does is what
  fires (`describe.go:34`).
- **The task limit counts per config (S20).** A user's tasks on one agent do not use up
  another agent's limit (`internal/store/scheduled_tasks.go:424`).
- **Quiet hours hold, never fail.** A held occurrence keeps its claim and its run
  stays `running`; nothing counts against the task, and no channel is attempted —
  retries and fallbacks happen after the hold, because the whole delivery does.

## Not done

Phase 3: `require_ack` and `escalate_after`; lifecycle webhooks
(`scheduled_task.created`, `scheduled_task.deleted`, `scheduled_task.run.delivered`,
`scheduled_task.run.failed`) and a `task_due` dispatch frame; devices naming their own
zone. The tools don't take `deliver_to` or `quiet_hours` yet. The answered leg of a
phone delivery is proven by the campaign pattern it copies, not by an api-suite test:
the suite's telephony has no Stream app to place a call into, so the phone step is
covered there by its failure and fallback behaviour. The device SDKs (Swift, Kotlin,
Dart) do not generate the scheduled-task operations yet: the seven device-reachable
ones are marked `x-client-accessible`, so each SDK's filter can take them when a
client surface is designed.

Open tickets (Linear, all Backlog on 2026-10-21):

Product questions

- AI-1527 Scheduled tasks: should deleting a session delete its tasks?
- AI-1528 Scheduled tasks: where should SMS replies to a task land?
- AI-1529 Scheduled tasks: calling a WhatsApp-only number
- AI-1530 Scheduled tasks: ledger change for old command retries
- AI-1531 Scheduled tasks: retention for run history and finished tasks
- AI-1532 Scheduled tasks: a guest's tasks after a guest claim

Follow-ups

- AI-1555 Flaky scheduled-task tests: test worker looks for due tasks only once at start
- AI-1556 Thread hook retries on any error, writing one more cancelled message per try
- AI-1557 Every follow-up keeps a reply copy in memory for the conversation's lifetime
- AI-1558 after a restart an interrupted/cancelled follow-up is never re-run
- AI-1559 Read-back carries only m.Text; first capture also carries TaskSettled.Text
- AI-1560 Delegated work of a scheduled turn is still counted on the user's day
- AI-1561 Retry wait holds the scheduler drain
- AI-1562 Scope the ten deliver.count() assertions to their own tasks
- AI-1563 Kept instant is not stored; rebuild from the store loses it
- AI-1564 COUNT counts a start already passed
- AI-1565 Data race on Session.records between Manager.Create and the consume goroutine
- AI-1566 TestATaskTextedBySMSArrivesThere panics on the first run on a fresh DB
- AI-1567 Flaky TestAgentSuite/TestASecondVoiceAtTheSameMicrophoneIsNotTakenForTheCaller
- AI-1592 Map store and config errors to their own code, not channel_unreachable
- AI-1593 Record a hand-made phone run more than 15 min late as missed
- AI-1594 Decide whether an edited one-shot rings again after its worker was lost
- AI-1595 Make the occurrence budget cover a call placed late in it
- AI-1596 Keep the not_answered reason when another worker takes over a run
- AI-1597 Show the original due time on a re-dued run in its prompt
- AI-1598 Make REST task create atomic with the config read
- AI-1599 Decide the cut-over design for replicating active scheduled tasks
- AI-1600 Settle a held run when its config's tasks are paused
