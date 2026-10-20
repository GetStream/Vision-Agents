# Scheduled tasks

Asked for directly, against the same gap Pipecat, Retell, OpenAI Agents and LiveKit all
leave: every timer those frameworks ship is an in-session idle timeout, so "remind me
tomorrow" needs the developer to build the store, the cron, the retries and the delivery.

## What exists

A user asks the agent to do something later, in conversation, and it happens in that same
conversation. Four tools (`schedule_task`, `list_scheduled_tasks`,
`update_scheduled_task`, `cancel_scheduled_task`) are offered when the agent config's
`scheduling.enabled` is on, the caller is identified, and the session is not incognito.
A schedule is a wall-clock time in the user's IANA zone, once or repeating by an RRULE
subset (`internal/scheduling`: FREQ, INTERVAL, BYDAY, BYMONTHDAY, COUNT and an inclusive
`until` date), with DST resolved explicitly — a 09:00 task stays at 09:00 across a
clock change. The zone resolves from the tool call, the session, `users.timezone`, then
the config's `default_timezone`, and is a refusal rather than a guess when none is known.

Every router runs the worker (`internal/scheduling.Scheduler`). Claims are one
transaction under `FOR UPDATE SKIP LOCKED` with a lease and a `claim_id` fence, so each
occurrence is claimed once across the deployment; the run row is the prompt snapshot, so
a retry says the same thing. Delivery (`internal/api/scheduled_delivery.go`) follows up
in the task's own conversation: live on this node, followed up in place; live on another,
handed over the peer forwarder and answered on acceptance; ended, reopened from its row
and held until it answers. A first occurrence with no session opens one. A session on a
call is waited out, never interrupted. The follow-up carries a command id derived from
the run, and the conversation's durable ledger refuses a second copy after a reopen.

The same resource is REST: `/v1/agents/scheduled-tasks` (create for the user
`X-Stream-User-Id` names, list, get, PATCH, pause, resume counted from now, run by hand,
delete, runs), reached the way sessions are — the owner's device sees its backend-made
tasks, anybody else's answer 404. Scheduled turns are customer spend: they never consume
the end user's daily caps. That is the follow-up turn and the replies that finish it; what
the user types after it is theirs again.

Delivery goes where the task says, in order (`deliver_to`, default the conversation
alone). The agent's reply is written once in its conversation, under the run's command
id; a text channel carries those same words out (`channels.Service.Notify`: the user's
identity on the channel names where they go, the agent's line carries them, the 10DLC
gate refuses who opted out, and WhatsApp is written only inside the day after the person
last wrote — outside it the next channel is tried, templates not being supported).
`phone` rings the user the way a campaign does (`phone.Service.Call`, a session opened
over the task's conversation) and the agent speaking is the delivery. `quiet_hours`
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
  call outlasts the delivery window, the occurrence fails and retries later.
- **Delivered means accepted.** A run is delivered once the session has taken the
  follow-up, never once the turn is over. A channel delivery answers later — once
  the words are sent — and the peer hand-off answers with where it landed.
- **Missed occurrences are recorded, not replayed.** A one-shot delivers up to 24h
  late, saying so; a recurrence skips occurrences over an hour old, each with a
  `missed` run row.
- **A call's outcomes are the ones the router can see.** Nobody answering within the
  ring is no answer, and the next channel is tried (or the occurrence retried); a
  vendor refusing the dial fails the channel at once. The router keeps no vendor
  call dispositions and has no answering-machine detection, so busy collapses into
  the dial's answer, and a mailbox that picks up hears the reminder and counts as
  delivered, the way a text in an inbox does.
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
