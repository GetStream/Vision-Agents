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
the end user's daily caps.

## Decisions

- **Customer spend, not user allowance.** A scheduled run is charged server-side
  (`routing.WithScheduledTurn`); usage recording and policy budgets are untouched.
- **Fire-time tools are the user's full tools.** The woken agent runs in the real
  conversation with its connectors, which is what makes "should I keep reminding
  you?" work.
- **Mid-call firing waits.** The reminder lands in writing after the call ends; if the
  call outlasts the delivery window, the occurrence fails and retries later.
- **Delivered means accepted.** A run is delivered once the session has taken the
  follow-up, never once the turn is over; the peer forwarder's 30 seconds cannot
  hold a turn.
- **Missed occurrences are recorded, not replayed.** A one-shot delivers up to 24h
  late, saying so; a recurrence skips occurrences over an hour old, each with a
  `missed` run row.

## Not done

Phase 2: `deliver_to` fallback beyond the conversation — proactive SMS, WhatsApp and
iMessage through the channels service (10DLC gate, WhatsApp's 24-hour window), a phone
call as the delivery, outcome-aware retries, and `quiet_hours`. Lifecycle webhooks
(`scheduled_task.created`, `scheduled_task.run.delivered`, ...) come with it. The device
SDKs (Swift, Kotlin, Dart) do not generate the scheduled-task operations yet: the seven
device-reachable ones are marked `x-client-accessible`, so each SDK's filter can take
them when a client surface is designed.
