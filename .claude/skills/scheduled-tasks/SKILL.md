---
name: scheduled-tasks
description: Walks a developer through scheduled tasks end to end - what they do, turning them on for an agent config, scheduling one locally or on staging through the agent's tool or the REST API, watching it run, and what to check when nothing arrived. Read before you schedule a task, test the feature, or debug a reminder that never came.
---

# Scheduled tasks

A scheduled task is something the agent agreed to do for one user later: once, or on a
repeat, at a wall-clock time in that user's zone. When it comes due, the router wakes the
agent in the user's own conversation with the task's instructions. The agent's answer is
the delivery.

How the code works, its terms and the tests behind each rule:
`acceleration/internal/scheduling/AGENTS.md`. What is decided and what is not done:
`.factory/features/scheduled-tasks.md`. Paths below are under `acceleration/`.

## One example, start to end

```
 Alice, 14:02 (Europe/Amsterdam): "remind me to call the dentist in 20 minutes"
   agent -> schedule_task {title: "Dentist", instructions: "Remind Alice to call the dentist.", in: "PT20M"}
   tool  -> {"id": "...", "schedule": "Once, Wed 21 Oct 2026 at 14:22 (Europe/Amsterdam)", "next": "..., in 20 minutes"}
 14:22  worker claims the occurrence -> run "running"
        agent is sent <scheduled_task ...>Remind Alice to call the dentist.</scheduled_task>
        agent writes "Time to call the dentist!" in Alice's conversation
        run "delivered", delivered_to "conversation"; task "completed"
```

The tool and its fields: `internal/session/schedule_tools.go:36`. The prompt the agent is
sent: `scheduledPrompt`, `internal/api/scheduled_delivery.go:456`.

## Turn it on

Scheduling is a block on the agent config (`AgentScheduling`, `internal/api/configs.go:2103`):

| Field | What it does | Default |
| --- | --- | --- |
| `scheduling.enabled` | Offers the model the four tools: `schedule_task`, `list_scheduled_tasks`, `update_scheduled_task`, `cancel_scheduled_task` | off |
| `scheduling.max_tasks_per_user` | Active tasks one user may hold on this config, 1 to 100 | 25 (`scheduling.MaxActivePerUser`) |
| `scheduling.default_timezone` | The IANA zone a task takes when neither the call, the session nor the user names one. An unknown zone is a 400 `invalid_timezone` | none: a task with no zone is refused with `timezone_unknown` |

The tools also need (`Manager.schedulingRunner`, `internal/session/schedule_tools.go:106`):

- a router with a database;
- a session that is not incognito, opened on a config;
- a caller that is a verified user or a backend acting for one (`X-Stream-User-Id`), or the
  person who wrote in an SMS (Telnyx) or WhatsApp thread, and only when that person writes,
  not someone else in the thread (the task is theirs and keeps the thread; a Linq chat and a
  Slack thread get no tools). A guest or an anonymous caller gets no tools.

A task made in an SMS or WhatsApp thread comes back in that thread: the router opens the
thread's conversation, the agent answers the task there, and the reply is texted to the
person from the line. Its run says `delivered_to` `sms` or `whatsapp` once the line took the
reply; a reply the line refused is `delivery_failed`, and the retry sends the same reply,
files included, without a second turn; a thread removed while the turn ran is `thread_gone`,
final (`internal/api/scheduled_thread_delivery.go`). Such a task's
`deliver_to` reads `sms` or `whatsapp` and cannot be changed.

The REST create does not need `scheduling.enabled` (`internal/api/scheduled_tasks.go:752`).

A follow-up needs a kept conversation. A router without Stream Chat credentials keeps none
and logs `not keeping the conversation in Stream Chat` (`internal/session/manager.go:340`);
its tasks cannot be delivered into a conversation.

## Schedule one locally

1. Run the router per the repo-root `AGENTS.md` (Local dev) with `ROUTER_AUTH_MODE=proxy`
   (the Volt override sets it, `.claude/skills/dashboard/SKILL.md`). The commands below name
   the app with `X-Customer-Id`, which only `proxy` mode reads; in `api_key` mode send the
   credentials of "Schedule one on staging" instead. In `noauth` mode the router reads no `X-Stream-User-Id`
   (`internal/auth/auth.go:89`), so a create is refused ("name the user the task is for
   with X-Stream-User-Id") and no session gets the tools.
2. Make a config with scheduling on. In `proxy` mode without `proxy_declares_kind`, a request
   naming its app with `X-Customer-Id` and no `Stream-Auth-Type` is a backend
   (`internal/auth/auth.go:291`):

   ```bash
   R=http://localhost:8080
   H=(-H 'X-Customer-Id: examples' -H 'Content-Type: application/json')
   curl -s "$R/v1/agents/configs" "${H[@]}" -d '{
     "name": "reminders",
     "instructions": "You are a helpful assistant.",
     "scheduling": {"enabled": true, "default_timezone": "Europe/Amsterdam"}
   }'
   ```

   A config with that name already exists on a second run (409 `name_taken`). Use another
   name, or give the existing config the same block with `PATCH /v1/agents/configs/{id}`.
3. Schedule through the agent: open a session for a user, then write to it.

   ```bash
   U=(-H 'X-Stream-User-Id: alice')
   SID=$(curl -s "$R/v1/agents/sessions" "${H[@]}" "${U[@]}" \
     -d '{"agent": "reminders", "timezone": "Europe/Amsterdam"}' | jq -r .id)
   curl -s "$R/v1/agents/sessions/$SID/respond" "${H[@]}" "${U[@]}" \
     -d '{"text": "remind me to stretch in 20 minutes", "request_id": "remind-1"}'
   curl -s "$R/v1/agents/sessions/$SID/commands/remind-1" "${H[@]}" "${U[@]}"   # state: thinking, then completed
   ```

   A text session with a named user needs a durable command, so `respond` takes a
   `request_id` (without it: 400 `personal conversations require a command ID`,
   `internal/session/session.go:507`). `GET .../commands/{request_id}`
   (`internal/api/sessions.go:1147`) answers the command's `state` (`thinking`, then
   `completed`), not the reply text: read that in the conversation on the dashboard. The
   model has made the task when the list in step 4 shows it.

   Or through REST (`createScheduledTask`, `internal/api/scheduled_tasks.go:742`). Name the
   agent with `agent`, `config_id` or `session_id`:

   ```bash
   TASK=$(curl -s "$R/v1/agents/scheduled-tasks" "${H[@]}" "${U[@]}" -d '{
     "agent": "reminders", "title": "Stretch",
     "instructions": "Remind Alice to stretch and ask how her back is.",
     "schedule": {"at": "09:00", "repeat": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"}
   }' | jq -r .id)
   ```

   `TASK` is the new task's id. A task the tool made: take its id from the list in step 4
   (with one task, `jq -r '.items[0].id'` gives it).

   Without `session_id`, the first occurrence opens a session for the user.
4. Watch it. Fire it now without moving the schedule, then read its runs:

   ```bash
   curl -s "$R/v1/agents/scheduled-tasks?state=active" "${H[@]}" "${U[@]}" | jq '.items[] | {id, title, next_run_at, last_run_at}'
   curl -s -X POST "$R/v1/agents/scheduled-tasks/$TASK/run" "${H[@]}" "${U[@]}"   # 202, the run
   curl -s "$R/v1/agents/scheduled-tasks/$TASK/runs" "${H[@]}" "${U[@]}" | jq '.items[] | {scheduled_for, trigger, state, delivered_to, error}'
   docker compose logs -f router | grep -i 'scheduled task' | grep -v 'Failed to forward'
   ```

   The `grep -v` drops the job queue's `Failed to forward scheduled tasks` noise. A delivered
   run logs nothing; its run row says `delivered`. Open the session on the
   dashboard to read the agent's message.

## Schedule one on staging

Staging is `https://accelerate.gcp.stream-io-api.com`, behind Stream's proxy. The proxy
wants `api_key` and `stream-auth-type: jwt` for any token, and a server token is HS256
`{"server": true}` signed with the app's secret (`sdks/js/src/backend.ts:318`,
`signToken` at `:486`). Never print the secret or the token.

```bash
set -a; . ./.env; set +a          # STREAM_API_KEY, STREAM_API_SECRET; prints nothing
# node 22 reads the secret from its environment, so it never appears in a command line
TOKEN=$(node -e '
const c = require("node:crypto");
const b = (v) => Buffer.from(JSON.stringify(v)).toString("base64url");
const now = Math.floor(Date.now() / 1000);
const input = b({ alg: "HS256", typ: "JWT" }) + "." + b({ server: true, iat: now, exp: now + 3600 });
process.stdout.write(input + "." + c.createHmac("sha256", process.env.STREAM_API_SECRET).update(input).digest("base64url"));
')
R=https://accelerate.gcp.stream-io-api.com
A=(-H "Authorization: Bearer $TOKEN" -H 'stream-auth-type: jwt' -H 'Content-Type: application/json' -H 'X-Stream-User-Id: alice')
curl -s "$R/v1/agents/scheduled-tasks?api_key=$STREAM_API_KEY" "${A[@]}" | jq '.items | length'
```

Every call from "Schedule one locally" works with `"${A[@]}"` in place of
`"${H[@]}" "${U[@]}"` and `api_key=$STREAM_API_KEY` on the URL: after `?` when the URL has no
query yet, after `&` when it has one (`...?state=active&api_key=$STREAM_API_KEY`). Staging logs: the `deploy`
skill.

## Nothing arrived?

Read the task, then its runs, then the logs, in that order.

| Check | How | What it means |
| --- | --- | --- |
| Task `state` | `GET /v1/agents/scheduled-tasks/{id}` | `paused`: a pause, or its config was deleted (`internal/api/configs.go:308`). `failed`: five failed occurrences in a row (`maxConsecutiveFailures`), or a last occurrence that failed; resume clears it. A failed task with no occurrence left, or a paused one that did not sleep through its time, answers resume with 400 `schedule_in_past`: PATCH a new schedule first. A paused one-shot whose time passed while paused is completed with a `missed` run (`task_paused`). `completed`: a one-shot ran, or the repeat has nothing left |
| `next_run_at`, `upcoming` | same | In the task's zone. Wrong hour: the zone (below) |
| `schedule.timezone` | same | Resolved at create from the call, the session, `users.timezone`, then `default_timezone`. A session opened with a zone moves `users.timezone` (`internal/api/session_create.go:76`) |
| `quiet_hours` | same | An occurrence due inside them is held to their end: its run stays `running` with nothing sent. A run by hand is never held |
| The runs | `GET /v1/agents/scheduled-tasks/{id}/runs` | No run at all: the occurrence was never claimed (worker logs). `running` long after `scheduled_for`: held, or a worker is on it |
| Run `state` `missed` | same | Too late: a one-shot over 24 h, a repeat over 1 h (`scheduler.go:51`), a call over 15 min. Or `error` `task_paused`: a one-shot resumed after its time |
| Run `error` | same | See the reason table below |
| Limit | create answers 409 `too_many_tasks` | The user holds `max_tasks_per_user` active tasks on this config |
| The tools | the model never calls `schedule_task` | `scheduling.enabled` off, incognito, a guest or anonymous caller, or a `noauth` router |
| Session deleted | task 404 | Deleting a session deletes the tasks made in it or following up in it (`internal/api/session_delete.go:47`) |

Run `error` codes (`internal/scheduling/scheduler.go:105`):

| Code | Meaning | Retried |
| --- | --- | --- |
| `delivery_failed` | Any other failure, or the occurrence was claimed three times (`maxScheduledTaskRunAttempts`: the first claim, the take-up after a hold, one after a lost worker) | yes, unless it was claimed three times or a call was already placed (`scheduled_delivery.go:783`, `:857`) |
| `timed_out` | The occurrence used its 20 min (`occurrenceBudget`) | no |
| `schedule_unreadable` | The stored schedule no longer parses; the task is paused | no |
| `user_on_call` | The user stayed on a call through the 10 min window | yes |
| `channel_unreachable` | SMS, WhatsApp or iMessage would refuse: never wrote on it, opted out, outside WhatsApp's 24 h window, no line. On a thread task: opted out (before the turn or by the time the reply is sent), or the sandbox gate holds it back | yes; not for a thread task's opted-out person |
| `outside_window` | A WhatsApp thread task whose person last wrote over 24 h ago, with no template to send after the window | no |
| `thread_gone` | A thread task's thread no longer exists, or is not its user's thread of its customer | no |
| `connection_removed` | A thread task's line (its connection) was removed | no |
| `no_reply` | The agent said nothing and made no file to carry | yes |
| `reply_failed` | The reply failed, or was cancelled or interrupted and, when run once more under `st_<run>_r1`, was cut off again (a thread task's first cut-off also gives it, and the retry runs it again) | yes |
| `no_phone_number` | The user never texted this agent, so there is no number to ring | yes |
| `no_agent_number` | The config has no SMS or WhatsApp line to ring from | yes |
| `not_answered` | Nobody picked up within the 90 s ring | no |
| `already_called` | A call was placed for this occurrence already | no |
| `task_paused` | On a `missed` run: a one-shot whose time came while it was paused | — |

A failure is retried after 30 s, 2 min and 10 min (`DeliveryRetries`) only when the failure
is not final; with several channels, it is final only when every channel's failure is.

Worker log lines, as the code writes them (`grep -i 'scheduled task'` finds all but the
last; it also finds the job queue's `Failed to forward scheduled tasks`, which is noise):

| Line | Level | Where |
| --- | --- | --- |
| `could not claim the scheduled tasks due` | error | `internal/scheduling/scheduler.go:271` |
| `could not find when the next scheduled task is due` | error | `scheduler.go:295` |
| `a scheduled task's stored schedule does not read` | error | `scheduler.go:312` |
| `a scheduled task's occurrence failed: it was started this many times and its worker was lost each time` | error | `scheduler.go:322` |
| `a scheduled task was too late to deliver` | info | `scheduler.go:351` |
| `a scheduled task changed under its run; its delivery was stopped` | info | `scheduler.go:364` |
| `a scheduled task failed to deliver` | warn | `scheduler.go:372` |
| `could not hold a scheduled task` | error | `scheduler.go:469` |
| `a scheduled task changed under its run; the hold was left alone` | info | `scheduler.go:471` |
| `a scheduled task is held` (with `until`) | info | `scheduler.go:473` |
| `could not renew a scheduled task's lease` | error | `scheduler.go:497` |
| `could not record a scheduled task run` | error | `scheduler.go:563` |
| `a scheduled task changed under its run; the schedule was left alone` | info | `scheduler.go:565` |
| `could not hand a scheduled task to the node holding its session` | warn | `internal/api/scheduled_delivery.go:577` |
| `the node holding a scheduled task's session did not take it` | warn | `scheduled_delivery.go:580` |
| `could not deliver a peer's scheduled task` | error | `scheduled_delivery.go:623` |
| `could not record the session a scheduled task opened` | error | `scheduled_delivery.go:690` |
| `could not give back a scheduled task's call` | error | `scheduled_delivery.go:745` |
| `could not record the session a scheduled task rang into` | error | `scheduled_delivery.go:786` |
| `could not end a scheduled task's call` | error | `scheduled_delivery.go:793` |
| `a scheduled task run by hand did not land` | error | `internal/api/scheduled_tasks.go:574` |
| `could not record a scheduled task run by hand` | error | `scheduled_tasks.go:578` |
| `could not read when a scheduled task last ran` | error | `scheduled_tasks.go:637` |
| `could not pause a deleted config's scheduled tasks` | error | `internal/api/configs.go:309` |
| `not keeping the conversation in Stream Chat` | warn | `internal/session/manager.go:340` |

Each line carries `task`, `run` and `customer` where it has them; the `error` field holds the
whole story the run's code leaves out.
