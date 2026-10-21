---
name: phonecheck
description: Place a live test phone call through a local or remote router and report where each hop ran (agent SFU, router address, Stream SIP host, SFUs used, person side) and how long it took. Use when asked to run a phone call check, to see which servers a call used, to test call latency sites, or to explain a phonecheck run.
---

# A call check with phonecheck

`acceleration/cmd/phonecheck` places a test call and logs where each hop of it ran. It only
logs; there is no pass or fail. Read `acceleration/cmd/phonecheck/README.md` for the flags and
for how the person side is set up by hand. This skill is how to run it end to end and clean up.

Calls cost money. In local mode the run also changes the Stream app's call hooks. Say both
before you start.

## 1. Check first (read-only)

Place nothing yet.

- **Which router.** Local (you start it) or remote (a URL the human gives you).
- **Which direction and provider.** `outbound` or `inbound`; `twilio` or `telnyx`.
- **Where the env lives.** No code here loads a `.env`; a process has only what its shell
  exports. Ask the human for the path of the `.env` (default: the repository root's `.env`) and
  call it `$ENV_FILE`. Every command below runs in bash with it loaded first:
  `( set -a; . "$ENV_FILE"; set +a; <command> )`.
- **The env is there.** The names the README lists for that provider, plus the Go SDK's router
  variables, checked in the process env after loading, names only, never values:

```bash
( set -a; . "$ENV_FILE"; set +a; for k in STREAM_API_KEY STREAM_API_SECRET TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TELNYX_API_KEY TELNYX_ACCOUNT_SID TELNYX_PERSON_CONNECTION_ID; do [ -n "${!k}" ] && echo "ok       $k" || echo "MISSING  $k"; done )
```

  Only the provider's own names matter; ignore the other vendor's.
- **The person side is set up** as the README says (voice URL or TeXML app, region). You cannot
  check this; ask the human to confirm it, and what to pass as `-region`.
- **Inbound only:** the customer has no other attached number and no other dispatch worker is
  connected. The tool refuses the first; ask about the second.

## 2. Ask the human, once

Before the hooks change and before any call, say in one message:

- local mode: the Stream app's call hooks will point at a tunnel, and go back afterwards;
- how many calls: `-runs` (default 1), each about `-talk` plus 30 to 40 s;
- that calls cost money.

Wait for a yes. Do not change hooks or place a call without it.

## 3. Local router

From `acceleration/`:

1. Start the router in the background with the env loaded, without printing values:
   `( set -a; . "$ENV_FILE"; set +a; go run ./cmd/router )`. Postgres must be reachable (`ROUTER_POSTGRES_DSN`), because the
   `call_sites` row needs it. Send its output to a file. Wait for the `listening` line.
2. Start a public tunnel to the router's address (`http://localhost:8080` unless configured
   otherwise) with whatever tunnel tool the human has. Note its public URL.
3. Point the hooks at it, env loaded: `( set -a; . "$ENV_FILE"; set +a; go run ./cmd/phone hooks -url <tunnel URL> )`. Add `-app <id>` only
   for a router in app mode.
4. From `acceleration/cmd/phonecheck`, with the env loaded the same way, run:

```bash
go run . outbound -provider twilio -router http://localhost:8080 -agent-number +15550100001 -person-number +15550100002 -log-format json
```

   Use `inbound` for the other direction, and add `-region`, `-runs`, `-talk` as agreed.
5. Collect the tool's log (stderr) and the router's lines for each run's `session_id`
   (the tool logs it in `the agent is on the call`).

The `call_sites` row appears about 30 to 40 s after hang-up. The tool waits for it.

## 4. Remote router

No router, no tunnel. The tool checks the router answers; a router that does not is an error
it reports. Run it as in section 3 with `-router <url>` and no router log to collect.

Say plainly that the remote router must already receive the app's call webhook. If it does not,
inbound calls never arrive and no `call_sites` row is written.

## 5. Clean up, always

Even when a step failed or the human stopped the run. In reverse order, and only what this run
started:

1. `go run ./cmd/phone hooks -remove <tunnel URL>` from `acceleration/`, env loaded.
2. Stop the tunnel.
3. Stop the router.

Local mode only. Check with `go run ./cmd/phone hooks` (read-only, env loaded) that the tunnel URL is gone.
If a call may still be up, the tool hangs the person's call up when it exits; tell the human if
it was killed hard.

## 6. Report

Per call, one short block from the logs:

- **Hops:** agent SFU, router address, Stream SIP host, SFUs used and where, person side and
  `region_label`. Each with how it is known (`observed`, `inferred`, `reported`).
- **Timings:** `connect_ms`, `talked_ms`, `total_ms`.
- **Gaps:** anything `not observed` or missing, said plainly: `person call not observed` (a
  Telnyx outbound call), no `call_sites` row, a host that did not resolve.

Never print secrets: tokens, keys, or the env's values.
