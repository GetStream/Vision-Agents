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

## 1. Check first

Place nothing and change nothing yet. For a local router the customer and the list need it running, so start it (section 3, step 1) before the two checks that follow; it changes nothing. (`phone list` and `phone hooks` below only read, though
`phone list` applies migrations to the database `ROUTER_POSTGRES_DSN` points at.)

- **Which router.** Local (you start it) or remote (a URL the human gives you).
- **Which direction and provider.** `outbound` or `inbound`; `twilio` or `telnyx`.
- **Where the env lives.** No code here loads a `.env`; a process has only what its shell
  exports. Ask the human for the path of the `.env` (default: the repository root's `.env`) and
  call it `$ENV_FILE`. Every command below runs in bash with it loaded first:
  `( set -a; . "$ENV_FILE"; set +a; <command> )`.
- **The env is there.** The names the README lists for that provider, plus the Go SDK's router
  variables, checked in the process env after loading, names only, never values:

```bash
( set -a; . "$ENV_FILE"; set +a; for k in STREAM_API_KEY STREAM_API_SECRET STREAM_ACCELERATION_CUSTOMER_ID ROUTER_POSTGRES_DSN TWILIO_PERSON_ACCOUNT_SID TWILIO_PERSON_AUTH_TOKEN TELNYX_PERSON_API_KEY TELNYX_PERSON_ACCOUNT_SID TELNYX_PERSON_CONNECTION_ID TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TELNYX_API_KEY TELNYX_CONNECTION_ID TELNYX_PUBLIC_KEY ROUTER_AUTH_KEK; do [ -n "${!k}" ] && echo "ok       $k" || echo "MISSING  $k"; done )
```

  Only the provider's own names matter; ignore the other vendor's.
  The `*_PERSON_*` names are the account that places and receives the person's calls, which the
  tool reads. It may be the router's account: in `.env`, after the router's lines, write
  `TWILIO_PERSON_ACCOUNT_SID=$TWILIO_ACCOUNT_SID` instead of copying secrets. The router's
  `TWILIO_*` / `TELNYX_*` names only matter for a local router, which is the one that reads
  them. `TELNYX_PUBLIC_KEY` and `ROUTER_AUTH_KEK` only matter for a local Telnyx router: the public
  key checks the calls Telnyx sends it, and the keyring seals a Telnyx number's trunk password
  when it is attached and opens it on each call. A local Telnyx router also needs a TeXML app of
  its own in `TELNYX_CONNECTION_ID`: `acceleration/README.md`, "Telnyx for a local router".
  `STREAM_ACCELERATION_CUSTOMER_ID` is not needed when you pass `-customer`, and for a local
  router it is the wrong id unless it matches the startup line.
  `ROUTER_POSTGRES_DSN` only matters for a local router.
- **The customer.** For a local router it is the Stream app id, not `STREAM_ACCELERATION_CUSTOMER_ID`:
  the local router runs in deployment mode and prints the id at startup, in
  `stream: the deployment acts in its own Stream app` (`stream_app=<id>`). The local dashboard
  uses the same id. Take it from that line once the router is up (section 3, step 1), call it
  `$CUSTOMER`, and pass `-customer "$CUSTOMER"` to both `phone list` and the tool. If `.env`'s
  `STREAM_ACCELERATION_CUSTOMER_ID` names a different id, tell the human. For a remote router the
  human gives the customer id.
- **The numbers belong to the right side.** The agent number must be one the router under test
  holds for that customer (inbound: and has attached), and the person number must be one it does
  not hold. For a local router, from `acceleration/`, env loaded, with the router running, list
  what it holds. Pass `-customer`: `phone list` defaults to `demo`.

```bash
( set -a; . "$ENV_FILE"; set +a; go run ./cmd/phone list -customer "$CUSTOMER" )
```

  The agent number must be listed; for inbound it must also have a trunk, and no other row may
  have one. The person number must not be listed. If the agent number is missing, or for
  inbound its trunk is `-` or another row has a trunk, or the person number is listed, show the
  human the list and stop. Buying or attaching is the human's call, never yours. Never conclude
  that the router holds no numbers from one customer id. For a remote router there is no list to
  read; the tool checks both numbers against it and refuses a wrong one before placing anything.
- **The person side is set up** as the README says (voice URL or TeXML app, region). You cannot
  check this; ask the human to confirm it, and what to pass as `-region`.
- **You change nothing at any provider, and attach nothing.** No voice URLs, no TeXML apps, no
  `phone attach`, no `phone buy`. Those spend money or take calls from somewhere else, and the
  README sets them up by hand. If one is needed, say exactly what and stop for the human.
- **Inbound only:** no other dispatch worker is connected. The list above covers other attached
  numbers; ask the human about this one.
- **Local only: what already uses the app's hooks.** From `acceleration/`, env loaded, run the
  read-only `go run ./cmd/phone hooks` and show the human what it lists. Adding a tunnel URL
  changes where Stream sends the app's call events, so they should see what else is there first.

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
   `call_sites` row needs it. Send its output to a file. Wait for the `listening` line, and read `stream_app=<id>` from the
   `the deployment acts in its own Stream app` line: that is `$CUSTOMER`. Then run the number check
   from section 1.
2. Start a public tunnel to the router's address (`http://localhost:8080` unless configured
   otherwise) with whatever tunnel tool the human has. Note its public URL.
3. Point the hooks at it, env loaded: `( set -a; . "$ENV_FILE"; set +a; go run ./cmd/phone hooks -url <tunnel URL> )`. Add `-app <id>` only
   for a router in app mode.
4. From `acceleration/cmd/phonecheck`, with the env loaded the same way, run:

```bash
go run . outbound -provider twilio -router http://localhost:8080 -customer "$CUSTOMER" -agent-number +15550100001 -person-number +15550100002 -log-format json
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
