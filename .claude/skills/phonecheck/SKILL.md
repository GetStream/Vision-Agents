---
name: phonecheck
description: Test inbound or outbound phone calls through a local or remote router. Asks what to test, sets up numbers, hooks and the person side, places the calls with phonecheck, reports where each hop ran (agent SFU, router address, Stream SIP host, SFUs used, person side) and how long it took, then puts everything back. Use when asked to test phone calls, run a phone call check, see which servers a call used, test call latency sites, or explain a phonecheck run.
---

# Test phone calls with phonecheck

`acceleration/cmd/phonecheck` places a test call and logs where each hop ran. It only logs; no
pass or fail. `acceleration/cmd/phonecheck/README.md` has the flags and the setup details; this
skill asks what to test, sets it up, runs, reports and cleans up.

## Rules for every step

- **Env.** No code here loads a `.env`. Ask for its path (default: the repository root's
  `.env`), call it `$ENV_FILE`, and run every command in bash as
  `( set -a; . "$ENV_FILE"; set +a; <command> )`. `phone` commands run from `acceleration/`.
- **Never print secrets.** Check env by name only. Pipe vendor API answers through `jq` and keep
  only the fields named below.
- **Ask with AskUserQuestion,** one step at a time, defaults first. Accept several answers at once.
- **State note.** Keep a short note in the session of everything changed: router and tunnel
  started, tunnel URL, hooks added, numbers bought (vendor, number, vendor id), numbers attached,
  old voice URLs and `connection_id`s. Cleanup reads it.
- **Nothing that costs money or changes an account before the yes** (section "Approve").

## Wizard

Read only until "Approve". `phone list` applies migrations to the `ROUTER_POSTGRES_DSN`
database; it changes nothing else.

1. **Router.**
   - *Local (default):* from `acceleration/`, start `go run ./cmd/router` in the background,
     output to a file, env loaded. Postgres must be reachable. Wait for `listening`. Read
     `stream_app=<id>` from `stream: the deployment acts in its own Stream app`: that is
     `$CUSTOMER`. If `.env`'s `STREAM_ACCELERATION_CUSTOMER_ID` differs, say so. Run the
     read-only `go run ./cmd/phone hooks` and show what already uses the app's hooks.
   - *Remote:* ask the URL and the customer id. phonecheck checks it answers and keeps logs
     before placing anything, and refuses it otherwise. It must already receive the app's call
     webhook. `phone` commands only reach its numbers if `ROUTER_POSTGRES_DSN` and the Stream
     keys in `.env` are that router's; otherwise steps 2 and 5 are the human's.
2. **Numbers.** Existing (default) or new. For existing, show
   `go run ./cmd/phone list -customer "$CUSTOMER"` (it defaults to `demo`; never conclude
   "no numbers" from one id).
3. **What to test.** One combination, "pick one at random", or "all":

   | Person on | Router number on | Inbound (person calls router number) | Outbound (router number calls person) |
   |---|---|---|---|
   | Twilio or Telnyx | Twilio | yes | yes |
   | Twilio or Telnyx | Telnyx | yes | yes |
   | Twilio or Telnyx | customer's own SIP trunk (`sip_trunk`) | no, phonecheck refuses it | yes |

   - Inbound: the router number must be attached, and no other number of the customer may be
     attached. No other dispatch worker of the customer may be connected; ask.
   - Outbound: use an unattached router number. Outbound from an attached number fails today
     while its own trunk holds the number (known limitation).
   - One router number per vendor can do both: run outbound first, then attach it and run
     inbound. There is no `phone detach`, so a second inbound router number needs the first
     one released. Before that run, ask about releasing it, separately; if no, skip that run and
     say so.
   - `sip_trunk` numbers are registered through the router's API (`/v1/phone/trunks/{id}/numbers`),
     not `phone`. Use an existing one, or the human registers it.
4. **Person number,** one per person vendor, not held by the router for the customer.
   Skill buys it (default) or the human does.
   - Twilio, with `-u "$TWILIO_PERSON_ACCOUNT_SID:$TWILIO_PERSON_AUTH_TOKEN"`: find one with
     `GET https://api.twilio.com/2010-04-01/Accounts/$TWILIO_PERSON_ACCOUNT_SID/AvailablePhoneNumbers/US/Local.json?VoiceEnabled=true&PageSize=1`,
     buy with `POST …/IncomingPhoneNumbers.json -d PhoneNumber=<n>`; keep `sid`.
   - Telnyx, with `Authorization: Bearer $TELNYX_PERSON_API_KEY`: find one with
     `GET https://api.telnyx.com/v2/available_phone_numbers?filter[country_code]=US&filter[features][]=voice&filter[limit]=1`
     (`curl -g`), buy with `POST /v2/number_orders`
     `{"phone_numbers":[{"phone_number":"<n>"}],"connection_id":"$TELNYX_PERSON_CONNECTION_ID"}`;
     keep the id from `GET /v2/phone_numbers?filter[phone_number]=<n>`.
5. **Router number,** per router vendor. Skill buys it (default) or the human does:
   `go run ./cmd/phone search -vendor <twilio|telnyx> -country US -limit 3`, then
   `go run ./cmd/phone buy -vendor <v> -number <n> -country US -customer "$CUSTOMER" -tag purpose=phonecheck`.
   For inbound, `go run ./cmd/phone attach -number <n> -customer "$CUSTOMER"`. A local Telnyx
   router also needs `TELNYX_PUBLIC_KEY`, `ROUTER_AUTH_KEK` and its own TeXML app in
   `TELNYX_CONNECTION_ID`, whose voice URL points at the router (the human sets that).
6. **Person setup.** Skill does it (default) or the human does it per README §4. Read and
   record the old value first.
   - Twilio, outbound: read `sid`, `voice_url`, `voice_method` from
     `GET …/IncomingPhoneNumbers.json?PhoneNumber=<n>`, then
     `POST …/IncomingPhoneNumbers/<sid>.json -d VoiceUrl=<echo URL> -d VoiceMethod=GET`.
     Inbound needs nothing.
   - Telnyx, both directions: the number on `TELNYX_PERSON_CONNECTION_ID`
     (`PATCH /v2/phone_numbers/<id>` `{"connection_id":…}`, record the old one). Outbound also:
     read `voice_url`, `voice_method` from `GET /v2/texml_applications/$TELNYX_PERSON_CONNECTION_ID`,
     then `PATCH` it to `{"voice_url":"<echo URL>","voice_method":"get"}`. That changes every
     number on the app; say so.
   - The echo URL is README §4's: `https`, `Say` first, `Pause length` raised to match `-talk`.
7. **Talk and runs.** `-talk` (default `20s`), `-runs` (default 1), `-region` label (default
   empty; ask which region the person number is set to).
8. **Afterwards.** Keep or release the numbers bought: not asked now. Asked after the runs
   (section "Clean up").

Check the env by name (only the chosen vendors' names matter):

```bash
( set -a; . "$ENV_FILE"; set +a; for k in STREAM_API_KEY STREAM_API_SECRET ROUTER_POSTGRES_DSN TWILIO_PERSON_ACCOUNT_SID TWILIO_PERSON_AUTH_TOKEN TELNYX_PERSON_API_KEY TELNYX_PERSON_ACCOUNT_SID TELNYX_PERSON_CONNECTION_ID TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TELNYX_API_KEY TELNYX_CONNECTION_ID TELNYX_PUBLIC_KEY ROUTER_AUTH_KEK; do [ -n "${!k}" ] && echo "ok       $k" || echo "MISSING  $k"; done )
```

The `*_PERSON_*` account may be the router's: `TWILIO_PERSON_ACCOUNT_SID=$TWILIO_ACCOUNT_SID`
in `.env`, after the router's lines. The router's `TWILIO_*`/`TELNYX_*` names matter for a local
router and for `phone search`/`buy`.

## Approve

Show ONE summary and wait for an explicit yes:

- the calls: each combination × runs, in order, each about `-talk` plus 30 to 40 s;
- numbers to buy, with monthly cost (router side from `phone search`; person side the vendor's
  price, about $1 a month for a US local number);
- account changes: hooks to a tunnel URL, voice URLs or TeXML app changed, numbers attached;
- each call costs a few cents.

No yes, nothing happens. Then do steps 4 to 6 as agreed, writing each change to the state note.

## Run

Local only: start `cloudflared tunnel --url http://localhost:8080` in the background, take its
`https://<random>.trycloudflare.com` URL, then `go run ./cmd/phone hooks -url <tunnel URL>`
(add `-remove <old URL>` for a gone tunnel still listed; `-app <id>` only for a router in app
mode).

Re-run `go run ./cmd/phone list -customer "$CUSTOMER"` and check the rules in step 3. Then from
`acceleration/cmd/phonecheck`, per combination, env loaded:

```bash
go run . outbound -provider twilio -router http://localhost:8080 -customer "$CUSTOMER" -agent-number +15550100001 -person-number +15550100002 -talk 20s -runs 1 -region us-east -log-format json
```

`-provider` is the person's vendor; `inbound` for the other direction. The tool waits up to 90 s
for the `call_sites` row. Keep its stderr and, locally, the router's lines with each run's
`session_id` (logged in `the agent is on the call`).

## Report

Per call, one short block:

- **Hops:** agent SFU, router address, Stream SIP host, SFUs used and where, person side and
  `region_label`, each with `observed`, `inferred` or `reported`.
- **Timings:** `connect_ms`, `talked_ms`, `total_ms`.
- **Gaps:** anything missing, plainly: `person call not observed` (Telnyx outbound), no
  `call_sites` row, a host that did not resolve, a run skipped and why.

## Clean up, always

On every exit: success, failure, or the human stopping. Reverse order, only what the state note
lists:

1. Put back each voice URL, method and Telnyx `connection_id` recorded.
2. `go run ./cmd/phone hooks -remove <tunnel URL>`; check with `go run ./cmd/phone hooks` that it
   is gone.
3. Stop the tunnel, then the router.
4. Ask, separately, whether to release the numbers bought for this test (irreversible; the
   monthly charge stops). If yes: router numbers with
   `go run ./cmd/phone release -number <n> -customer "$CUSTOMER"` (it also unwires an attached
   one); person numbers with Twilio `DELETE …/IncomingPhoneNumbers/<sid>.json` or Telnyx
   `DELETE /v2/phone_numbers/<id>`. If no, list what is kept and what stays attached: an
   attached number blocks the next inbound test on another number.

If the tool was killed hard, a call may still be up; tell the human.
