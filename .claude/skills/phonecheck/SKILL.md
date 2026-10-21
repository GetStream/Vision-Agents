---
name: phonecheck
description: Test inbound or outbound phone calls through a local or remote router. Asks what to test, sets up numbers, hooks and the person side, places the calls with phonecheck, reports where each hop ran (agent SFU, router address, Stream SIP host, SFUs used, person side) and how long it took, then puts everything back. Use when asked to test phone calls, run a phone call check, see which servers a call used, test call latency sites, or explain a phonecheck run.
---

# Test phone calls with phonecheck

`acceleration/cmd/phonecheck` places a test call and logs where each hop ran. It only logs; no
pass or fail. `acceleration/cmd/phonecheck/README.md` has the flags and setup details. This
skill asks what to test, sets it up, runs, reports and puts things back.

## Rules for every step

- **Shell.** The user's shell is fish. Run every snippet as `bash -c '…'`. Shell variables do not
  last between tool calls: write literal values (env file path, customer id, numbers, ids) into
  each command.
- **Env.** No code here loads a `.env`. Ask for its path (default: the repository root's
  `.env`) and load it in every command: `bash -c 'set -a; . <env file>; set +a; <command>'`.
  `phone` commands run from `acceleration/`.
- **Never print secrets.** Check env by name only. Keep credentials out of process args: pass
  them to curl on stdin with `-K -`, for example
  `printf 'user = "%s:%s"\n' "$TWILIO_PERSON_ACCOUNT_SID" "$TWILIO_PERSON_AUTH_TOKEN" | curl -sS -K - …`
  or `printf 'header = "Authorization: Bearer %s"\n' "$TELNYX_PERSON_API_KEY" | curl -sS -K - …`.
  Pipe every reply through `jq` and keep only the fields named here.
- **Encode numbers.** A `+` in a query must be encoded: `curl -G --data-urlencode 'PhoneNumber=+15550100002'`,
  and the same for Telnyx `filter[…]` keys.
- **Ask with AskUserQuestion,** one step at a time, defaults first. Accept several answers at once.
- **State file.** `<scratchpad>/phonecheck-state.md`. Write a line right after each change, and
  before a change write the values needed to undo it: vendor, number, vendor id, old voice URL,
  old method, old `connection_id`, tunnel URL, hooks added, router and tunnel process ids. A
  section "bought this run" lists numbers the skill bought. No secrets in it. Cleanup reads it.
- **Never touch a number the skill did not buy.** It attaches and releases only numbers listed
  under "bought this run". An existing number is used as it is.

## 1. Ask and look

Nothing here costs money or changes a vendor or Stream account. Starting a local router and
`phone list` do write to the database (migrations, and the router's own background work).

1. **Router.** Local (default) or remote. *Remote:* ask the URL. phonecheck checks it answers and
   keeps logs, and refuses it otherwise. It must already receive the Stream app's call webhook.
2. **Database.** Before the router starts and before any `phone` command, show only the host of
   `ROUTER_POSTGRES_DSN`:

   ```bash
   bash -c 'set -a; . <env file>; set +a; python3 -I -c "import os,urllib.parse as u; print(u.urlsplit(os.environ[\"ROUTER_POSTGRES_DSN\"]).hostname)"'
   ```

   (A `key=value` DSN prints `None`; then show only its `host=` value.) If the host is not
   `localhost` or `127.0.0.1`, get a yes first. For a remote router, `phone` only sees its
   numbers if this DSN and the Stream keys are that router's; otherwise numbers are the human's.
3. **Env names** (only the chosen vendors' matter):

   ```bash
   bash -c 'set -a; . <env file>; set +a; for k in STREAM_API_KEY STREAM_API_SECRET ROUTER_POSTGRES_DSN TWILIO_PERSON_ACCOUNT_SID TWILIO_PERSON_AUTH_TOKEN TELNYX_PERSON_API_KEY TELNYX_PERSON_ACCOUNT_SID TELNYX_PERSON_CONNECTION_ID TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TELNYX_API_KEY TELNYX_CONNECTION_ID TELNYX_PUBLIC_KEY ROUTER_AUTH_KEK; do [ -n "${!k}" ] && echo "ok       $k" || echo "MISSING  $k"; done'
   ```

   The `*_PERSON_*` account may be the router's (`TWILIO_PERSON_ACCOUNT_SID=$TWILIO_ACCOUNT_SID`
   in `.env`, after the router's lines). See README §3.
4. **Local router.** Check nothing listens: `lsof -nP -iTCP:8080 -sTCP:LISTEN`; if something
   does, show it and ask. Starting it runs migrations, binds :8080 and may start background
   work that writes rows or calls models; ask a yes. From `acceleration/`, start
   `go run ./cmd/router` in the background, output to a file; write its process id to the state
   file. Wait for `listening`. If it never comes, show the last lines of the file and stop. Note
   the Stream app id from `stream_app=<id>` in `stream: the deployment acts in its own Stream app`,
   if that line appears. It is the app, not the customer.
5. **Customer.** Never assumed. List numbers for `.env`'s `STREAM_ACCELERATION_CUSTOMER_ID`
   (`go run ./cmd/phone list -customer <id>`; it defaults to `demo`, so always pass it). If the
   app id is known and differs, list for it too: the local dashboard buys numbers under the app
   id. Show both lists and ask which customer to use (default: the one holding the numbers the
   human named; for a remote router, ask). Use that literal id in every `phone` and phonecheck
   command.
6. **What to test.** One combination, "pick one at random", or "all" = person vendor (Twilio,
   Telnyx) × router number vendor (Twilio, Telnyx, `sip_trunk`) × direction (inbound, outbound),
   minus what the rules forbid:
   - Inbound (person calls the router number): the router number must be attached, and no other
     number of the customer may be. No other dispatch worker of the customer may be connected;
     ask. phonecheck refuses a `sip_trunk` number.
   - Inbound on a Telnyx router number the skill buys: `phone attach` cannot attach it. The
     `phone` CLI builds its service without a keyring, and Telnyx numbers need one
     (`ErrNoSealer`), whatever `ROUTER_AUTH_KEK` says. The human attaches it another way (for
     example the router's `POST /v1/phone/numbers/{e164}/attach`) or that row is skipped.
   - Outbound (the router number calls the person): use an unattached router number. A call
     placed from an attached number fails while the number's own trunk holds it.
   - A number the skill buys can do outbound first, then be attached for inbound. There is no
     `phone detach`: a second bought number for inbound needs the first one released mid-run.
     That release goes in the Approve summary.
   - `sip_trunk` numbers are registered through the router's API, not `phone`. Use an existing one.
7. **Numbers.** Existing (default) or buy new, per side.
   - Existing router numbers: from the chosen customer's list (step 5). An existing number used
     for inbound must already be attached and be the only attached one; used for outbound it
     must be unattached. If not, show it and stop. Never attach or release it.
   - The person number must not be in that list.
   - New router number: `go run ./cmd/phone search -vendor <twilio|telnyx> -country US -limit 3`
     (`NUMBER`, `PER MONTH`).
   - New person number: Twilio
     `GET https://api.twilio.com/2010-04-01/Accounts/<sid>/AvailablePhoneNumbers/US/Local.json`
     with `VoiceEnabled=true&PageSize=1` (keep `available_phone_numbers[].phone_number`); Telnyx
     `GET https://api.telnyx.com/v2/available_phone_numbers` with `filter[country_code]=US`,
     `filter[features][]=voice`, `filter[limit]=1` (keep `data[].phone_number`,
     `data[].cost_information.monthly_cost`).
8. **Talk and runs.** `-talk` (default `20s`), `-runs` (default 1), `-region` label (default
   empty; ask which region the person number is set to).
9. **Person setup.** Skill (default) or the human per README §4. Read the current values now:
   Twilio `GET …/IncomingPhoneNumbers.json` with `PhoneNumber=<n>` (keep
   `incoming_phone_numbers[].sid`, `.voice_url`, `.voice_method`); Telnyx
   `GET /v2/phone_numbers` with `filter[phone_number]=<n>` (keep `data[].id`,
   `data[].connection_id`), `GET /v2/texml_applications/<person app id>` (keep
   `data.voice_url`, `data.voice_method`), and the numbers on that app,
   `GET /v2/phone_numbers` with `filter[connection_id]=<person app id>` (keep `data[].phone_number`).
10. **Local only: hooks.** Run the read-only `go run ./cmd/phone hooks` and show it. Note any stale tunnel URL to remove.
11. **Keep or release** the bought numbers: not asked now. Asked per number at cleanup.

## Approve

Show ONE summary and wait for an explicit yes:

- the calls, in order: each combination × runs, each about `-talk` plus 30 to 40 s, a few cents each;
- numbers to buy, vendor and monthly cost, and any mid-run release of a just-bought number;
- the customer id every command uses;
- account changes: hooks pointed at a tunnel URL (app-wide: while it is up the local router
  also gets the Stream app's other call events and its new-message events); any stale hook URL
  to remove; numbers to attach; Twilio voice URLs to set; the Telnyx person app's voice URL,
  with the numbers on that app listed (it changes all of them);
- cleanup: what goes back (hooks, voice URLs, `connection_id`s, tunnel, router), what stays (the
  bought numbers, until you say otherwise), and that release is asked per number afterwards.

No yes: nothing is bought or changed; stop the router this skill started and end.

## 2. Do (after the yes)

In this order, each change into the state file:

1. **Tunnel and hooks** (local only). `command -v cloudflared`; if missing, ask the human. Start
   `cloudflared tunnel --url http://localhost:8080` in the background, take its
   `https://<random>.trycloudflare.com` URL, then `go run ./cmd/phone hooks -url <tunnel URL>`
   (`-remove <stale URL>` only if approved; `-app <id>` only for a router in app mode). A local
   Telnyx router's own TeXML app (`TELNYX_CONNECTION_ID`) needs its voice URL on the tunnel; the
   human sets that.
2. **Router numbers.** `go run ./cmd/phone buy -vendor <v> -number <n> -country US -customer <customer> -tag purpose=phonecheck`;
   add to "bought this run". For inbound, after its outbound runs:
   `go run ./cmd/phone attach -number <n> -customer <customer>` (Twilio only, see 1.6).
3. **Person numbers.** Twilio `POST …/IncomingPhoneNumbers.json` with `PhoneNumber=<n>` (keep
   `sid`, `phone_number`). Telnyx `POST /v2/number_orders` with
   `{"phone_numbers":[{"phone_number":"<n>"}],"connection_id":"<person app id>"}` (keep
   `data.id`, `data.status`). Add to "bought this run".
4. **Person setup.** The echo URL is README §4's: `https`, `Say` first, `Pause length` matching
   `-talk`. Twilio, outbound: `POST …/IncomingPhoneNumbers/<sid>.json` with `VoiceUrl`,
   `VoiceMethod=GET`. Telnyx: put the number on the person app (`PATCH /v2/phone_numbers/<id>`
   `{"connection_id":"<person app id>"}`); outbound also `PATCH /v2/texml_applications/<id>`
   `{"voice_url":"<echo URL>","voice_method":"get"}`.

## Run

Re-run `go run ./cmd/phone list -customer <customer>` and check the rules in 1.6. If another
attached number blocks an inbound run, stop and tell the human. Then, per combination:

```bash
bash -c 'set -a; . <env file>; set +a; cd acceleration/cmd/phonecheck && go run . outbound -provider twilio -router http://localhost:8080 -customer <customer> -agent-number +15550100001 -person-number +15550100002 -talk 20s -runs 1 -region us-east -log-format json'
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

On every exit: success, failure, or the human stopping. Read the state file and undo only what
it lists, in reverse:

1. Put back each voice URL, method and Telnyx `connection_id`.
2. `go run ./cmd/phone hooks -remove <tunnel URL>`; check with `go run ./cmd/phone hooks` that it
   is gone.
3. Stop the tunnel, then the router.
4. For each number under "bought this run", ask separately (number, vendor, attached or not):
   release or keep. Releasing is irreversible and stops the monthly charge. Router number:
   `go run ./cmd/phone release -number <n> -customer <customer>` (also unwires an attached one).
   Person number: Twilio `DELETE …/IncomingPhoneNumbers/<sid>.json`, Telnyx
   `DELETE /v2/phone_numbers/<id>`. Say what is kept and what stays attached: an attached number
   blocks inbound tests on other numbers.

If a run was killed hard, a call may still be up and the state file says what is left to undo;
tell the human its path.
