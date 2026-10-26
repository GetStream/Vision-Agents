---
name: phonecheck
description: Test inbound or outbound phone calls through a local or remote router. Asks what to test, sets up numbers, hooks and the person side, places the calls with phonecheck, reports where each hop ran (agent SFU, router address, Stream SIP host, SFUs used, person side) and how long it took, then puts everything back. Use when asked to test phone calls, run a phone call check, see which servers a call used, test call latency sites, or explain a phonecheck run.
---

# Test phone calls with phonecheck

`acceleration/cmd/phonecheck` places a test call and logs where each hop ran. It only logs; no
pass or fail. `acceleration/cmd/phonecheck/README.md` has the flags and setup details. This
skill asks what to test, sets it up, runs, reports and puts things back.

## Rules for every step

- **Main session only.** The wizard (every question and the Approve) runs in the main session
  with the human, never in a sub-agent. If this skill is invoked inside a sub-agent, stop and
  say the wizard must run in the main session. Sub-agents only read logs and write the
  per-call report.
- **Long processes run in the background.** Start the router, the tunnel and each phonecheck run
  with the Bash tool's `run_in_background`, as
  `bash -c 'set -o pipefail; … 2>&1 | tee -a <scratchpad>/<name>.log'`, so the output shows in
  the Claude Code shell panel and is kept in its own log in the scratchpad:
  `<scratchpad>/router.log`, `<scratchpad>/tunnel.log`, `<scratchpad>/phonecheck-<n>.log`.
  `pipefail` keeps the process's own exit status, not `tee`'s. The background task's process is
  the `bash -c` wrapper, not the process inside it: record and stop the inner one as each step
  says. Right after starting one, print
  its log's full path, so the human can `tail -f` it or ask questions while the call runs. The
  main session stays free to answer.
- **Shell.** The user's shell is fish. Run every snippet as `bash -c '…'`. Shell variables do not
  last between tool calls: write literal values (env file path, router address, customer id,
  numbers, ids) into each command.
- **Env.** No code here loads a `.env`. Ask for its path (default: the repository root's
  `.env`) and load it in every command: `bash -c 'set -a; . <env file>; set +a; <command>'`.
  `go run` commands run from `acceleration/`.
- **Never print secrets.** Check env by name only. Keep credentials out of process args: pass
  them to curl on stdin with `-K -`, for example
  `printf 'user = "%s:%s"\n' "$TWILIO_PERSON_ACCOUNT_SID" "$TWILIO_PERSON_AUTH_TOKEN" | curl -sS -K - …`
  or `printf 'header = "Authorization: Bearer %s"\n' "$TELNYX_PERSON_API_KEY" | curl -sS -K - …`.
  Pipe every reply through `jq` and keep only the fields named here.
- **Voice URLs: classify, never print.** A vendor's voice URL or connection setting can embed a
  password: an attached router number's Twilio voice URL holds the Stream SIP trunk's. Never
  print a voice URL; keep one only after the credential check in "Voice URLs" below passes.
- **Encode numbers.** A `+` in a query must be encoded: `curl -G --data-urlencode 'PhoneNumber=+15550100002'`,
  and the same for Telnyx `filter[…]` keys. In a router path it stays as is.
- **Ask with AskUserQuestion,** one step at a time, defaults first. Accept several answers at once.
- **State file.** `<scratchpad>/phonecheck-state.md`. Write a line right after each change, and
  before a change write the values needed to undo it: vendor, number, vendor id, old voice URL
  and method (person numbers only, and only as "Voice URLs" allows), old `connection_id`,
  tunnel URL, hooks added, router and tunnel process ids. A section "bought this run" lists
  numbers the skill bought. No secrets in it. Cleanup reads it.
- **Never touch a number the skill did not buy.** It attaches and releases only numbers listed
  under "bought this run". An existing number is used as it is.

## Voice URLs

Write `<scratchpad>/voice.jq` once:

```jq
def risky: (. // "") | ascii_downcase | test("sip(:|%3a)|dial|@|%40|\\?.*(token|key|secret|password|auth)");
def kind: (. // "") as $u | ($u | ascii_downcase) as $l
  | if $u == "" then "empty"
    elif ($l | test("sip(:|%3a)")) then "attached (points at Stream SIP)"
    elif ($l | test("twimlets\\.com/echo")) and ($u | risky | not)
      then "test script (echo twimlet without <Dial>)"
    else "other" end;
```

To look at a number, print only the class: `jq -L <scratchpad> 'include "voice"; … | (.voice_url | kind)'`.
Only before changing a person number, save its old value straight into the state file, never
to the screen. The check refuses a value with `sip:`, `<Dial>` or `@` in it, or a query string naming a
token, key, secret, password or auth:

```bash
… | jq -L <scratchpad> -r 'include "voice"; .incoming_phone_numbers[0] | if (.voice_url | risky) then error("holds SIP credentials or a <Dial>") else "twilio <n> sid=\(.sid) old_voice_url=\(.voice_url // "") old_method=\(.voice_method)" end' >> <scratchpad>/phonecheck-state.md
```

Telnyx: the same on `.data` of `GET /v2/texml_applications/<id>`. If the check fails, do not
change that number: tell the human why and skip its runs.

## Router numbers: the router's API

Every router number step goes through the router's HTTP API, the way a customer does it. Never
the `phone` CLI for numbers. Write `<scratchpad>/router-auth.py` once:

```python
import base64, hashlib, hmac, json, os, sys, time

if len(sys.argv) > 1:  # local router in noauth: the header names the customer
    print(f'header = "X-Customer-Id: {sys.argv[1]}"')
    sys.exit()
b64 = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
now = int(time.time())
head = b64(b'{"alg":"HS256","typ":"JWT"}')
claims = b64(json.dumps({"server": True, "iat": now, "exp": now + 600}).encode())
key = os.environ["STREAM_API_SECRET"].encode()
sig = b64(hmac.new(key, f"{head}.{claims}".encode(), hashlib.sha256).digest())
print(f'header = "api_key: {os.environ["STREAM_API_KEY"]}"')
print('header = "Stream-Auth-Type: jwt"')
print(f'header = "Authorization: Bearer {head}.{claims}.{sig}"')
```

It prints curl config, so nothing secret is in an arg. Local: pass the customer id. Remote
(Stream's hosted router, behind its proxy): no argument; it signs a 10-minute server token with
`STREAM_API_SECRET`, and the customer is the app `STREAM_API_KEY` belongs to. Every call, with
`<auth>` the customer id or nothing:

```bash
bash -c 'set -a; . <env file>; set +a; python3 -I <scratchpad>/router-auth.py <auth> | curl -sS --fail-with-body -K - http://localhost:8080/v1/phone/numbers | jq "if (.error? // null) then .error | {code, message} else [.[] | {e164, vendor, stream_trunk_id}] end"'
```

| Step | Request | Keep |
|---|---|---|
| list | `GET /v1/phone/numbers` | `[].e164`, `.vendor`, `.stream_trunk_id` (present = attached) |
| search | `GET /v1/phone/numbers/available?vendor=<twilio\|telnyx>&country=US&features=voice&limit=3` | `.numbers[].e164`, `.vendor`, `.monthly_cost_micros`; `.skipped` |
| buy | `POST /v1/phone/numbers` with `-H 'Content-Type: application/json' -d '{"vendor":"<v>","e164":"<n>","country":"US","tags":{"purpose":"phonecheck"}}'` | `.e164`, `.vendor`, `.monthly_cost_micros` |
| attach | `-X POST /v1/phone/numbers/<n>/attach` with `-H 'Content-Type: application/json' -d '{}'` | `.trunk_id`, `.sip_uri` |
| release | `-X DELETE /v1/phone/numbers/<n>` with `-o /dev/null -w '%{http_code}\n'`, no `jq` | `204` |

`monthly_cost_micros` is millionths of a dollar. An error is `{"error":{…}}`: keep
`.error.code`, `.error.message`, as above.

## 1. Ask and look

Nothing here costs money or changes a vendor or Stream account. Starting a local router writes
to its database (migrations, and the router's own background work).

1. **Router.** Local (default) or remote. *Remote:* ask the URL. phonecheck checks it answers and
   keeps logs, and refuses it otherwise. It must already receive the Stream app's call webhook.
2. **Env names** (only the chosen vendors' matter):

   ```bash
   bash -c 'set -a; . <env file>; set +a; for k in STREAM_API_KEY STREAM_API_SECRET ROUTER_POSTGRES_DSN ROUTER_AUTH_KEK TWILIO_PERSON_ACCOUNT_SID TWILIO_PERSON_AUTH_TOKEN TELNYX_PERSON_API_KEY TELNYX_PERSON_ACCOUNT_SID TELNYX_PERSON_CONNECTION_ID TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TELNYX_API_KEY TELNYX_CONNECTION_ID TELNYX_PUBLIC_KEY; do [ -n "${!k}" ] && echo "ok       $k" || echo "MISSING  $k"; done'
   ```

   Remote needs only the Stream keys and the person's. Local also needs `ROUTER_POSTGRES_DSN`
   (the `call_sites` row), the router's vendor keys, and `ROUTER_AUTH_KEK` to attach a Telnyx
   number. The `*_PERSON_*` account may be the router's (`TWILIO_PERSON_ACCOUNT_SID=$TWILIO_ACCOUNT_SID`
   in `.env`, after the router's lines). See README §3.
3. **Local router.** Check nothing listens: `lsof -nP -iTCP:8080 -sTCP:LISTEN`; if something
   does, show it and ask. Starting it runs migrations, binds :8080 and may start background
   work that writes rows or calls models; ask a yes. Start
   `bash -c 'set -o pipefail; set -a; . <env file>; set +a; cd acceleration && ROUTER_AUTH_MODE=noauth go run ./cmd/router 2>&1 | tee -a <scratchpad>/router.log'`
   in the background;
   once it listens, write its process id (`lsof -tiTCP:8080 -sTCP:LISTEN`) to the state file. Wait for `listening`. If it never comes, show the last lines of
   the file and stop. Note the Stream app id from `stream_app=<id>` in
   `stream: the deployment acts in its own Stream app`, if that line appears.
4. **Customer.** Never assumed. *Local:* list numbers for `.env`'s
   `STREAM_ACCELERATION_CUSTOMER_ID`. If the app id is known and differs, list for it too: the
   local dashboard buys numbers under the app id. Show both lists and ask which customer to use
   (default: the one holding the numbers the human named). *Remote:* list; the key decides the
   customer, so ask the human for that app id. Use the literal id in every request and
   phonecheck command.
5. **What to test.** One combination, "pick one at random", or "all" = person vendor (Twilio,
   Telnyx) × router number vendor (Twilio, Telnyx, `sip_trunk`) × direction (inbound, outbound),
   minus what the rules forbid:
   - Inbound (person calls the router number): the router number must be attached, and no other
     number of the customer may be. No other dispatch worker of the customer may be connected;
     ask. phonecheck refuses a `sip_trunk` number.
   - Outbound (the router number calls the person): on Twilio, Telnyx and `sip_trunk`, an
     attached number may place calls. On Vonage, Plivo, Bird, Sinch and Bandwidth, use an
     unattached router number: a call placed from an attached number fails while the number's
     own trunk holds it.
   - A number the skill buys can do outbound first, then be attached for inbound. There is no
     detach: a second bought number for inbound needs the first one released mid-run. That
     release goes in the Approve summary.
   - `sip_trunk` numbers are not bought here. Use an existing one.
6. **Numbers.** Existing (default) or buy new, per side.
   - Existing router numbers: from the chosen customer's list (step 4). An existing number used
     for inbound must already be attached and be the only attached one; used for outbound on
     Vonage, Plivo, Bird, Sinch or Bandwidth it must be unattached. If not, show it and stop. Never attach or release it.
   - The person number must not be in that list.
   - New router number: search (see the table).
   - New person number: Twilio
     `GET https://api.twilio.com/2010-04-01/Accounts/<sid>/AvailablePhoneNumbers/US/Local.json`
     with `VoiceEnabled=true&PageSize=1` (keep `available_phone_numbers[].phone_number`); Telnyx
     `GET https://api.telnyx.com/v2/available_phone_numbers` with `filter[country_code]=US`,
     `filter[features][]=voice`, `filter[limit]=1` (keep `data[].phone_number`,
     `data[].cost_information.monthly_cost`).
7. **Talk and runs.** `-talk` (default `20s`), `-runs` (default 1), `-region` label (default
   empty; ask which region the person number is set to).
8. **Person setup.** Skill (default) or the human per README §4. Look at the current values now,
   classes only: Twilio `GET …/IncomingPhoneNumbers.json` with `PhoneNumber=<n>` (keep
   `incoming_phone_numbers[].sid`, `.voice_url | kind`, `.voice_method`); Telnyx
   `GET /v2/phone_numbers` with `filter[phone_number]=<n>` (keep `data[].id`,
   `data[].connection_id`), `GET /v2/texml_applications/<person app id>` (keep
   `data.voice_url | kind`, `data.voice_method`), and the numbers on that app,
   `GET /v2/phone_numbers` with `filter[connection_id]=<person app id>` (keep `data[].phone_number`).
9. **Local only: hooks.** Run the read-only `go run ./cmd/phone hooks` and show it. Note any
   stale tunnel URL to remove.
10. **Keep or release** the bought numbers: not asked now. Asked per number at cleanup.

## Approve

Show ONE summary and wait for an explicit yes:

- the calls, in order: each combination × runs, each about `-talk` plus 10 s plus 35 to 65 s for the `call_sites` row, so a minute or more, a few cents each;
- numbers to buy, vendor and monthly cost, and any mid-run release of a just-bought number;
- the router address and the customer id every request uses;
- account changes: hooks pointed at a tunnel URL (app-wide: while it is up the local router
  also gets the Stream app's other call events and its new-message events); any stale hook URL
  to remove; numbers to attach; Twilio voice URLs to set; the Telnyx person app's voice URL,
  with the numbers on that app listed (it changes all of them);
- for each person number whose voice setting is `other` (it points at some other app or
  webhook), say plainly: while the test runs, real calls to that number reach the test script
  instead; it is put back afterwards. Recommend a number used only for tests. Never print the URL;
- cleanup: what goes back (hooks, voice URLs, `connection_id`s, tunnel, router), what stays (the
  bought numbers, until you say otherwise), and that release is asked per number afterwards.

No yes: nothing is bought or changed; stop the router this skill started and end.

## 2. Do (after the yes)

In this order, each change into the state file:

1. **Tunnel and hooks** (local only). `command -v cloudflared`; if missing, ask the human. Start
   `bash -c 'set -o pipefail; cloudflared tunnel --url http://localhost:8080 2>&1 | tee -a <scratchpad>/tunnel.log'`
   in the background (it needs no env file), write cloudflared's own process id
   (`pgrep -f 'cloudflared tunnel --url http://localhost:8080'`) to the state file, take its
   `https://<random>.trycloudflare.com` URL, then `go run ./cmd/phone hooks -url <tunnel URL>`
   (`-remove <stale URL>` only if approved; `-app <id>` only for a router in app mode). A local
   Telnyx router's own TeXML app (`TELNYX_CONNECTION_ID`) needs its voice URL on the tunnel; the
   human sets that.
2. **Router numbers.** Buy (see the table); add to "bought this run". For inbound, after its
   outbound runs: attach.
3. **Person numbers.** Twilio `POST …/IncomingPhoneNumbers.json` with `PhoneNumber=<n>` (keep
   `sid`, `phone_number`). Telnyx `POST /v2/number_orders` with
   `{"phone_numbers":[{"phone_number":"<n>"}],"connection_id":"<person app id>"}` (keep
   `data.id`, `data.status`). Add to "bought this run".
4. **Person setup.** First save each old value as "Voice URLs" says. The echo URL is README §4's: `https`, `Say` first, `Pause length` matching
   `-talk`. Twilio, outbound: `POST …/IncomingPhoneNumbers/<sid>.json` with `VoiceUrl`,
   `VoiceMethod=GET`. Telnyx: put the number on the person app (`PATCH /v2/phone_numbers/<id>`
   `{"connection_id":"<person app id>"}`); outbound also `PATCH /v2/texml_applications/<id>`
   `{"voice_url":"<echo URL>","voice_method":"get"}`.

## Run

List the customer's numbers again and check the rules in 1.5. If another attached number
blocks an inbound run, stop and tell the human. Then, per combination `<n>` (1, 2, …), in the
background, one at a time:

```bash
bash -c 'set -o pipefail; set -a; . <env file>; set +a; cd acceleration/cmd/phonecheck && go run . outbound -provider twilio -router http://localhost:8080 -customer <customer> -agent-number +15550100001 -person-number +15550100002 -talk 20s -runs 1 -region us-east 2>&1 | tee -a <scratchpad>/phonecheck-<n>.log'
```

`-provider` is the person's vendor; `inbound` for the other direction. Print the log's path. The
tool waits up to 90 s for the `call_sites` row and logs `call_sites row after N s` when it comes. Start the next run once this one's exit is
reported. To stop a run early, interrupt the tool itself, so it still hangs up and closes the
session: `pkill -INT -x phonecheck` (`go run` names the binary after its folder). Then check
`pgrep -x phonecheck` finds nothing.

## Report

After each run, dispatch one sub-agent (Agent tool) to read `<scratchpad>/phonecheck-<n>.log`
and, locally, the lines of `<scratchpad>/router.log` with that run's `session_id` (logged in
`the agent is on the call`). It returns the block below, so the large logs stay out of the
main session. Per call, one short block:

- **Hops:** agent SFU, caller SFU (and where Stream placed each side), router address, Stream
  SIP host (on a trunk call, Stream's media address from its SDP answer, `observed`), the
  customer's trunk when there is one, SFUs used, person side and `region_label`, each with `observed`, `inferred` or
  `reported`.
- **Timings:** `connect_ms`, `session_ms` (from the agent on the call until its session and the person's call ended), `person_call_s` (the call itself), `total_ms`, and when the `call_sites`
  row came.
- **Gaps:** anything missing, plainly: `person call not observed` (Telnyx outbound), no
  `call_sites` row, a host that did not resolve, a run skipped and why.

## Clean up, always

On every exit: success, failure, or the human stopping. Read the state file and undo only what
it lists, in reverse:

1. Put back each voice URL, method and Telnyx `connection_id`.
2. `go run ./cmd/phone hooks -remove <tunnel URL>`; check with `go run ./cmd/phone hooks` that it
   is gone.
3. For each number under "bought this run", ask separately (number, vendor, attached or not):
   release or keep. Releasing is irreversible and stops the monthly charge. Router number:
   release (see the table; it also unwires an attached one), while the router still runs.
   Person number: Twilio `DELETE …/IncomingPhoneNumbers/<sid>.json`, Telnyx
   `DELETE /v2/phone_numbers/<id>`. Say what is kept and what stays attached: an attached
   number blocks inbound tests on other numbers.
4. Stop any phonecheck run left (`pkill -INT -x phonecheck`), then the tunnel, then the router:
   `kill <id>` for each process id in the state file, then their background tasks. Check that
   nothing is left: `pgrep -x phonecheck`, `pgrep -f 'cloudflared tunnel --url http://localhost:8080'`
   and `lsof -tiTCP:8080 -sTCP:LISTEN` each print nothing.

If a run was killed hard, a call may still be up and the state file says what is left to undo;
tell the human its path.
