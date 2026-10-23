---
name: telnyx
description: How an agent texts people over SMS on a Telnyx number - the telnyx channel connector, staging's permanent SMS line (+1 218 702 1098), setting up a line locally or on staging, testing it live from a phone (a reply, a scheduled reminder), and what to check when a text gets no answer. Read before you set up, test or debug an SMS line, or touch the telnyx connector. Phone calls are the phonecheck skill.
---

# Telnyx: SMS lines

An agent answers people over SMS on a Telnyx number through the `telnyx` channel connector.
A person texts the number. Telnyx posts a webhook to the router. The channel bridge gives the
text to the agent bound to the line. The bridge sends the reply from the same number. Paths
below are under `acceleration/`.

```
person's phone ──SMS──► +1 218 702 1098 ──► Telnyx messaging profile 4001a107…
                                              │ webhook_url
                                              ▼
            router POST /v1/connectors/events/telnyx/{provider app id}
              verifies Telnyx-Signature-Ed25519 → channel bridge → thread session → agent
                                              │
person's phone ◄──SMS── POST https://api.telnyx.com/v2/messages (from the line, to the person)
```

Phone calls on Telnyx numbers are a different path: read the `phonecheck` skill.

## Terms

| Term | What it is | Example |
| --- | --- | --- |
| line | One Telnyx number that an agent answers on | +1 218 702 1098 |
| messaging profile | The Telnyx object that holds the number's `webhook_url` | `4001a107-bb22-4c6a-b01c-941d9024d596` |
| provider app | The router's record of the customer's Telnyx account: the messaging profile id, with the account's public key as the signing secret. One per app | `PUT /v1/agents/connectors/telnyx/oauth-client` |
| connection | The number as an app-owned `telnyx` connection, with the account's API key as its token | `81112518ea0565808d658c1464ed801b` |
| binding | A config's `connectors` entry that ties the connection to the agent (`fixed`) | `{"name": "sms", "connector_id": "telnyx", ...}` |
| thread | One person on one line. The person's number is the thread key | `+14165550123` on the line |

The manifest is `internal/connectors/providers/telnyx.yaml`: the inputs, the signature check,
which webhook is a person's message, and the send. `internal/channels/telnyx.go` is the older
`channels:` path, which moves onto the bridge in AI-921. Use the connector for anything new.

## Staging's SMS line

Staging has a permanent SMS line since 2026-10-23.

| What | Value |
| --- | --- |
| Number | +1 218 702 1098 (`TELNYX_SMS_NUMBER` in the repo `.env`) |
| Telnyx messaging profile | `4001a107-bb22-4c6a-b01c-941d9024d596` ("vision-agents mcp_plugins") |
| Its `webhook_url` | `https://accelerate.gcp.stream-io-api.com/v1/connectors/events/telnyx/4001a107-bb22-4c6a-b01c-941d9024d596` |
| Stream app | 1257545, keys `ATHENA_STREAM_API_KEY` / `ATHENA_STREAM_API_SECRET` (not `STREAM_API_KEY`, which is app 1728173) |
| Provider app | `4001a107-bb22-4c6a-b01c-941d9024d596`, signing secret `TELNYX_PUBLIC_KEY` |
| Connection | `81112518ea0565808d658c1464ed801b`, app-owned, token `TELNYX_API_KEY` |
| Config | `d6187808c0b169a2d2adf279d4aae97e` "staging-sms-line": text, `llm-fast`, `scheduling.enabled`, `default_timezone` America/Toronto, binding `sms` |

**US phones get no reply.** The number has no approved 10DLC campaign, so US carriers refuse
every reply with Telnyx error 40010, «Not 10DLC registered». The agent still answers; the
carrier drops the text. On 2026-10-23 a US T-Mobile phone got no reply, and a Canadian phone
(Fido) got every reply. Phones in other countries are not tested. The number has
`features.sms.international_outbound` false (`GET /v2/phone_numbers/%2B12187021098/messaging`),
which can mean no SMS to them; check the outbound record (below) after a test. The Stream brand is verified, but its campaign
(`4b3001a1-0de5-ae6e-5501-5f7aeb071f04`, "Internal Sandbox for Stream AI agents") is
`TELNYX_FAILED` over its opt-in description and privacy policy, and the number is in no
campaign. Check it:

```bash
curl -s -H "Authorization: Bearer $TELNYX_API_KEY" https://api.telnyx.com/10dlc/campaign/4b3001a1-0de5-ae6e-5501-5f7aeb071f04 | jq '{campaignStatus, failureReasons}'
curl -sg -H "Authorization: Bearer $TELNYX_API_KEY" "https://api.telnyx.com/10dlc/phone_number_campaigns?filter[telnyx_phone_number]=%2B12187021098" | jq .totalRecords
```

To try it, text the number from your phone. To test a feature on it, change the config, or bind
the connection to another config: the next text on any thread, old or new, is answered by the
config the line is bound to then (AI-1915, staging v0.6.139). Never make a second `telnyx` connection for this number: the
bridge finds a connection by its number (`store.AppConnectionByAccount`).

Before 2026-10-23 the profile pointed at an ngrok URL, `https://6f8b-24-8-28-137.ngrok-free.app/sms`.
It was someone's local router, offline since then. To test this number on a local router,
point the profile at your tunnel, then point it back at staging.

## Call the staging API

Every call needs `api_key` on the URL, `stream-auth-type: jwt`, and a server token: HS256
`{"server": true}` signed with the app's secret. Never print the secret or the token.

```bash
set -a; . ./.env; set +a
TOKEN=$(node -e '
const c = require("node:crypto");
const b = (v) => Buffer.from(JSON.stringify(v)).toString("base64url");
const now = Math.floor(Date.now() / 1000);
const input = b({ alg: "HS256", typ: "JWT" }) + "." + b({ server: true, iat: now, exp: now + 3600 });
process.stdout.write(input + "." + c.createHmac("sha256", process.env.ATHENA_STREAM_API_SECRET).update(input).digest("base64url"));
')
R=https://accelerate.gcp.stream-io-api.com
A=(-H "Authorization: Bearer $TOKEN" -H 'stream-auth-type: jwt' -H 'Content-Type: application/json')
K="api_key=$ATHENA_STREAM_API_KEY"
curl -s "$R/v1/agents/connections?connector_id=telnyx&$K" "${A[@]}" | jq '.items[] | {id, status, account_id}'
```

Telnyx's own API takes `Authorization: Bearer $TELNYX_API_KEY` at `https://api.telnyx.com/v2`.
In a `filter[...]` query, write the number's `+` as `%2B`, and call curl with `-g`: without it,
curl reads the brackets as a pattern, and `-s` hides the error.

## Set up a line

Do these steps for a new number, or for the staging line if it is gone. Check each answer
before you go to the next step.

1. **Read the number's messaging profile, and write down its `webhook_url`.** You need it to
   go back.
   ```bash
   curl -sg -H "Authorization: Bearer $TELNYX_API_KEY" "https://api.telnyx.com/v2/phone_numbers?filter[phone_number]=%2B12187021098" | jq '.data[] | {id, messaging_profile_id}'
   curl -s -H "Authorization: Bearer $TELNYX_API_KEY" "https://api.telnyx.com/v2/messaging_profiles/$PROFILE" | jq '.data | {name, webhook_url, webhook_failover_url}'
   curl -s -H "Authorization: Bearer $TELNYX_API_KEY" "https://api.telnyx.com/v2/messaging_profiles/$PROFILE/phone_numbers" | jq '.meta.total_results'
   ```
   If the profile has more than one number, changing its webhook moves all of them. Ask the
   owner first.
2. **Register the provider app** (the profile id, with the account's public key):
   ```bash
   curl -s -X PUT "$R/v1/agents/connectors/telnyx/oauth-client?$K" "${A[@]}" \
     -d "$(jq -n --arg p "$PROFILE" --arg s "$TELNYX_PUBLIC_KEY" '{provider_app_id: $p, signing_secret: $s}')"   # 201
   ```
3. **Connect the number**, then give it the API key:
   ```bash
   CONN=$(curl -s -X POST "$R/v1/agents/connections?$K" "${A[@]}" \
     -d '{"connector_id": "telnyx", "owner": {"type": "app"}, "inputs": {"phone_number": "+12187021098"}}' | jq -r .id)   # pending
   curl -s -X PUT "$R/v1/agents/connections/$CONN/credentials?$K" "${A[@]}" \
     -d "$(jq -n --arg t "$TELNYX_API_KEY" '{expected_revision: 1, values: {token: $t}}')" | jq '{status, account_id}'   # connected
   ```
4. **Bind it to a config.** Every binding must send `tools` (`internal/api/configs.go:2281`).
   A channel binding has no tools, so it sends `"tools": []`.
   Turn on `scheduling` only if you want reminders.
   ```bash
   curl -s -X POST "$R/v1/agents/configs?$K" "${A[@]}" -d "$(jq -n --arg c "$CONN" '{
     name: "staging-sms-line", mode: "text", llm: "llm-fast",
     instructions: "Answer briefly. When asked for a reminder, schedule it with schedule_task, then confirm briefly.",
     scheduling: {enabled: true, default_timezone: "America/Toronto"},
     connectors: [{name: "sms", connector_id: "telnyx", connection: {type: "fixed", connection_id: $c}, tools: []}]}')" | jq .id
   ```
5. **Point the profile at the router**, and check that the route reaches it. An unsigned
   request must get the router's own 401, not the proxy's «api_key is required»:
   ```bash
   curl -s -X PATCH -H "Authorization: Bearer $TELNYX_API_KEY" -H 'Content-Type: application/json' \
     "https://api.telnyx.com/v2/messaging_profiles/$PROFILE" \
     -d "{\"webhook_url\": \"$R/v1/connectors/events/telnyx/$PROFILE\", \"webhook_failover_url\": null}"
   curl -s -X POST "$R/v1/connectors/events/telnyx/$PROFILE" -d '{}'   # 401 "the request is not signed by the provider"
   ```

On a local router the same steps work. Use `http://localhost:8080` and the local auth in the
`connectors` skill, and give Telnyx a tunnel URL (`ngrok http 8080`). Connectors must be on
in the router (`connectors` skill, step 1).

## Test it live

The person must be a real phone. Our Twilio account cannot text US numbers: it answers 400,
code 21408 (checked 2026-10-23). The repo `.env` sets `TWILIO_ACCOUNT_SID` twice, and the
later one has no numbers. So ask a human to text the line, and give them the exact words.

1. **A reply.** Text "hi" from a phone that gets replies, such as a Canadian one (see «Staging's SMS line»). A short
   answer comes back within a few seconds.

   A phone that texted the line before keeps its thread, and the agent reads the whole
   thread's history. It may act on old messages: on 2026-10-23, `llm-fast` tried to cancel a
   reminder deleted hours before, got "there is no task with that id", and still told the
   person it had cancelled it. Read what the agent did in the router logs (below), not only
   its reply.
2. **A scheduled reminder** (scheduled tasks, AI-1842 and AI-1843). Text "Please remind me
   in 2 minutes to drink water". A confirmation comes back. About 2 minutes later the reminder
   comes back on the same thread. Check the task and its run:
   ```bash
   curl -s "$R/v1/agents/scheduled-tasks?config_id=$CONFIG&$K" "${A[@]}" | jq '.items[] | {id, deliver_to, user_id, next_run_at, schedule}'
   curl -s "$R/v1/agents/scheduled-tasks/$TASK/runs?$K" "${A[@]}" | jq '.items[] | {state, delivered_to, attempts, error}'
   ```
   Pass: `deliver_to` is `["sms"]`, `user_id` is `telnyx-…` (the thread's person), the run is
   `delivered` with `delivered_to` `sms` after 1 attempt, and exactly one reminder arrives.
   The time is in the config's `default_timezone`: a texting person has no zone of their own
   (AI-1910).
3. **A control.** Something that must behave as before, for example an app user's task run by
   hand delivers to `conversation` (`scheduled-tasks` skill).
4. **Clean up.** Delete the test tasks. Leave the staging line as the table above says.

A live run on 2026-10-23 is logged step by step, with every call and answer, in the AI-1750
ticket comments (AI-1842, AI-1843).

## What the bridge does to a text

- **STOP, START and HELP never reach the agent** (`internal/channelbridge/keywords.go`).
  Telnyx answers them itself and says so in `data.payload.autoresponse_type`. The bridge
  answers only the rest, such as REVOKE and OPT OUT. STOP writes an opt-out. A person with an
  opt-out reaches no agent and gets no reply.
- **The sandbox gate** (`dlc.Gate`) can hold back a text or a reply on a line that is not
  registered for 10DLC. A text it refuses reaches no agent.
- **A webhook older than 5 minutes is refused** (`Telnyx-Timestamp`, `max_age: 5m` in the
  manifest). Telnyx may send a webhook twice. The bridge claims each message once (`store.ClaimChannelThreadMessage`), so a retried delivery is dropped.
- **The line's current binding answers.** A text on an existing thread is answered by the
  config bound to the line now, the same one the bridge picks (oldest binding first). The
  history stays in the thread channel, so a newly bound config sees the earlier messages.
  Before AI-1915 (staging v0.6.139), an old thread kept the config of its last session: a
  moved binding left it on the old config, and a deleted config gave no answer (`there is no
  agent config …`).
- **One turn per thread.** A session the hook opens is closed after the turn
  (`api/threadhooks.go` `answerThread`). The next text, or a reminder that comes due, reopens
  the conversation from the channel.
- **Some failed sends are sent again**, after 2 s, 10 s and 30 s (`Options.RetryBackoff`): no
  answer, a 5xx, a 429, or a refusal the scheme reads as transient or rate limited. Any other
  refusal, such as a 4xx, is not sent again (`internal/channelbridge/bridge.go:754-777`).

## When a text gets no answer

| What you see | Where it stopped | What to do |
| --- | --- | --- |
| Nothing at all, and the router logs nothing | The profile's `webhook_url` points somewhere else (an old tunnel) | Read the profile (step 1). Point it at the router (step 5) |
| The agent answered, but the phone got nothing | A US carrier refused the reply: Telnyx error 40010, the number has no approved 10DLC campaign | Read the outbound record's `errors` (below). Text from a non-US phone, such as a Canadian one, or get the number into an approved campaign (see «Staging's SMS line») |
| The proxy answers 401 «api_key is required» | The staging proxy does not pass the route | The route passed on 2026-10-23. See the `connectors` skill, «When it does not work» |
| The router answers 401 «not signed by the provider» to Telnyx | Wrong signing secret, or a clock older than 5 min | Register the provider app again with `TELNYX_PUBLIC_KEY` (step 2) |
| 200, but no reply | No connection has this number, or no config binds it (the router logs a Warn: «no agent config binds its line»). Two configs that bind it: the oldest answers, so the newer one stays silent | `GET /v1/agents/connections?connector_id=telnyx`, then the configs' `connectors`: exactly one config binds the connection |
| 200, but no reply, and the binding is right | The sandbox gate holds the text back, or the person has an opt-out | See «What the bridge does to a text» |
| No reply to STOP, START or HELP | Telnyx answered it, not the agent | Expected. Read `autoresponse_type` in the webhook |
| No reply after an earlier STOP | The person has an opt-out | Text START, or remove the opt-out |
| The agent answers, but the time is wrong | The config's `default_timezone` | Set it, or name the zone in the text (AI-1910) |
| A reminder never comes | Read the run's `error` | The reasons table in the `scheduled-tasks` skill |

To see what Telnyx did with each text and reply today, with the error code of a failed one:

```bash
curl -sg -H "Authorization: Bearer $TELNYX_API_KEY" "https://api.telnyx.com/v2/detail_records?filter[record_type]=messaging&filter[date_range]=today&page[size]=20" | jq -c '.data[] | {created_at, direction, cld, status, errors, carrier}'
```

Error codes: https://developers.telnyx.com/docs/messaging/messages/error-codes

To read what the staging router did with a text (the webhook, the agent's tool calls, any
error), read its pods' logs. The router pods are the ones without `gateway` or `rollup`:

```bash
C=gke_stream-gcp-2026-v2_us-east1_us-east1-acceleration
for p in $(kubectl --context $C -n accelerate-staging get pods -o name | grep -v -e gateway -e rollup); do
  kubectl --context $C -n accelerate-staging logs $p -c router --since=15m | grep -E 'connectors/events/telnyx|thread channel|a tool call came back|level=(ERROR|WARN)'
done
```

## Tests

```bash
cd acceleration
go test -tags integration -p 1 ./internal/api -run 'TestTelnyxChannelSuite|TestTelnyxSandboxSuite'
go test ./internal/channelbridge/
```

Each suite talks to a fake Telnyx. Integration tests need your own test database (`go-testing`
skill).

## Where the code is

| Path | What it does |
| --- | --- |
| `acceleration/internal/connectors/providers/telnyx.yaml` | The manifest: inputs, signature, message match, send |
| `acceleration/internal/channelbridge/` | A text to the agent, and the reply. Keywords, gate, retries (`AGENTS.md`) |
| `acceleration/internal/api/threadhooks.go` | The thread session one turn runs in |
| `acceleration/internal/api/scheduled_thread_delivery.go` | A reminder delivered back on its thread |
| `acceleration/internal/channels/telnyx.go` | The older `channels:` path, until AI-921 |
| `acceleration/cmd/phonecheck/` | Phone calls, not SMS (`phonecheck` skill) |
