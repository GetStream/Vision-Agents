# phonecheck

Places a live test phone call through a router and logs where each hop of it ran: the agent's
SFU, the router's address, the Stream SIP server, the SFUs the call used, and the person side.

It only logs. There is no pass or fail. It changes nothing at any vendor or at the router: it
checks, places, watches and ends calls. The person side is set up by hand first (see below).

Two directions:

- `outbound`: the router's agent calls the person number. The person number answers with a
  script you set up by hand.
- `inbound`: the tool calls the agent number as the person, with an inline script.

## 1. A router

**Local.** Start it from `acceleration/` with Postgres, because the `call_sites` row the tool
reads is only written when the router has a database (`ROUTER_POSTGRES_DSN`):

```bash
cd acceleration
go run ./cmd/router
```

Wait for the `listening` log line. The router listens on `http://localhost:8080` unless you
configured another address.

**Remote.** Pass `-router https://router.example.com`. The tool checks that it answers and that
it keeps logs: a remote router also needs Postgres, or the tool refuses it.

## 2. Stream's call webhook must reach the router

Needed for both directions. Inbound calls are handed to the router through it. On every call
it drives the `call_sites` row and releases an outbound call's trunk.

For a remote router, the Stream app's call webhook must already point at it. Locally, open a
public tunnel to the router, then point the app's hooks at it.

A Cloudflare quick tunnel needs no account. In its own terminal:

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8080
```

It prints a `https://<random>.trycloudflare.com` address. That address changes every time
the tunnel starts, so point the hooks at the new one each time:

```bash
cd acceleration
go run ./cmd/phone hooks -url https://<random>.trycloudflare.com
```

Stream refuses the update while a hook still points at a tunnel that is gone, so drop the
old one in the same run: add `-remove https://<old>.trycloudflare.com`.

When done, remove it:

```bash
go run ./cmd/phone hooks -remove https://<random>.trycloudflare.com
```

Stream sends `call.session_ended` about 30 s after the agent leaves, so the `call_sites` row
appears about 30 to 40 s after hang-up. The tool waits up to 90 s for it.

Inbound also needs:

- no other dispatch worker of the customer connected, on any router node;
- no other attached number on the customer. The tool refuses if there is one.

## 3. Environment

The person side is someone with a phone number of their own, so its provider account is set
apart from the router's: it can be the same account, or your own. The router's
`TWILIO_*` and `TELNYX_*` variables are never read for it.

| Variable | Used for |
|---|---|
| `STREAM_ACCELERATION_CUSTOMER_ID` | The customer. `-customer` overrides it |
| `STREAM_API_KEY`, `STREAM_API_SECRET` | As the Go SDK needs them for your router |
| `TWILIO_PERSON_ACCOUNT_SID`, `TWILIO_PERSON_AUTH_TOKEN` | `-provider twilio`: the account the person number is in |
| `TELNYX_PERSON_API_KEY` | `-provider telnyx`: the account the person number is in |
| `TELNYX_PERSON_ACCOUNT_SID` | `-provider telnyx`: that account's UUID, which Telnyx's call endpoints need in the path |
| `TELNYX_PERSON_CONNECTION_ID` | `-provider telnyx`: the TeXML app the person number is on. Only read. Not the router's `TELNYX_CONNECTION_ID` |
| `IPINFO_TOKEN` | Optional. Better place names for addresses |

The `*_PERSON_*` credentials are the account that places and receives the person's calls. It
may be the router's account. `.env` is loaded with `set -a; . .env`, so after the router's
lines it can say `TWILIO_PERSON_ACCOUNT_SID=$TWILIO_ACCOUNT_SID` instead of copying secrets.

The router's address is `-router` only. The tool checks the provider's variables before it
places anything.

## 4. Set up the person side by hand

**The tool changes nothing at any vendor.** The person number must not be registered on the
router for the customer. The tool refuses one that is.

### Twilio, outbound

On the person number, set the voice URL, method `GET`, to the echo twimlet below. It says
`This is a phonecheck test call.`, pauses 20 s, and hangs up. Use `https`: Twilio does not
follow the redirect `http` answers with.

```
https://twimlets.com/echo?Twiml=%3CResponse%3E%3CSay%3EThis%20is%20a%20phonecheck%20test%20call.%3C%2FSay%3E%3CPause%20length%3D%2220%22%2F%3E%3CHangup%2F%3E%3C%2FResponse%3E
```

The script must start with `Say`. A leading `Pause` makes Twilio wait before it answers.
For a longer `-talk`, raise `length="20"` to match.

Region, if wanted: the number's inbound processing region in the Twilio console.

### Twilio, inbound

Nothing on the number. The tool dials with an inline script. Inbound calls go through
Twilio's default region; another region is not supported by the tool.

### Telnyx

Make a TeXML app once. Its id is `TELNYX_PERSON_CONNECTION_ID`. Then:

1. Assign the person number to the app.
2. Set the app's voice URL to the same echo URL as above, method `GET`.
3. Set the app's anchor site as wanted.

For Telnyx outbound the person's call is not observed. The tool logs `person call not observed`
and waits out the talk time.

### Then

Pass `-region <label>` naming what you set, so the log says it (`region_label`). The tool does
not check it.

When you are done, put the number back as you found it.

## 5. Run

From `acceleration/cmd/phonecheck`:

```bash
go run . outbound -provider twilio -router http://localhost:8080 \
  -agent-number +15550100001 -person-number +15550100002 -region us-east

go run . inbound -provider telnyx -router https://router.example.com \
  -agent-number +15550100001 -person-number +15550100002
```

Flags, shared by both:

| Flag | Default | Meaning |
|---|---|---|
| `-provider` | none, required | `twilio` or `telnyx`: the person side's vendor |
| `-router` | `http://localhost:8080` | The router's address |
| `-customer` | empty | Customer id. Empty uses `STREAM_ACCELERATION_CUSTOMER_ID` |
| `-agent-number` | none, required | The router's number the agent is on |
| `-person-number` | none, required | The number the person side is on |
| `-talk` | `20s` | How long the person stays on |
| `-runs` | `1` | How many calls, one after another |
| `-sip-lookup-from` | `us` | Where to resolve the Stream SIP host from |
| `-region` | empty | Free label, logged as `region_label`. Changes nothing |
| `-log-format` | `text` | `text` or `json` |

`go run . outbound -provider twilio -h` lists them all.

## 6. Reading the output

One line per hop: `hop`, `host`, `address`, `place`, `how`. `how` says how we know:

- `observed`: the tool resolved it itself;
- `inferred`: resolved from a place you chose, not seen on the call;
- `reported`: the router or Stream said so.

A `person call` line gives the person side's status, start and end where the tool can see them.

Then one `call summary` line: `direction`, `provider`, `session_id`, `outcome`, `connect_ms`,
`talked_ms`, `total_ms`, `agent_sfu`, `router_address`, `sip_host`, `sfus_used`, `region_label`
(and `vendor_call_id` outbound). `-log-format json` writes all of it as JSON on stderr.

The router's own lines for the call carry the same `session_id`. Filter its log by it:

```bash
grep session_id=<id> router.log
```

## 7. Known limits

- The agent number's own vendor leg is not logged.
- The Stream SIP server is inferred from DNS, not seen on the call.
- The region is only the label you passed, unless the vendor's call record reports one.
- A Telnyx person answering an outbound call is not observed (`person call not observed`).

## 8. Tests

```bash
cd acceleration/cmd/phonecheck && go test ./...
```

This is its own module, so `go test ./...` in `acceleration/` skips it.
