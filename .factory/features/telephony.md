# Telephony

[Sprint 4](../sprint4.md), "Phone numbers", widened by
[sprint 12](../sprint12.md) to eight vendors and
[sprint 13](../sprint13.md) to outbound calls at seven of them.

## Asked for

A standardisation layer for phone across eleven vendors — Twilio, Sinch, Telnyx, Bandwidth,
BICS, Infobip, Vonage, Tata Communications / Kaleyra, Bird, DIDWW, Plivo — supporting: search
for a number, buy a number, and connect a Stream call to SIP, inbound and outbound, per
[Stream's SIP docs](https://getstream.io/video/docs/api/sip/inbound-trunk/).

## What exists

[internal/phone](../../acceleration/internal/phone) is the contract. `phone.Provider` is the
things vendors agree on: search, buy, release, point a number at the bridge, dial out, press
digits.

All eleven vendors are declared in
[phone.yaml](../../acceleration/internal/phone/phone.yaml) with their capabilities, the
credentials they need and what this service can do with them. Eight are implemented; the other
three resolve to a stub that refuses every operation *by name*. They list rather than being
absent, so `cmd/phone vendors` and `GET /v1/phone/vendors` show the whole landscape and say
plainly which work.

Being implemented means three different things, so each vendor declares its `operations`:

- **Everything.** [twilio](../../acceleration/internal/phone/twilio) and
  [telnyx](../../acceleration/internal/phone/telnyx), including answering on a number.
- **Numbers and outbound calls.** [sinch](../../acceleration/internal/phone/sinch),
  [bandwidth](../../acceleration/internal/phone/bandwidth),
  [vonage](../../acceleration/internal/phone/vonage),
  [bird](../../acceleration/internal/phone/bird) and
  [plivo](../../acceleration/internal/phone/plivo) search, buy, release and dial. They cannot
  `attach`: pointing one of their numbers at a trunk is a per-vendor application rather than
  a property of the number.
- **Numbers only.** [didww](../../acceleration/internal/phone/didww), which has no call
  control API at all — outbound at DIDWW is SIP origination against a `voice_out_trunks`
  resource you point your own switch at, with nothing to ask over HTTP. Seven of eight is
  therefore the ceiling, and `dial` is absent from its operations so a number is not bought
  from it for an agent that has to call people.

```bash
go run ./cmd/phone vendors
go run ./cmd/phone search -vendor twilio -country US -area 512
go run ./cmd/phone buy -vendor twilio -number +15125551234 -tag project=support
go run ./cmd/phone attach -number +15125551234 -call support-line
go run ./cmd/phone dial -from +15125551234 -to +15550001111 -ring-timeout 20s
```

The same operations are on the HTTP API under `/v1/phone`.

## Vendors do not agree on how to search

Telnyx filters by US state and Sinch does not. Plivo matches digits only at the front of a
number; Vonage matches at either end but not both at once. Bandwidth takes a city and a state
but no digit pattern. DIDWW names a country by its own identifier rather than its ISO code.

So `Provider.Supports(Filter)` is part of the contract: a vendor declares what its API can
express. Searching without naming a vendor asks every vendor that has its credentials, at
once, merges what they offer cheapest first, and reports in `skipped` every vendor it could
not ask and why. Dropping a filter a vendor cannot express would answer a search for Colorado
with numbers from Ohio, which reads as a result rather than as a gap.

Capabilities are the exception. Every vendor reports what a number carries even where it
cannot filter on it, so those are checked on the results instead of skipping the vendor.

```bash
go run ./cmd/phone search -country US -state CO -type local \
  -feature hd_voice -feature emergency -limit 5
```

## Inbound and outbound are not symmetric

Stream's SIP support is **inbound only** today, which shapes the whole feature:

- **Inbound** is what the docs describe. A number reaches an agent because the vendor sends the
  call to a Stream inbound trunk.
- **Outbound** is originated at the vendor and bridged into that same trunk, because there is
  nothing to ask Stream to dial with. It is the same call path arrived at from the other end,
  not a second mechanism.

Attaching a number creates the trunk and a routing rule whose caller id is a handlebars
template. That detail matters: it makes the SIP caller a participant with a stable id, which is
what per-participant transcription keys on. Without it the caller would be anonymous to the
agent and every call would look like the same speaker.

Placing a call does the same two things and pins the routing rule to a named call, so the
answered leg lands where an agent is waiting. The response says which call that is, because
an agent that is not in it hears nothing when the person picks up. In Python that is one
context manager:

```python
async with agent.outbound_call(from_=held, to=person, call_type="default", call_id="hello"):
    await agent.simple_response("greet the user and let them know you're a friendly AI agent")
    await agent.finish()
```

## Vendors do not agree on what a call can be asked for

Telnyx takes custom SIP headers on a dial and Twilio does not. Bird's call has no ring timer
at all; Sinch has a maximum call duration, which is a different promise. So a provider also
declares which call features it can express, and a call naming one it cannot is refused
rather than placed without it: a ring timeout that was silently dropped is a call sitting in
somebody's voicemail for a minute.

Getting past the trunk splits the eight two ways, and the split is not about politeness.
Twilio, Telnyx and Plivo can name SIP digest credentials in their call plans. Vonage's `sip`
endpoint, Bird's `transfer` step, Sinch's `connectSip` and Bandwidth's `<SipUri>` have no
password field anywhere, so those four are recognised by the address they call from, declared
as `signalling` in `phone.yaml`. Stream reads an empty allowlist as "accept everything"
rather than "password only" — confirmed by asking its pre-auth endpoint directly — so a
vendor that authenticates by address refuses to place a call until its addresses are
declared. A trunk with neither is a way into a customer's calls for anyone who learns its
uri.

Plivo and Bandwidth also refuse to take a call plan on the request that places a call, and
fetch one when the person answers. For those, placing a call parks the plan and hands the
vendor a single-use expiring token at `GET /v1/phone/answer/{token}` — the one unauthenticated
path on the API, because the vendor fetching it has no customer to name. That costs a public
`ROUTER_PUBLIC_URL` and buys one thing back at Bandwidth: the BXML runs on the person's own
leg, so its `<SendDtmf>` presses keys at them rather than at the trunk.

## What is recorded

A number bought is a `requests` row with modality `phone` and a row in `phone_numbers`. The
`phone_numbers` row is kept after the number is released, because it was billed and the history
of what a customer paid for must survive the number going away.

Phone is recorded but not routed: there is one vendor per number, chosen by the customer, so
the provider and route paths do not serve the modality while the statistics paths do.

## Not done

**Three of eleven vendors.** BICS, Infobip and Tata / Kaleyra are declared and stubbed. Each is
a package implementing the six methods; the contract has needed two additions across the
sprints since — `SendDigits` for [transfer and IVR navigation](transfer.md), and `Supports`
plus a country on a purchase for the six vendors added in sprint 12 — which is a small enough
change to be evidence the shape is right.

**Answering on a number at six of the eight implemented vendors.** Sinch, Bandwidth, Vonage,
Bird, DIDWW and Plivo buy numbers and, all but DIDWW, call out on them, but none can point
one at a Stream trunk. Each needs its own inbound work: an application answering with XML or
an NCCO, or a SIP peer on the vendor's own site.

**Dialling at DIDWW, ever.** Not a gap to close: there is no API to call.

**Post-connect call control beyond digits.** Nothing here can mute, hold, record on demand or
hang up a leg. Transfer works without any of it, because a transfer is a second call rather
than an operation on the first, but an agent that wanted to put somebody on hold would need
call control this does not wrap.
