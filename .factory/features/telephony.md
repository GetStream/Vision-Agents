# Telephony

[Sprint 4](../sprint4.md), "Phone numbers".

## Asked for

A standardisation layer for phone across eleven vendors — Twilio, Sinch, Telnyx, Bandwidth,
BICS, Infobip, Vonage, Tata Communications / Kaleyra, Bird, DIDWW, Plivo — supporting: search
for a number, buy a number, and connect a Stream call to SIP, inbound and outbound, per
[Stream's SIP docs](https://getstream.io/video/docs/api/sip/inbound-trunk/).

## What exists

[internal/phone](../../acceleration/internal/phone) is the contract. `phone.Provider` is the
five things every vendor agrees on: search, buy, release, point a number at the bridge, dial
out.

All eleven vendors are declared in
[phone.yaml](../../acceleration/internal/phone/phone.yaml) with their capabilities and the
credentials they need. [twilio](../../acceleration/internal/phone/twilio) and
[telnyx](../../acceleration/internal/phone/telnyx) are implemented; the other nine resolve to a
stub that refuses every operation *by name*. They list rather than being absent, so
`cmd/phone vendors` and `GET /v1/phone/vendors` show the whole landscape and say plainly which
two work.

```bash
go run ./cmd/phone vendors
go run ./cmd/phone search -vendor twilio -country US -area 512
go run ./cmd/phone buy -vendor twilio -number +15125551234 -tag project=support
go run ./cmd/phone attach -number +15125551234 -call support-line
```

The same operations are on the HTTP API under `/v1/phone`.

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

## What is recorded

A number bought is a `requests` row with modality `phone` and a row in `phone_numbers`. The
`phone_numbers` row is kept after the number is released, because it was billed and the history
of what a customer paid for must survive the number going away.

Phone is recorded but not routed: there is one vendor per number, chosen by the customer, so
the provider and route paths do not serve the modality while the statistics paths do.

## Not done

**Nine of eleven vendors.** Sinch, Bandwidth, BICS, Infobip, Vonage, Tata / Kaleyra, Bird,
DIDWW and Plivo are declared and stubbed. Each is a package implementing the six methods; the
contract has needed one addition since, `SendDigits` for
[transfer and IVR navigation](transfer.md), which is a small enough change in three sprints to
be evidence the shape is right.

**Post-connect call control beyond digits.** Nothing here can mute, hold, record on demand or
hang up a leg. Transfer works without any of it, because a transfer is a second call rather
than an operation on the first, but an agent that wanted to put somebody on hold would need
call control this does not wrap.
