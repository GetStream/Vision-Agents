# internal/phone/outcome

How each phone call ended. The `Keeper` keeps one row per call the router placed
(`phone_calls`), and `decide` (`outcome.go`) is the one place that turns the vendor's reports
and the phone's join and leave into its stored outcome: `completed`, `busy`, `no_answer`,
`failed`, `canceled` or `unknown`. The API and billing read those values: add to them, never
change one.

`canceled` is also stored when the router hangs up a call its session left ringing
(`Keeper.Canceled`), decided as a vendor's final report: the first one kept wins.

## The metric

`router_phone_call_outcomes_total{vendor, direction, outcome}` (`metric.go`) counts each call
once, when its leg first ends. `phone.Service.tell` (`internal/phone/legs.go`) tells the
`Keeper` every leg end: a phone leaving, the Stream call ending, or the leg sweep.

| Label | Values |
| --- | --- |
| `vendor` | a vendor of `internal/phone/phone.yaml`, or `sip_trunk`; any other name is `other` |
| `direction` | `outbound`: a call the router placed. `inbound`: a call that arrived on a number |
| `outcome` | the stored outcome, `dropped`, or `refused`: a dial the vendor refused. An inbound call is `completed`, `failed` (the router refused it) or `dropped` |

Series: at most (vendors + 2) × 11 (8 outbound outcomes, 3 inbound). `NewKeeper` sets each to 0
(`seed`), so a pod's first call moves its series from 0 to 1 and `increase()` reads it.

Count once, whatever the order of events:

- **Outbound**: the event that first sets the row's `ended_at` counts the call, under the row's
  lock (`Keeper.decide`). A second end, or a vendor report after the end, counts nothing.
- **Inbound**: the store ends a leg once (`ended_at IS NULL` in `internal/store/calllegs.go`),
  and an inbound leg is never revived, so each end the `Keeper` is told is a new call.
- **Refused inbound**: an arriving call the router refuses is ended at Stream, whose events
  end its leg on whichever node they reach. The refusing node marks the caller's Stream session
  of the call in Redis (`phone.Service.Refused`), for `MaxCallLength` + `LegSweepEvery`, by when
  the leg sweep has ended a leg nothing else did. The node that ends the leg reads the mark and
  counts the call `failed`. The mark is per session: a number attached before per-caller calls
  has every caller in one call, and a later caller there was not refused.
- **Refused dial**: a dial the vendor was asked for and refused is counted `refused` when it
  refused (`Keeper.Refused`, from `phone.Service.call`). It has no row, so no leg end counts it
  again. A dial never sent (the context ended first, or the row for its trunk could not be
  written) and a request that went away while the vendor was dialling are not counted: the
  phone may be ringing.
- **Not counted**: a transfer's leg (an outbound leg with no row).
- **Known gap**: a join that Stream delivers again after the call ended revives the leg
  (`ConnectCallLeg`), and the stored outcome becomes `completed`. The call was already counted
  under the outcome known at its first end (`no_answer`, or `unknown`), and it is not counted again.

## What a dropped call is

A dropped call is an answered call that the router's agent left on the line: the call's
session closed before the phone left, for a reason that does not end the call. The caller then
hears nobody. Its stored outcome stays `completed`; only the metric says `dropped`.

The signal is the session row (`agent_sessions`, id = the Stream call id, as for a placed call
and an inbound call the router answers), read when the answered leg ends (`leftOnTheLine`):

- `closed_at` is at or before the leg's end, and
- `end_reason` is not one that ends the call (`EndsTheCall`): `agent_hangup` and `voicemail`
  end the Stream call, `transferred` leaves the caller with a person, and `hangup` comes after
  the caller left. Every other reason (`closed`, `shutdown`, a failure such as `model_failed`,
  `start` or `unanswered`, the agent giving up before the phone joined) and an empty one (a
  session that ended before reasons were kept) count as dropped.

| Case | Signal | Counted as |
| --- | --- | --- |
| The person hangs up | the phone leaves while the session is open | `completed` |
| The agent hangs up (`end_call`) | session end reason `agent_hangup` | `completed` |
| The agent stops, fails or the router shuts down mid-call | session closed with `closed`, `shutdown`, a failure reason (`agent_failed`, `model_failed`, ...) or an empty reason, before the phone left | `dropped` |
| The phone joins after the agent gave up waiting | session closed with `start` or `unanswered` before the phone left | `dropped` |
| The network or the vendor cuts the call | none: the phone leaves, as when the person hangs up. The outcome reads no hang-up cause from the vendors' reports: a SIP code in `phone.CallStatus` (Telnyx's `SipHangupCause`, for one) feeds only the SIP code series | `completed` (not detected) |
| The router process dies without closing the session | none: the session row stays open | `completed` (not detected) |
| The session close is written after the phone left (the person hangs up within the records writer's delay) | `closed_at` is not yet written when the leg ends | `completed` (not detected) |
| An inbound call a worker answers (dispatch, no router session) | none: no session row | `completed` (not detected) |
| An incognito session drops the call | none: an incognito session writes no session row (`internal/session/manager.go`) | `completed` (not detected) |
| The router refuses an arriving call: a suspended app, an organization that may not pay, a router with no sessions, the number's agent config gone, its agent could not start, or no worker could take the call (`dispatch.Assign` fails, `refuseArrivingCall`, `internal/api/callhooks.go`) | the Redis mark of the refusal, read when the leg ends | `failed` |
| A refused call's leg ends before its mark is written, or the mark cannot be written or read (Redis down, or the router has none), or is not read within `refusedReadTimeout` (`internal/phone/legs.go`) | no mark, logged at WARN when Redis fails | `completed` (not detected) |
| An arriving call to a number with no agent reaches a router that runs no dispatch: nobody answers, and the call is not ended | none: no refusal mark and no session row, not logged | `completed` (not detected) |
| Nothing says the leg ended (lost events) | the leg sweep ends it at `MaxCallLength`, logged at ERROR | its stored outcome |
| The session row cannot be read | a store error, logged at WARN | its stored outcome |

An alert on the share of failed or dropped calls per vendor reads
`outcome=~"failed|dropped|refused"` over all outcomes of that vendor and direction. `refused`
is a dial the vendor refused, so a vendor that refuses every dial raises the share.

## Carrier KPIs

What the carrier of a placed call did, per vendor, beside the outcome. Each is counted once a
call, under the row's lock, from what the vendor reports (`phone.CallStatus`, read by the
vendor's `ReadStatus`) or from the call's join and leave. A vendor that does not report a
value has none counted for it; nothing is guessed.

| Metric | What | Counted |
| --- | --- | --- |
| `router_phone_call_sip_responses_total{vendor, direction, code}` | the final SIP response to the INVITE: a code of RFC 3261 §21, any other as its class (`4xx`) | at the call's first final report (`finalReportKept`) |
| `router_phone_post_dial_delay_seconds{vendor, direction}` | RFC 6076 §4.3: the INVITE to its first response other than 100, 401, 402 or 407 (the router's own bridge), or the dial to the vendor's first ringing report (router clock, so it includes the vendor API and its webhook) | at that response or report |
| `router_phone_short_calls_total{vendor, direction}` | an answered call under 6 s (Twilio's `pstn_short_duration`) | where `router_phone_call_outcomes_total` counts the call |

`direction` is `outbound` for the first two: the router sees no carrier response for an
inbound call. Series: (vendors + 2) × 50 codes, (vendors + 2) × 14 for the histogram (11 buckets, +Inf, sum,
count), (vendors + 2) × 2 short. `seed` sets all of them to 0 too, for the vendors the keeper names and
`other`, under the directions each is counted with.

| Vendor | SIP code | Post-dial delay | Source |
| --- | --- | --- | --- |
| `twilio` | `SipResponseCode` of the final StatusCallback | no: only `completed` is asked for | [call resource](https://www.twilio.com/docs/voice/api/call-resource) |
| `telnyx` (TeXML) | `SipHangupCause` of the completed StatusCallback, documented as conditional; the ringing callback has none | the `ringing` callback (`StatusCallbackEvent` is `ringing completed`); no SIP code comes with it, so it is the dial to the callback, not RFC 6076's INVITE to 180/183 | [outbound call](https://developers.telnyx.com/api-reference/texml-rest-commands/initiate-an-outbound-call) (lists `StatusCallbackEvent` for the account-scoped endpoint, not the connection-scoped one the router dials; on staging on 2026-11-08 that endpoint took it, and Telnyx posted `ringing` and `completed` for 3 of 3 calls), [ringing](https://developers.telnyx.com/api-reference/callbacks/texml-call-ringing), [completed](https://developers.telnyx.com/api-reference/callbacks/texml-call-completed) |
| `bandwidth` | `sipResponseCode` of `disconnect`, for a SIP rejection only | no | [disconnect](https://dev.bandwidth.com/docs/voice/webhooks/disconnect) |
| `plivo` | no: `hangup_url` carries none | no: `ring_url` is not set | [make a call](https://www.plivo.com/docs/voice/api/call/make-a-call) |
| `vonage` | `sip_code` of busy, unanswered, rejected, failed, completed | the `ringing` event | [webhook reference](https://developer.vonage.com/en/voice/voice-api/webhook-reference) |
| `sip_trunk` | the final response to the router's own INVITE; 200 for an answered call | the bridge's INVITE | `internal/phone/sipbridge` |
| `bird`, `didww`, `sinch` | no: they post no status | no | |

Short calls are counted for every vendor: they come from Stream's join and leave.

Ratios, per vendor:

- network failures (NER style): `outcome=~"failed|refused"` over all outcomes of
  `router_phone_call_outcomes_total`; `busy` and `no_answer` are the callee's. Vonage stores
  `rejected` as `busy`, and that includes carrier rejections: read its `sip_responses` codes.
- SEER (RFC 6076 §4.7), for a vendor that reports a code on every final report (`twilio`,
  `vonage`, `sip_trunk`): `code=~"200|480|486|600|603"` over all codes but `3..`.
- short calls: `router_phone_short_calls_total` over `outcome=~"completed|dropped"`.

Known gaps: a vendor that posts two final reports after the leg ended has both counted. A
Bandwidth call that was answered carries no code, so its codes are rejections only.
