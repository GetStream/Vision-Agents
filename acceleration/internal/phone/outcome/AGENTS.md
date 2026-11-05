# internal/phone/outcome

How each phone call ended. The `Keeper` keeps one row per call the router placed
(`phone_calls`), and `decide` (`outcome.go`) is the one place that turns the vendor's reports
and the phone's join and leave into its stored outcome: `completed`, `busy`, `no_answer`,
`failed`, `canceled` or `unknown`. The API and billing read those values: add to them, never
change one.

## The metric

`router_phone_call_outcomes_total{vendor, direction, outcome}` (`metric.go`) counts each call
once, when its leg first ends. `phone.Service.tell` (`internal/phone/legs.go`) tells the
`Keeper` every leg end: a phone leaving, the Stream call ending, or the leg sweep.

| Label | Values |
| --- | --- |
| `vendor` | a vendor of `internal/phone/phone.yaml`, or `sip_trunk`; any other name is `other` |
| `direction` | `outbound`: a call the router placed. `inbound`: a call that arrived on a number |
| `outcome` | the stored outcome, or `dropped`. An inbound call is `completed`, `failed` (the router refused it) or `dropped` |

Series: at most (vendors + 2) × 10 (7 outbound outcomes, 3 inbound).

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
- **Not counted**: a transfer's leg (an outbound leg with no row), and a dial the vendor
  refused (`Keeper.Failed`: it never rang; `router_requests_total` counts the failed dial).
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
| The network or the vendor cuts the call | none: the phone leaves, as when the person hangs up. The vendors' reports carry no hang-up cause (`phone.CallStatus`) | `completed` (not detected) |
| The router process dies without closing the session | none: the session row stays open | `completed` (not detected) |
| The session close is written after the phone left (the person hangs up within the records writer's delay) | `closed_at` is not yet written when the leg ends | `completed` (not detected) |
| An inbound call a worker answers (dispatch, no router session) | none: no session row | `completed` (not detected) |
| An incognito session drops the call | none: an incognito session writes no session row (`internal/session/manager.go`) | `completed` (not detected) |
| The router refuses an arriving call: a suspended app, an organization that may not pay, a router with no sessions, the number's agent config gone, or its agent could not start (`refuseArrivingCall`, `internal/api/callhooks.go`) | the Redis mark of the refusal, read when the leg ends | `failed` |
| A refused call's leg ends before its mark is written, or the mark cannot be written or read (Redis down, or the router has none), or is not read within `refusedReadTimeout` (`internal/phone/legs.go`) | no mark, logged at WARN when Redis fails | `completed` (not detected) |
| No worker takes an arriving call to a number with no agent (`dispatch.Assign` fails, `internal/api/callhooks.go`): nobody answers, and the call is not ended | none: no refusal mark and no session row, logged at ERROR ("nobody could answer an arriving call") | `completed` (not detected) |
| An arriving call to a number with no agent reaches a router that runs no dispatch: nobody answers, and the call is not ended | none: no refusal mark and no session row, not logged | `completed` (not detected) |
| Nothing says the leg ended (lost events) | the leg sweep ends it at `MaxCallLength`, logged at ERROR | its stored outcome |
| The session row cannot be read | a store error, logged at WARN | its stored outcome |

An alert on the share of failed or dropped calls per vendor reads
`outcome=~"failed|dropped"` over all outcomes of that vendor and direction.
