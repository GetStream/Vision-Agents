# Shared trunks

Not in a sprint.

## Asked for

Make [telephony](telephony.md) usable for [campaigns](campaigns.md).

- Parallel calls from one number failed after the first. They also failed for about 30
  seconds after a call ended.
- An attached number could not place calls.
- Per-call trunks leaked toward Stream's cap of 25 trunks per app.
- Each inbound caller in a call of their own.

The root cause was one thing. The router made a Stream trunk and a routing rule per call, with
the From number on the trunk. Stream allows a number on only one trunk per app, so a second
call from the same number was refused. A per-call trunk was deleted only on
`call.session_ended`, so a call that never reported its end kept its trunk.

## What exists

One Stream trunk per Stream app, customer and vendor. It holds up to 50 numbers, which is
the most Stream allows on a trunk. When it is full, a sibling trunk opens. `sip_trunk` is one
more vendor, so a customer's own trunk numbers group the same way.

Each number keeps a routing rule of its own, its line.
[lines.go](../../acceleration/internal/phone/lines.go) makes it: `Service.line` puts a number
on the emptiest trunk with room, or opens a trunk, then creates the rule.

```
number +1555...  ->  rule of its own  ->  shared trunk (app, customer, vendor)
number +1556...  ->  rule of its own  ->  same trunk
```

| Piece                                                                          | What it does |
| ------------------------------------------------------------------------------ | ------------ |
| [lines.go](../../acceleration/internal/phone/lines.go)                          | `line`: join a trunk, adopt an old one, undo a failed join |
| [service.go](../../acceleration/internal/phone/service.go)                      | `Call` and `Transfer` use the line; `Release` takes the number off; the sweep |
| [stream.go](../../acceleration/internal/phone/stream.go)                        | `UpdateTrunk`, `UpdateRoute`, `CreateCall` |
| [phone.go](../../acceleration/internal/phone/phone.go)                          | `StreamLegger`, `StreamLegOf`, `Bridge.WithCallID` |
| [store/streamtrunks.go](../../acceleration/internal/store/streamtrunks.go)      | The `stream_trunks` table and the per-group lock |
| [store/callresources.go](../../acceleration/internal/store/callresources.go)    | `call_resources`, now with a `shared` flag |
| [callhooks.go](../../acceleration/internal/api/callhooks.go)                    | The hook skips a call the router placed |
| [phonesweep.go](../../acceleration/cmd/router/phonesweep.go)                    | The sweeper, every 10 minutes |
| [the migration](../../acceleration/migrations/20261113120000_shared_trunks.sql)  | `stream_trunks`, `attached_at`, `call_resources.id` and `shared` |

### One trunk for inbound and outbound

Stream allows a number on one trunk per app. A second trunk for outbound would be refused. So
the same trunk and the same rule serve both directions.

### One call id template tells the two apart

Every rule names its call with `{{x_stream_call_id}}{{#unless x_stream_call_id}}{{uuid}}{{/unless}}`.

- An outbound leg carries `x-stream-call-id` set to the session id. It lands in
  `agent:<session>`, where the agent already is.
- A callback carries no header. It gets a new uuid, so each inbound caller has a call of their
  own.

The obvious template is `{{#if x_stream_call_id}}…{{else}}{{uuid}}{{/if}}`. Stream refused the
leg with it, live, on 2026-10-23, so the `{{#unless}}` form stays.

### The header rides on the leg to Stream

- **Twilio and Telnyx** append `?x-stream-call-id=` to the `<Sip>` URI. Tested live. Telnyx
  does not document this.
- **sipbridge** sends an `X-Stream-Call-Id` INVITE header on the Stream leg only. The
  customer's leg never sees it.

It does not go through `Outbound.Headers`. Every vendor but didww refuses those.

### Providers say whether they share

A provider that can carry the header implements `StreamLegger`. `StreamLegOf` reads it and
answers per call for one that does not. Twilio, Telnyx and `sip_trunk` share. Vonage, Plivo,
Bird, Sinch and Bandwidth keep a trunk per call, as before.

That keeps the choice in the provider, where the knowledge is. A sixth vendor opts in by
adding one method.

### The router creates the agent's call first

When a call has `custom`, the router creates the Stream call with it before dialing, with
`created_by` set to `phone-router`. The agent creates the call only when it joins, and a
shared rule carries no custom data of its own, so otherwise the leg would create the call
without it.

### What is stored

- **`stream_trunks`.** One row per shared trunk, with its password sealed. Stream returns the
  password only when the trunk is made, so it is kept to let the next number join.
- **`phone_numbers.attached_at`.** A number can sit on a trunk only to place calls. It is
  attached for inbound only when this is set. The API shows `stream_trunk_id` only then, so
  the public shape did not change.
- **`call_resources`.** Keyed by its own id, with a `shared` flag. Every placed call records a
  row before dialing. That keeps call sites, sandbox minutes and the hook working. The hook
  skips any call that has a row, so the router does not dial its own outbound call again.

### Changes to a group run one at a time

A change to a group runs under a Postgres advisory lock for that group. It also runs under an
in-process mutex for the group. Without the mutex, every call in a campaign burst would wait
on the database lock with a pooled connection in hand, and the pool would run out.

Release takes the number off its trunk under the same lock. It deletes the trunk when that
was the last number.

### Old numbers are adopted

A number attached before this change has a trunk of its own. That trunk becomes a shared
trunk of one, and its rule is updated to the new template. Nothing moves at Stream.

### The sweeper

Every 10 minutes it handles call rows older than 6 hours. For a per-call row it deletes the
trunk, since the end hook never came. For a shared row it forgets the row and leaves the
trunk, because the line belongs to the number and not to the call.

### Fallbacks

- No `ROUTER_AUTH_KEK`, or no stored password: the call uses a trunk of its own.
- A number attached in another app: that call uses a per-call trunk in this app.

Both still place the call. Neither is an error.

### Transfers

A transfer from a shared vendor uses the From number's line. The header is the call id of the
call the agent is on. This holds for the `agent` call type. A transfer into another call type
keeps a trunk and rule of its own, pinned to that call. See [transfer](transfer.md).

### Capacity

Say a number dials about 75 times a day, a vendor rule of thumb. With 25 trunks of 50 numbers,
one app holds 1,250 numbers. One trunk per number would have meant 25 numbers and about 4
calls at once.

### Verified live, 2026-10-23

- Two outbound calls at once from one attached Twilio number.
- Two inbound calls at once.
- Telnyx outbound.
- `custom` kept on the call.
- One trunk per vendor after 6 calls.
- Release removes the trunk and the rule.

## Not done

- **Five vendors still dial a trunk per call.** Vonage, Plivo, Bird, Sinch and Bandwidth.
  Whether each can carry the header is unchecked.
- **A rare race.** Attach in app A and a call from the same number in app B, at the same
  moment, can leave the row attached on A's trunk while A's trunk lacks the number. The fix is
  to run `AttachNumber` under the group lock.
- **The deployment's own app holds at most 25 customer and vendor pairs.** Each pair needs a
  trunk, and the app caps at 25.
- **Trunk credentials must never reach customers.** Anyone holding a shared trunk's password
  can put a leg into any agent call in that app with `x-stream-call-id`.
- **Rollout.** A number with a live per-call trunk at deploy time cannot join a shared trunk
  until that trunk is deleted. The sweeper does that after up to about 6 hours.
