# Campaigns

[Sprint 9](../sprint9.md), "Campaign API".

## Asked for

An API for outbound calling: define the concurrency, point at an agent config, and give each
person you are calling their own instructions.

## What exists

[internal/campaign](../../acceleration/internal/campaign) is the runner and
[store/campaigns.go](../../acceleration/internal/store/campaigns.go) is the two tables behind
it, a campaign and its contacts. A campaign is `draft` until started, then `running`,
`paused` or `finished`.

```
POST /v1/agents/campaigns              name, agent config, concurrency
POST /v1/agents/campaigns/{id}/contacts numbers, each with its own instructions
POST /v1/agents/campaigns/{id}/start
POST /v1/agents/campaigns/{id}/pause
```

Running one holds a semaphore of `concurrency` slots. Each slot claims a contact, places the
call through [telephony](telephony.md), creates a session from the named config and waits for
the call to end before taking another. The contact's own instructions go last in the prompt,
after the config's, so they are the most recent thing the model read.

Parallel calls from one number work for Twilio, Telnyx and SIP trunks. They go through shared
trunks, so a second call from a number no longer fails while the first is up. See
[shared trunks](shared-trunks.md). The other five vendors still make a trunk per call.

## A contact shows how its call ended

Listing a campaign's contacts gives each one rung the `outcome` of its call, read by its
`call_id` from [telephony](telephony.md#how-a-placed-call-ended). The `state` says only whether
they were rung: a contact nobody answered is `done`, with `outcome` `no_answer`. Nobody is rung
again for it. A call reported busy, failed or canceled before its session opened is `done`
the same way, but its session was never opened, so its `call_id` names no session or call
row: the outcome is on the contact only.

## Contacts are claimed, not handed out

`ClaimContact` marks a row `SKIP LOCKED`. Two routers running the same campaign therefore
split the list between them instead of both ringing the same person, and a router that dies
mid-call leaves a claimed contact rather than a lost one.

Concurrency is enforced per process, which is the honest thing to say about it: it is a limit
on this router, not on the campaign. Two routers running one campaign run it twice as fast as
asked.

## Not done

- **No page in the [dashboard](dashboard.md)**, only the API.
- **No update or delete**, for either a campaign or a contact.
- **Needs the whole stack.** Without Postgres, telephony and sessions the endpoints say so
  rather than half-working.
