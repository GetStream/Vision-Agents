# The dashboard

Asked for in [sprint 9](../sprint9.md), built in [sprint 15](../sprint15.md).

## Asked for

A dashboard showing the last calls, what they cost, and agent configs; number search and
purchase; and a call detail page with the summary, a Gong-style picture of who was speaking
when, the transcript and a review score. Sprint 15 added the part that makes it worth
opening: the decisions the conversation made, live, next to what the agent was hearing while
it made them.

## What exists

[dashboard/](../../dashboard) is a Next.js app that talks to the router straight from the
browser. It has no server of its own and no store: everything on a page came from the router
on this page load, which is why there is nothing to keep in step.

| Page            | What it shows                                                          |
| --------------- | ---------------------------------------------------------------------- |
| `/`             | The last five calls, how the week's turns went, and what the models cost |
| `/calls/{id}`   | One call: who was talking when, what it heard, every judgement it made, and what each turn took |
| `/agents`       | Agent configs: instructions, the model targets, skills, keyterms, knowledge |
| `/voices`       | [Voices of your own](voices.md), their recordings and each provider's binding state |
| `/telephony`    | Numbers held, and search across every vendor, buy, attach              |

Three things had to give for a browser to be the client: `withCORS` in
[server.go](../../acceleration/internal/api/server.go) driven by `ROUTER_CORS_ORIGINS`, a
`customer_id` query parameter on the session socket because a browser cannot set a header on
a WebSocket, and `dashboard/src/lib/api.d.ts` generated from the same
[openapi.yaml](../../acceleration/api/openapi.yaml) as the Go and Python clients, checked in
CI against the spec.

## A running call and a finished one read the same

The monitoring page draws the speaking timeline out of the decision log rather than out of
audio: the agent knows when it started answering and when it was cut off, because it decided
each of those. A picture derived from the reasoning cannot drift from it.

That log arrives two ways and looks identical either way — `decision` frames on the session
socket while the call is running, `GET /v1/agents/calls/{id}/events` once it has ended. A
call with no decisions recorded falls back to `getCallTimeline`, which is per-turn rather
than per-judgement and coarser for it. See [observability](observability.md).

## Not done

- **The review score is a placeholder.** `review_score` is a column nothing computes.
- **Spend is not broken down by tag**, though `GET /v1/{modality}/stats/tags` would serve it.
- **[Campaigns](campaigns.md) have no page**, only an API.
- **There is no login.** The customer is a header the router trusts, so this is an operator's
  tool rather than something to put in front of a customer.
