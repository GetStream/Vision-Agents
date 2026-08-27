# Memory

[Sprint 4](../sprint4.md), "Memory", asked for again in [sprint 9](../sprint9.md). What an
agent looks up rather than remembers is [knowledge](knowledge.md).

## Asked for

Implement mem0 so agents have memory. Scope it by the API key or `app_id` using the backend
plus the `customer_id`. For calls without auth, do not store memory for now.

## What exists

[internal/memory](../../acceleration/internal/memory) is the contract — recall and remember —
and [mem0](../../acceleration/internal/memory/mem0) is the one provider. With `MEM0_API_KEY`
set, an agent recalls what it knows about the customer when it joins and prepends it to its
instructions, then hands each finished exchange over to be learned from.

Scoping is as asked: `app_id` is the deployment and `user_id` is the customer, so two
deployments sharing one mem0 account do not read each other's memories. Without an app id
nothing is stored, which is the "calls without auth" case.

## Both directions stay off the conversation's path

- **Recall is bounded, and failing means starting empty.** An agent that cannot reach mem0
  starts the call knowing nothing rather than not taking the call. Memory improves a
  conversation; it is not a precondition for one.
- **Remembering is queued and dropped under backpressure.** The same rule as turns and
  transcripts: losing a memory costs the next call some context, while blocking costs this
  caller a silence.

Every call is recorded as a `requests` row with modality `memory`, so what memory costs shows
up in the same statistics as everything else. It is recorded but not routed: there is one store,
so the provider and route paths do not serve the modality while the statistics paths do.

## Not done

Nothing outstanding. Only mem0 is implemented, which is what was asked; the contract is what a
second provider would implement.
