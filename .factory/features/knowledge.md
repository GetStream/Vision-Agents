# Knowledge

[Sprint 9](../sprint9.md), where it is a field on an agent config: "Knowledge/RAG
(turbopuffer best practices based)".

## Asked for

Somewhere for an agent to look things up, per config, built on turbopuffer.

## What exists

[internal/knowledge](../../acceleration/internal/knowledge) is the contract and it is one
method, `Search`. How a provider chunks, indexes and ranks is its own business; what the
agent needs back is passages it can put in front of the model.
[turbopuffer](../../acceleration/internal/knowledge/turbopuffer) is the one provider, behind
`TURBOPUFFER_API_KEY`.

A config's `knowledge_namespace` is what an agent may read. A session that has one is given
a `lookup` tool; a session without one is not, so a model with nothing to search cannot try.
Filling a base is `POST /v1/agents/knowledge` or
[cmd/knowledge](../../acceleration/cmd/knowledge) from disk, both cutting documents into
passages the same way so an SDK and a directory can replace each other's work.

## Memory and knowledge are not the same thing

[Memory](memory.md) is what the agent learned about a person and is recalled once, on
joining. Knowledge is what the business wrote down and is read mid-sentence, when the caller
asks something the instructions do not answer. That difference in timing is the reason they
are separate features rather than one store with two scopes.

It also decides where each sits. Remembering is queued and dropped under backpressure;
a lookup is on the live path, because the caller is waiting on the answer that is the whole
reason the model asked. What bounds it is the store's own timeout, and a lookup that finds
nothing is answered in words — "nothing covers that, say so rather than guessing" — since a
model handed an empty result invents one.

Nothing is searched without a namespace. Searching everything would answer one customer's
caller out of another customer's handbook.

A lookup is recorded as a `requests` row with modality `knowledge` and model `search`, so
what looking things up costs shows up beside what the models cost.

## Not done

Nothing outstanding. Only turbopuffer is implemented, which is what was asked; the contract
is what a second provider would implement.
