# Routing

Asked for three times, once per modality: [sprint 1](../sprint1.md) steps 3 and 4,
[sprint 2](../sprint2.md) steps 3 and 4, [sprint 3](../sprint3.md) steps 3 and 4. All three
asked for the same two things, so it was built once.

## Asked for

Track, per provider and model: performance, uptime, usage (audio duration or tokens), the
number of API calls and the cost. Roll the same figures up hourly and daily. Then let a
caller name a capability rather than a model:

- `en-low-latency`
- `multilingual-low-latency`
- `en-high-accuracy`
- `multilingual-high-accuracy`

Sprint 3 adds `llm-fast`.

## What exists

[internal/routing](../../acceleration/internal/routing) is the whole of it, and it knows
nothing about audio, text or tokens. `Router[P]` is generic over a provider that can be
opened, closed and named; each modality supplies that provider and a session that knows
which of its own events count as a unit of work.

| Piece                                                              | What it does                                    |
| ------------------------------------------------------------------ | ----------------------------------------------- |
| [router.yaml](../../acceleration/internal/routing/router.yaml)      | Providers, capabilities, prices and shortcuts   |
| [config.go](../../acceleration/internal/routing/config.go)          | Parses it; `ROUTER_CONFIG` replaces the built-in |
| [registry.go](../../acceleration/internal/routing/registry.go)      | Builds a provider from a `Spec`                 |
| [router.go](../../acceleration/internal/routing/router.go)          | Resolves a target, ranks candidates, fails over |
| [stats.go](../../acceleration/internal/routing/stats.go)            | Prices each row and writes it off the hot path  |

## How it decides

A target is either a concrete `provider/model`, which resolves to itself, or a shortcut,
which resolves to every provider meeting its requirements. Candidates are then ranked by
availability, then error rate, then average latency, read live from Redis.

Two decisions in that ranking are worth keeping:

- **A provider with no recent history keeps its config order.** An unmeasured provider has
  an average latency of zero, and sorting on that would put every cold provider first.
- **A Redis failure means "no information", not "unhealthy".** Health is an optimisation;
  a broken stats path must not take routing down with it.

Failover happens at session start rather than mid-request: `Select` walks the ranked list
until a provider starts, records the ones that did not, and hands back the first that did.
A provider that keeps failing falls down the ranking on its own, without a config edit.

## What it records

One row per unit of work, in `requests`, keyed by modality so a provider serving two of
them does not mix its numbers. Latency is deliberately the number the customer felt: decode
time for speech-to-text, wait for first audio for text-to-speech, wait for first token for a
completion. Uptime and latency come from the same rows as billing, so there is no separate
health probe to keep in sync.

Prices come from config, not from the provider: each model declares what this deployment is
billed and the recorder prices every row once. A model with no price reports zero rather
than a guess, which is why
[routing_test.go](../../acceleration/internal/routing/routing_test.go) asserts every
configured provider has one.

Rollups land in `stats_hourly` and `stats_daily` and are served by
`GET /v1/{modality}/stats`. `GET /v1/{modality}/providers` and
`GET /v1/{modality}/routes/{target}` answer what exists and where a target would go, both
through an `Inspector` interface, since `Router[P]` is a different type per modality and one
HTTP surface has to serve all of them.

## Not done

Nothing outstanding from the sprints. Worth knowing: `customer_id` is a trusted header, so
there is no real authentication in this version.
