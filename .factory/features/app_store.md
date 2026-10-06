# The app config store

Landed in [099de30b](https://github.com/GetStream/Vision-Agents/commit/099de30b).

## Asked for

Every request shares one access pattern: look up the API key, the app behind it and that
app's organization, then whatever that organization has configured. That was a join on
Postgres before anything the caller asked for had begun. Put it behind a Redis client-side
cache, measure the overhead first so the saving is a number rather than a claim, and keep
the two in step when a key is revoked or a config is edited.

## What exists

[internal/appconfig](../../acceleration/internal/appconfig) reads the configuration a
request is measured against: API keys with their app and organization, policies,
organization membership, agent configs, skills, router configs, and voices with their
provider bindings.

Two tiers over one connection. `rueidisaside` holds the value in Redis, and the
client-side cache rueidis keeps alongside it answers without a round trip at all. Redis
invalidates that local copy itself when the key is deleted, so every replica forgets a
revoked key at once rather than each on its own timer. Postgres stays the only writer and
the only truth: every write goes there first and drops the keys it changed, and the
hour-long TTL is only the backstop for a deletion that did not land.

The authenticator, `policy.Enforcer`, `api.Server`, `session.Manager` and the voice
resolver and service all read through it. A deployment with no Redis address, or one whose
Redis is unreachable, reads Postgres and nothing fails.

`TouchAPIKey` is throttled per process as well. The store already refused to write
`last_used_at` more than once a minute, but it took a query to find that out, which on a
busy key was a second round trip per request for a column nothing reads in anger.

## Tracing, which is how it was measured

[internal/tracing](../../acceleration/internal/tracing) is `go.opentelemetry.io/otel` with
an OTLP HTTP exporter that only installs itself when `OTEL_EXPORTER_OTLP_ENDPOINT` is set;
otherwise the global no-op provider stands. `otelhttp` wraps the handler and renames the
span to the chi route pattern, `bunotel` puts a span on every query, and auth, quota,
policy admission and LLM creation have spans of their own.

`internal/api/performance_test.go` reads the per-request query counts back out of those
spans: it groups ended spans by trace, keeps the traces whose root span is the route being
measured, and counts them by instrumentation scope. That is what makes the `postgres` and
`redis` columns below reads the request itself made, rather than background work that
happened at the same time.

```
case                                     mean        p50        p99   postgres   redis
authenticate only                       517µs     →141µs     →225µs    1.00→0.00  0.00
create a session                     16.044ms   →16.946ms  →23.205ms   2.00→1.00  2.00
create a session from an agent config 16.885ms  →20.663ms →180.812ms   4.01→2.00  2.00
ask a question                       44.565ms    →42.43ms  →59.944ms   1.00→0.00  0.00
ask a question, four at a time      142.977ms →111.535ms →197.464ms    1.00→0.00  0.00
```

Authentication is 3.7x faster and makes no query at all; against the state before any of
this work, which also wrote `last_used_at` per request, it is 5.0x. The p99 of four
concurrent callers falls from 806ms to 197ms, because they no longer contend for
connections to re-read the same key.

The per-message path was never config-bound, so its mean barely moves. **Agent configs are
not read per message**: they are loaded once at session create and held in the session.
That is why the cache's win is at the door rather than in the conversation.

## What stays in Postgres

`SessionExists` on every session create, because a cached answer to "is this id taken" is
the wrong answer. `agent_plugin_connections`, because its OAuth tokens are refreshed and
written back, which is the one read on the session-create path that is not configuration.
Those are the two queries left in the table above.

## Tests

[appconfig_test.go](../../acceleration/internal/appconfig/appconfig_test.go) runs two
stores over one Postgres and one Redis, which is what two replicas are. A key revoked on
one stops working on the other; a saved policy and an organization move are seen by the
other; a renamed agent config stops answering to its old name; an edited skill is read
with its new instructions; a key cached while it was live stops working when it lapses;
and a store with no Redis still sees its own writes.

## Not done

Nothing revokes a key over HTTP yet — `RevokeAPIKey` is there for when something does, and
`router keys create` needs no counterpart because an id nothing has seen is an id nothing
has cached.

The OTLP exporter is wired but no deployment sets an endpoint, so in production the spans
are built and dropped. Metrics are not exported at all; the cache's own hit rate is
visible to rueidis and to nothing else.
