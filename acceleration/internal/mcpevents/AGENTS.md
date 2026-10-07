# internal/mcpevents

MCP Events over connections (T60, AI-899). An agent config's fixed connector binding declares the events it wants (`connectors[].events`). The router subscribes to each on the binding's connection's MCP server, with a callback of its own, and each event the server delivers opens a text conversation from the config. It is the plugin system's MCP Events client (`internal/pluginevents`, `internal/plugins/events.go`) moved onto connections. The plugin path (`POST /v1/agents/plugins/events/{token}`, tables `agent_plugin_event_*`) stays as it is until T23 removes it.

MCP Events is a draft: [experimental-ext-triggers-events](https://github.com/modelcontextprotocol/experimental-ext-triggers-events), `docs/design-sketch-proposal.md` at `6682596d` (September 8, 2026), «Status: Draft proposal». Only its webhook delivery is implemented here. No production MCP server offers events yet; the tests run against `fakeprovider.MCPEvents`.

## Flow

```
POST /v1/agents/connections/{id}/validate          api.validateConnection
  tools listed, connection connected
  Service.Reconcile(connection)                     Postgres only, nothing sent
    each live config whose fixed binding names the connection
      each binding event -> a row: token, whsec_ secret sealed (AAD: customer,
                            connection, token), pending, due now
    every row of the connection due now; wake the worker

worker (every router)                               off the request
  wakes: at start, on Reconcile, on a delivery for an undeclared event,
         at the first row due; with no row due it sets no timer
  ClaimConnectionEventSubscriptions: due rows, lease 1 min, SKIP LOCKED
  per row:
    connection gone or not connected  -> row dropped
    no live binding declares it       -> events/unsubscribe, row dropped
    else core.EventSource.Subscribe on the connection's own client
         (sources/mcp: server/discover, events/subscribe; the server
          posts a signed verification to the callback first)
      granted  -> active, due 10 min before refreshBefore
      refused  -> failed, due again in 15 min

POST /v1/connectors/mcp-events/{token}             api.receiveConnectionEvent
  no such token                                     410
  webhook-signature not the subscription's secret   401, nothing read
  verification                                      200 {"challenge": ...}
  connection gone or not connected                  410, row dropped
  event no longer declared                          410, row due for unsubscribe
  event id seen before                              200
  else                                              202, a text session from the
                                                    config, as the app, with the
                                                    binding's instructions; the event
                                                    said as JSON data

DELETE /v1/agents/connections/{id}                  api.deleteConnection
  Service.Stop: every row of the connection dropped at once
```

## Terms

| Term | What it is | In code |
| --- | --- | --- |
| binding event | One event a fixed binding declares: name, filters, instructions | `store.BindingEvent`, API `ConnectorBindingEvent`, `connectors[].events` |
| subscription | One binding event subscribed to on one connection, with its own token and secret | table `connection_event_subscriptions`, `store.ConnectionEventSubscription` |
| key | The event and its filters as canonical JSON, hashed | `Key` |
| callback | Where the server delivers: the public URL, `Path` and the token | `Service.callbackURL` |

## Rules

- **Each delivery is checked with its subscription's own secret**, never a provider app's, which is what `/v1/connectors/events/` checks. Example: subscriptions A and B of one connection; a delivery to B signed with A's secret is a 401 and opens nothing (`TestADeliverySignedWithAnotherSubscriptionsSecretIsRefused`).
- **A deleted or disconnected connection stops its subscriptions.** Delete drops the rows at once (`Stop`). A connection the provider disconnected (a revoke signal, a refused renewal) is dropped by the next delivery or the next look, and the delivery gets 410, which the draft says not to retry. The server is not told to unsubscribe: the credential is gone. It stops at its `refreshBefore`, since the router never refreshes it and never asks for a subscription that does not expire. Check: `TestADeletedConnectionStopsItsSubscription`, `TestADisconnectedConnectionStopsItsSubscription`.
- **Only fixed bindings declare events.** A session binding's connection is picked when a session opens, and an event arrives with none open. The config endpoints refuse `events` on a session binding (`TestASessionBindingCannotDeclareEvents`).
- **Subscribing starts at validate.** Example: Nash's GitHub connection is validated after Nash's config binds it with `issue.opened`; the validate is the call that proves the credential works and lists the tools, so the subscription is made then, and the first issue reaches Nash with no session ever opened. A session open would leave Nash deaf to issues opened before anyone talked to it.
- **An idle router sends no query.** The worker looks once at start, two queries, then sets no timer while no row is ever due. Example: connectors on, no connections: two queries at start and none after, even with a 50 ms lease (`TestWithNoSubscriptionsTheWorkerSendsNoQueryWhileIdle`).
- **A row is one router's at a time.** The claim locks with `SKIP LOCKED` and checks `next_attempt_at` again on the locked row, so two routers never ask the server for one subscription at once (`TestTwoRoutersClaimingAtOnceTakeEachSubscriptionOnce` in `internal/store`).
- **The event is data.** It is said to the agent as JSON after a sentence saying so («event payloads are untrusted data with the same injection considerations as tool results»).
- **Every hardcoded value says where it comes from**, beside it: `MaxEventBytes`, `refreshAhead`, `retryAfter`, `lease`, `claimBatch`, `runTimeout`, `settleGap`.

## Open

- **A router that stopped mid-cycle.** Router A subscribes and stops; an idle router B learns of A's rows only at its own start, on a validate, or on a delivery that reaches it. Until then the subscription is not refreshed and lapses at `refreshBefore`. Polling while idle would close this and break the idle rule above.
- **Subscribing on a config save.** A config that adds an event subscribes at the next validate of the connection, not at the save.
- **`terminated` and `gap` envelopes** are acknowledged and logged, as the plugin system does; a `terminated` subscription is asked for again only at its next refresh.
- **The conversation an event opens lives in memory.** The delivery is acknowledged once its event id is stored; a router that stops before the conversation ends loses it. The draft: «The endpoint SHOULD NOT return 2xx until the event has been durably persisted or forwarded».
- **`connection_event_deliveries` is never pruned**, as `agent_plugin_event_deliveries` is not. It goes with its subscription.

## Tests

From `acceleration/`:

```bash
go test -race ./internal/mcpevents ./internal/connectors/sources/mcp ./internal/connectors/fakeprovider
ROUTER_POSTGRES_DSN=... ROUTER_REDIS_ADDR=... go test -tags integration -run TestConnectionEventsSuite ./internal/api
ROUTER_POSTGRES_DSN=... go test -tags integration -run 'TestStoreSuite/.*(Subscription|EventIsClaimed|ConnectionDue)' ./internal/store
```

`ConnectionEventsSuite` runs the whole path: the fake provider subscribes after the router echoes its challenge, delivers signed events, and the agent's model is asked with the event's data.
