# internal/eventforward

Raw event forwarding (T46, AI-875). A customer's provider app, such as its Slack app, delivers events to the router. The router verifies each delivery and acts on it. Then this package forwards the delivery to the customer's own URLs, its event destinations. The design is «Integration modes: full platform, customizations, pass-through» in `docs/connectors/channels.md` on `connectors/planning`: mode B forwards what the router does not handle, mode C forwards everything.

## Flow

```
Slack -> POST /v1/connectors/events/{connector}/{app}   api.receiveProviderAppEvent
  verify with the app's signing secret                   401, nothing forwarded
  a URL handshake                                        answered, never forwarded
  signals -> Resolver.Revoke; messages -> Bridge.Deliver
  handled = a signal, or a message an agent answers
  ProviderEvent: webhook-id from channel.event_id (Slack: event_id,
    trigger_id), else the body; provider headers verify until
    their timestamp + channel.verifier.max_age (Slack: 5 min)
  Forwarder.Forward                         on the request, Postgres only
    QueueEventDeliveries: one row per destination that takes it
      forward all: every delivery; forward unhandled: only when !handled
  200
Bridge.write fails after the 200            off the request
  Forwarder.ForwardUnanswered: forward unhandled destinations only, once
worker (every router)                       off the request
  wakes: at start, on Forward, on a finished send, at the first queued
    row due (at least 1 s), and at least once a lease (1 min)
  ClaimEventDeliveries: due rows, lease 1 min, SKIP LOCKED, due checked
    again on the locked row, at most 2 in flight per destination per router
  POST url: raw body, Content-Type, the verifier's headers (only until
            they stop verifying), webhook-id, webhook-timestamp,
            webhook-signature
  2xx: done   5xx, 429, no answer: again after 5 s, 5 min, 30 min, 2 h
  3xx, 4xx: dropped                 out of waits: dropped and logged
```

## Terms

| Term | What it is | In code |
| --- | --- | --- |
| event destination | One customer URL for one connector, with its own signing secret. At most 3 per connector | table `connector_event_destinations`, `store.EventDestination` |
| forward mode | `unhandled` or `all`: which deliveries a destination takes | `store.ForwardUnhandled`, `store.ForwardAll` |
| handled | The router acted on the delivery: a signal, or a message an agent of the customer answers | `eventforward.Event.Handled` |
| forward | One delivery queued for one destination, until it is sent or given up | table `connector_event_deliveries`, `store.EventDelivery` |

## Rules

- **The ack never waits on a customer.** `Forward` only writes rows. The worker sends them. Example: a destination that takes 15 s to answer does not delay Slack's 200, which Slack needs within 3 s.
- **The body and the provider's headers are sent as they came.** Only `Content-Type` and the headers the connector's `channel.verifier` names are copied (`ProviderHeaders`). Example: Slack Bolt verifies a forwarded `block_actions` with the app's own signing secret, from `X-Slack-Signature` and `X-Slack-Request-Timestamp`. Proxy headers and `X-Slack-Retry-Num` are not copied.
- **Each destination signs with its own secret.** Never with a deployment secret. The secret is sealed with the customer, the connector and the destination id as AAD (`secretAAD`). It is shown once, on create or rotation.
- **A rotation signs with both secrets for 24 hours.** `webhook-signature` then holds two `v1,` signatures. A receiver with either secret verifies.
- **`webhook-id` is the provider's event id.** The manifest's `channel.event_id` names it (`core.ChannelRule.DeliveryEventID`): Slack's `event_id` on an event, `trigger_id` on an interaction. A body with no such id is keyed by a digest of the body. The same id on every attempt, and for the provider's own retry. Example: Slack delivers `Ev0000ONE` again with another body; it is one forward while the first is pending, and two Slack events with the same bytes but two `event_id`s are two forwards. Either way the id is hashed, since `trigger_id` holds dots.
- **One destination cannot take every send slot.** A router runs 16 sends at once, at most 2 to one destination (`perDestination`). Example: one customer's URL never answers; it holds 2 slots for 15 s each, and another customer's forward goes out at once, not after 15 s (`TestADestinationThatNeverAnswersDoesNotHoldUpAnother`).
- **The provider's signature headers go only while they verify.** Each forward keeps its provider timestamp plus `channel.verifier.max_age` (`provider_headers_until`). An attempt after that carries `Content-Type` alone of the provider's headers. Example: Slack Bolt refuses a request whose `X-Slack-Request-Timestamp` is more than 5 minutes old (`requestTimestampMaxDeltaMin = 5`, bolt-js `src/receivers/verify-request.ts`), so the retries after 5 min, 30 min and 2 h carry no Slack signature, and the receiver verifies `webhook-signature`. A stale Slack signature is never presented, for a receiver that skips the timestamp check to take as Slack's.
- **A message the bridge could not write is unhandled after all.** `Bridge.Deliver` counts a message as answered before it writes it into its thread channel, after the ack. When that write fails, the bridge calls the route's `unanswered`, which queues the delivery for the `unhandled` destinations alone (`ForwardUnanswered`); the `all` destinations already have it.
- **An idle router looks once a lease, not once a second.** The worker looks at start, on `Forward`, on a finished send, at the first queued row due, and at least once a lease (1 min) whatever is queued. Example: a router with connectors on and no destination sends Postgres two queries at start and two a minute after (`TestAForwarderWithNothingQueuedSendsNoQuery`); router A queues a forward and is scaled down before it sends it, and idle router B sends it within about two minutes, its next look plus A's lease (`TestAnIdleRouterSendsAForwardAnotherRouterLeftQueued`, `TestARouterWaitingOnALaterRetryStillSendsAForwardAnotherRouterLeft`).
- **A claimed row is one router's.** The claim ranks due rows from the statement's start, then locks them and checks `next_attempt_at` again on the locked row: a row another router leased in between is skipped (`TestTwoRoutersClaimingAtOnceTakeEachDeliveryOnce`).
- **Every destination URL goes through egress.** `CheckURL` at create (`egress.ValidatePublicHTTPSURL`), and the worker's client is `egress.NewClient`. Redirects are never followed.
- **Only a provider app's route forwards.** A connector's own route is the operator's app, whose events are no one customer's.
- **The channel bridge does not change by mode.** A message is written to its thread channel whenever one agent config binds the connection, in either mode. Mode C is a connection no agent config binds plus a destination of `all`.

## Open

- **Interactivity on a managed app.** An app the router creates (`slackapps`, T54) has no `settings.interactivity.request_url`, and its bot events do not list `reaction_added`. A customer's own app sets them at Slack. Until T54 adds them, a managed app sends no button click to forward.
- **No delivery log.** A forward given up is logged only. The spec recommends telling the customer by other means and disabling a destination that fails for days.
- **No jitter and no Retry-After.** The spec recommends both.
- **A failing destination keeps the full timeout.** Each attempt to a URL that never answers waits the whole 15 s, so its own backlog drains at 2 forwards per 15 s on each router. A shorter timeout after a failed attempt would drain it faster.
- **The cap is per router.** With N routers a destination can hold 2 × N sends. Each router's own slots are what it protects.
- **A forward a stopped router left waits up to about two minutes**, a lease for its claim to run out and up to a lease for another router's next look. A shorter look costs more idle queries.
- **An interaction's ack body.** The router answers 200 with no body. A `view_submission` that needs `response_action` in the ack cannot get it from a forward.

## Tests

From `acceleration/`:

```bash
go test ./internal/eventforward
ROUTER_POSTGRES_DSN=... go test -tags integration ./internal/eventforward
ROUTER_POSTGRES_DSN=... go test -tags integration -run 'TestStoreSuite/.*(EventDestination|Deliver|Rotation)' ./internal/store
ROUTER_POSTGRES_DSN=... ROUTER_REDIS_ADDR=... \
  go test -tags integration -run 'TestEventForwardingSuite|TestConnectorEventDestinationsSuite' ./internal/api
```

`EventForwardingSuite` runs the whole path: the fake Slack delivers, the router acks, and local TLS servers stand for the customer's URLs.
