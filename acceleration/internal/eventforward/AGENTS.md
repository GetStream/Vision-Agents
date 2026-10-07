# internal/eventforward

Raw event forwarding (T46, AI-875). A customer's provider app, such as its Slack app, delivers events to the router. The router verifies each delivery and acts on it. Then this package forwards the delivery to the customer's own URLs, its event destinations. The design is «Integration modes: full platform, customizations, pass-through» in `docs/connectors/channels.md` on `connectors/planning`: mode B forwards what the router does not handle, mode C forwards everything.

## Flow

```
Slack -> POST /v1/connectors/events/{connector}/{app}   api.receiveProviderAppEvent
  verify with the app's signing secret                   401, nothing forwarded
  a URL handshake                                        answered, never forwarded
  signals -> Resolver.Revoke; messages -> Bridge.Deliver
  handled = a signal, or a message an agent answers
  Forwarder.Forward                         on the request, Postgres only
    QueueEventDeliveries: one row per destination that takes it
      forward all: every delivery; forward unhandled: only when !handled
  200
worker (every router)                       off the request
  ClaimEventDeliveries: due rows, lease 1 min, SKIP LOCKED
  POST url: raw body, Content-Type, the verifier's headers,
            webhook-id, webhook-timestamp, webhook-signature
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
- **`webhook-id` is a digest of the body.** The same id on every attempt, and for the provider's own retry of the same body. A retry that arrives while the first forward is pending is not queued twice.
- **Every destination URL goes through egress.** `CheckURL` at create (`egress.ValidatePublicHTTPSURL`), and the worker's client is `egress.NewClient`. Redirects are never followed.
- **Only a provider app's route forwards.** A connector's own route is the operator's app, whose events are no one customer's.
- **The channel bridge does not change by mode.** A message is written to its thread channel whenever one agent config binds the connection, in either mode. Mode C is a connection no agent config binds plus a destination of `all`.

## Open

- **Interactivity on a managed app.** An app the router creates (`slackapps`, T54) has no `settings.interactivity.request_url`, and its bot events do not list `reaction_added`. A customer's own app sets them at Slack. Until T54 adds them, a managed app sends no button click to forward.
- **No delivery log.** A forward given up is logged only. The spec recommends telling the customer by other means and disabling a destination that fails for days.
- **No jitter and no Retry-After.** The spec recommends both.
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
