# internal/webhook

Standard Webhooks for secrets the router makes: `NewSecret` (a `whsec_` secret, 32 random bytes), `Sign` (the `v1,` HMAC-SHA256 of `id.timestamp.body`) and `Verify` (`webhook-id`, `webhook-timestamp` within 5 minutes, any one of the `webhook-signature` list; else `ErrBadSignature`). `Event` and `MaxEventBytes` are one MCP Events delivery and its 256 KiB cap.

Callers: `internal/eventforward` signs forwards to event destinations; `internal/mcpevents` makes subscription secrets and verifies MCP Events deliveries.

A provider app's inbound Standard Webhooks are checked by `internal/connectors/verifiers/standardwebhooks`, a `core.Verifier` whose age limit comes from the connector manifest.
