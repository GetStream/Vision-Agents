# internal/webhook

Standard Webhooks signing for deliveries the router sends and receives, and the MCP Events delivery body. No state, no I/O.

- `NewSecret`: a fresh `whsec_` secret (32 random bytes, base64).
- `Sign`: the `v1,` HMAC-SHA256 of `id.timestamp.body`, as the `webhook-signature` header carries it.
- `Verify`: checks `webhook-id`, `webhook-timestamp` (5 min either way) and `webhook-signature` (any one of a space-separated list) against a secret; `ErrBadSignature` otherwise.
- `Event`, `MaxEventBytes`: one MCP Events delivery and the most it may hold (256 KiB).

Used by `internal/eventforward` (signs what it forwards to a customer's event destinations), `internal/mcpevents` (connector MCP Events subscriptions) and `internal/pluginevents` (the plugin system's MCP Events, until it is removed).

`internal/connectors/verifiers/standardwebhooks` checks the same signature on a provider app's inbound webhooks. It is a `core.Verifier` that reads the age limit from the connector manifest; this package is for secrets the router itself makes.
