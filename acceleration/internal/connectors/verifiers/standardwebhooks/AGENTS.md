# internal/connectors/verifiers/standardwebhooks

The `standard_webhooks` verifier (`core.Verifier`): the [Standard Webhooks](https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md) signature, which Linq signs its webhooks with ([Linq webhooks](https://docs.linqapp.com/guides/webhooks/index.md), both opened October 8, 2026). The specification fixes the headers, the signed content and the secret format, so the manifest's `channel.verifier` block gives only `max_age` (`core/channel.go`). It is registered in `Registry.Verifiers` under `standard_webhooks` by `cmd/router` (`newConnectorRegistry`), and `internal/api/connector_events.go` calls it. `providers/linq.yaml` is the built-in that names it.

## Flow

```
Verify(r, body, m, secret)
  m.channel.verifier.kind is not standard_webhooks   -> error
  secret, whsec_ prefix optional, not base64 or empty -> ErrUnsigned (an HMAC anybody can compute)
  no Webhook-Id, Webhook-Timestamp not Unix seconds   -> ErrUnsigned
  |now - timestamp| > max_age                         -> ErrStale (past and future)
  HMAC-SHA256(secret, {id}.{timestamp}.{body})
  one v1,<base64> in Webhook-Signature matches (hmac.Equal), else -> ErrUnsigned
  m.channel.Read(m.id, body)
    error (a body the block does not describe)        -> zero VerifiedEvent, no error
    -> VerifiedEvent{Challenge, Signals, Messages}
```

## Rules

- **Nothing is read before the signature checks.** `Read` runs on the body only after a signature matched, so an unsigned body is never parsed. Check: `go test -run 'TestVerifierSuite/(TestAnUnsigned|TestASignatureWith|TestABodyChanged|TestAnIDChanged)' ./internal/connectors/verifiers/standardwebhooks`.
- **Signatures are compared in constant time** with `hmac.Equal`, as the specification asks («use a constant time comparison function»). `GuardSuite` reads `verifier.go`'s source, as hmacheader's does: one `hmac.Equal`, and no `==`, `!=` or `bytes.Equal` on a conversion. Check: `go test -run TestGuardSuite ./internal/connectors/verifiers/standardwebhooks`.
- **Any one `v1` signature may match.** The header is a space-delimited list, so a provider rotating its secret signs with both. `v1a`, the asymmetric variant, is not read: Linq names only `v1`.
- **The age is checked both ways**, as hmacheader checks it. Linq: «Reject if the timestamp is more than 5 minutes old». Check: `go test -run 'TestVerifierSuite/TestARequestSigned' ./internal/connectors/verifiers/standardwebhooks`.
- **The secret comes from the caller**, the provider app's on T38's route (`api.ProviderApp`), as the customer put it: `whsec_` and base64, or the base64 alone, as `internal/channels/linq.go` reads it.
- **Errors say nothing a forger can use.** `ErrUnsigned` covers a missing, malformed or wrong signature and a missing secret alike, and no error or log carries a signature, a digest or the secret.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/verifiers/standardwebhooks`. `VerifierSuite` signs the core's synthetic Linq events (`core/testdata/recorded/linq.*.json`) as the specification says and runs them through the built-in `providers/linq.yaml`. No mocks (`.claude/skills/go-testing/SKILL.md`).
