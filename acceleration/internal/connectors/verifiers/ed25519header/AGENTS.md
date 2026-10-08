# internal/connectors/verifiers/ed25519header

The `ed25519` verifier (`core.Verifier`): an Ed25519 signature ([RFC 8032](https://www.rfc-editor.org/rfc/rfc8032)) of the raw body, and optionally of a timestamp header, sent in base64 in a request header and checked with the provider's public key. Telnyx signs its webhooks this way ([Telnyx, receiving webhooks](https://developers.telnyx.com/docs/messaging/messages/receiving-webhooks), opened October 8, 2026). The manifest's `channel.verifier` block gives `header`, `signed`, `timestamp_header` and `max_age`, as `hmac_header`'s; the algorithm and the base64 are fixed (`core/channel.go`). It is registered in `Registry.Verifiers` under `ed25519` by `cmd/router` (`newConnectorRegistry`), and `internal/api/connector_events.go` calls it. `providers/telnyx.yaml` is the built-in that names it.

## Flow

```
Verify(r, body, m, secret)
  m.channel.verifier.kind is not ed25519               -> error
  secret not base64, or not a 32-byte public key        -> ErrUnsigned
  header not base64, or not a 64-byte signature          -> ErrUnsigned
  timestamp_header set and not Unix seconds              -> ErrUnsigned
  |now - timestamp| > max_age                            -> ErrStale (past and future)
  ed25519.Verify(key, signed with {timestamp} and {body}, signature), else -> ErrUnsigned
  m.channel.Read(m.id, body)
    error (a body the block does not describe)           -> zero VerifiedEvent, no error
    -> VerifiedEvent{Challenge, Signals, Messages}
```

## Rules

- **Nothing is read before the signature checks.** `Read` runs on the body only after the signature verified, so an unsigned body is never parsed. Check: `go test -run 'TestVerifierSuite/(TestAnUnsigned|TestASignatureBy|TestABodyChanged)' ./internal/connectors/verifiers/ed25519header`.
- **The timestamp is signed and its age is checked both ways**, as hmacheader checks it. Telnyx: «reject webhooks where `telnyx-timestamp` is more than 5 minutes old». Check: `go test -run 'TestVerifierSuite/TestARequestSigned|TestATimestamp' ./internal/connectors/verifiers/ed25519header`.
- **The secret is a public key**, the provider app's on T38's route (`api.ProviderApp`), as the customer put it: base64 of 32 bytes (RFC 8032, section 5.1.5). Anything else takes nothing. Being public, it needs no constant-time comparison; the signature check is `crypto/ed25519`'s.
- **Errors say nothing a forger can use.** `ErrUnsigned` covers a missing, malformed or wrong signature and an unusable key alike, and no error or log carries a signature or the key.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/verifiers/ed25519header`. `VerifierSuite` signs synthetic events (`testdata/telnyx.*.json`, shaped as Telnyx's example) with a key it generates and runs them through the built-in `providers/telnyx.yaml`. No mocks (`.claude/skills/go-testing/SKILL.md`).
