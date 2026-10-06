# internal/connectors/verifiers/hmacheader

The `hmac_header` verifier (`core.Verifier`): an HMAC of the raw body, and optionally of a signed timestamp, sent in a request header. Every parameter is the manifest's `channel.verifier` block (`core/channel.go`): header, algorithm, encoding, prefix, the `signed` template, `timestamp_header` and `max_age`. Slack's request signing is one (`providers/slack.yaml`). It is registered in `Registry.Verifiers` under `hmac_header` by `cmd/router` (`newConnectorRegistry`), and `internal/api/connector_events.go` calls it.

## Flow

```
Verify(r, body, m, secret)
  m.channel.verifier.kind is not hmac_header       -> error
  secret is empty                                  -> ErrUnsigned (an HMAC anybody can compute)
  header without prefix, or not encoding           -> ErrUnsigned
  timestamp_header set: not Unix seconds           -> ErrUnsigned
                        |now - timestamp| > max_age -> ErrStale (past and future)
  HMAC(algorithm, secret, signed with {body}, {timestamp})
  hmac.Equal with the header's digest, else        -> ErrUnsigned
  m.channel.Read(m.id, body)
    error (a body the block does not describe)     -> zero VerifiedEvent, no error
    -> VerifiedEvent{Challenge, Signals, Messages}
```

## Rules

- **Nothing is read before the signature checks.** `Read` runs on the body only after `hmac.Equal`, so an unsigned body is never parsed. Check: `go test -run 'TestVerifierSuite/(TestAnUnsigned|TestASignatureWith|TestABodyChanged)' ./internal/connectors/verifiers/hmacheader`.
- **Digests are compared in constant time** with `hmac.Equal`, as Slack's page asks («use an hmac compare function instead of directly comparing the signatures for equality», https://docs.slack.dev/authentication/verifying-requests-from-slack, opened October 6, 2026). A timing difference is not something a test can observe reliably, so `GuardSuite` reads `verifier.go`'s source, as core's guard tests do: one `hmac.Equal`, and no `==`, `!=` or `bytes.Equal` on a conversion. Check: `go test -run TestGuardSuite ./internal/connectors/verifiers/hmacheader`.
- **The age is checked both ways.** Slack refuses a timestamp «more than five minutes from local time», which is `abs(now - timestamp) > max_age`. Check: `go test -run 'TestVerifierSuite/TestARequestSigned' ./internal/connectors/verifiers/hmacheader`.
- **The timestamp is decimal Unix seconds**, which is what Slack sends. A provider that writes it another way needs a parameter in `core.VerifierRule` first, not a guess here.
- **The body is signed as it is.** The `signed` template is filled with `strings.Replacer`, which reads only the template, so a body holding the text `{timestamp}` is not rewritten. Check: `go test -run TestVerifierSuite/TestABodyHoldingAPlaceholder ./internal/connectors/verifiers/hmacheader`.
- **The secret comes from the caller.** The verifier never looks one up: the route a request came in on decides whose secret it is (`core.Verifier`), the operator's on a connector's route and a provider app's on T38's route.
- **Errors say nothing a forger can use.** `ErrUnsigned` covers a missing, malformed or wrong signature and a missing secret alike, and no error or log carries a signature, a digest or the secret.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/verifiers/hmacheader`. `VerifierSuite` signs the core's synthetic Slack events (`core/testdata/recorded/slack_bot.*.json`) the way Slack does and runs them through the built-in `providers/slack.yaml` and the core's `slack_bot` fixture. No mocks (`.claude/skills/go-testing/SKILL.md`).
