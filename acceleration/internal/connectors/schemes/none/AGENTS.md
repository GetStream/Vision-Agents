# internal/connectors/schemes/none

The `none` scheme: a connector that needs no credential, such as a public MCP server. The connection still exists, so it is bound, listed and deleted like any other; its stored credentials are an empty object. It replaces the prototype's `AuthNone` path (`internal/connectors/runtime.go:188-189` on `codex/connector-support` at `cf62af0d`).

## Flow

```
Begin      -> Done
Complete   Supplied must be empty -> StoredCredentials{none, v1, {}}, empty AccountInfo
Retrieve   -> AccessCredential{none, no expiry, no secret}, stored as it is
Wrap       -> base, unchanged, whatever credential it is handed
Classify   oauth2code.Classify, then a bare 401 -> InvalidGrant
Revoke     -> nil: the provider holds nothing for this connection
```

## Rules

- **Holds nothing, applies nothing.** The access credential has no secret and `Wrap` returns `base`, so a request leaves exactly as the tool source built it. Check: `go test ./internal/connectors/schemes/none -run TestNoneSchemeContract` (the subject is `Anonymous`, which makes the contract assert both).
- **A supplied value is refused, not dropped.** Whoever sent one expected it to be used. Check: `go test ./internal/connectors/schemes/none -run TestNoneSuite/TestASuppliedValueIsRefusedRatherThanDropped`.
- **Revoke is nil because there is nothing to revoke**, not because something was revoked. Check: `SchemeContract`'s `TestRevokeSaysWhatItDid`.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/schemes/none`. `TestNoneSchemeContract` runs `core/contracttest.SchemeContract` against a local TLS server that answers anyone; `NoneSuite` covers what is this scheme's own. No mocks (`.claude/skills/go-testing/SKILL.md`).
