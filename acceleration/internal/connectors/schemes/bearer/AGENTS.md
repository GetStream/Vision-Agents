# internal/connectors/schemes/bearer

The `bearer` scheme: a static token the developer supplies (a bot token, a personal access token), sent as `Authorization: Bearer <token>` (RFC 6750 §2.1) on every request. It replaces the prototype's `AuthBearer` path (`internal/connectors/runtime.go:190-195` on `codex/connector-support` at `cf62af0d`). A token an OAuth consent issued is `oauth2_code`'s, not this one's.

## Flow

```
Begin      -> Done
Complete   Supplied{token}: any other key refused; the token an RFC 6750 b64token
           -> StoredCredentials{bearer, v1, {token}}, empty AccountInfo
Retrieve   -> AccessCredential{bearer, no expiry, {token}}, stored as it is
Wrap       Authorization: Bearer <token> on a clone of each request; a credential it did not issue
           fails every request
Classify   oauth2code.Classify, then a bare 401 -> InvalidGrant
Revoke     -> ErrNotRevocable: nothing sent, the token lives on at the provider
```

## Rules

- **RFC 6750 §2.1 on the wire.** `credentials = "Bearer" 1*SP b64token`: the prefix is `Bearer ` with one space, and a token outside b64token is refused at `Complete`, not at the first call. Check: `go test ./internal/connectors/schemes/bearer -run 'TestBearerSuite/TestATokenThatIsNotAB64TokenIsRefused|TestBearerSchemeContract'`.
- **Never the URL.** RFC 6750 §2.3: the URI method «SHOULD NOT be used». Check: `SchemeContract` looks for the token in every URL Wrap sent.
- **Errors name what is wrong, never the value.** Check: `go test ./internal/connectors/schemes/bearer -run TestBearerSchemeContract`.
- **Revoke does not pretend.** RFC 7009 revokes what an authorization server issued to this client, and nothing says one did; `Revoke` returns `ErrNotRevocable` and the token still works at the provider. Check: `SchemeContract`'s `TestRevokeSaysWhatItDid`.
- **Every hardcoded value cites its source** beside it: an RFC section or a line of the prototype at `cf62af0d`.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/schemes/bearer`. `TestBearerSchemeContract` runs `core/contracttest.SchemeContract` against a local TLS server that answers 200 only to `Bearer <token>` exactly; `BearerSuite` covers what is this scheme's own. No mocks (`.claude/skills/go-testing/SKILL.md`).
