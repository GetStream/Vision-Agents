# internal/connectors/schemes/apikey

The `api_key` scheme: a static key the developer supplies, sent in a header the developer names (`X-Api-Key`, `X-Shopify-Access-Token`), on every request. One implementation for every provider; the prototype's `AuthAPIKey` path (`internal/connectors/runtime.go:196-201` on `codex/connector-support` at `cf62af0d`) is what it replaces.

## Flow

```
Begin      -> Done
Complete   Supplied{api_key, header}: any other key refused; header canonicalized, an RFC 9110 token,
           not on the forbidden list; key non-empty, a valid field value with no edge whitespace
           -> StoredCredentials{api_key, v1, {header, key}}, empty AccountInfo
Retrieve   -> AccessCredential{api_key, no expiry, {header, key}}, stored as it is
Wrap       header: key on a clone of each request; a credential it did not issue fails every request
Export     -> ExportedCredential{header, key, no client}: the key is the app's own; a credential it did not issue is refused
Classify   oauth2code.ClassifyStatic: oauth2code.Classify, then a bare 401 -> InvalidGrant
Revoke     -> ErrNotRevocable: nothing sent, the key lives on at the provider
Static     core.Static: a rejected key is replaced with PUT .../credentials, never consented to
```

## Rules

- **The header is the connection's, supplied with the key.** `core.ResolvedManifest` has no field for it, so `Complete` reads `Supplied["header"]` and seals it beside the key. A manifest field for it is a core change (open in the PR that added this package). Check: `go test ./internal/connectors/schemes/apikey -run TestAPIKeySuite/TestTheKeyGoesInTheHeaderTheDeveloperNamedInAnyCase`.
- **Never a header the router owns or a proxy reads.** `forbidden` is the prototype's list (`internal/api/connectors.go:1050-1057` at `cf62af0d`) plus the connection and framing fields (`Keep-Alive`, `TE`, `Upgrade`, `Trailer`, `Expect`) that net/http drops, refuses, or fails the request on with the key in the error, each entry's reason beside it, and `Wrap` checks it again on what it unseals. Check: `go test ./internal/connectors/schemes/apikey -run TestAPIKeySuite/TestAHeaderTheRouterOwnsOrAProxyReadsIsRefused`.
- **Never the URL.** The key goes in a header only; RFC 6750 §2.3 says a credential in the URI «SHOULD NOT be used», and the architecture doc keeps query auth forbidden («Axes where providers differ», row 4). Check: `SchemeContract` looks for the key in every URL Wrap sent.
- **Errors name what is wrong, never the value.** Neither the key nor the header appears in an error or a log. Check: `go test ./internal/connectors/schemes/apikey -run TestAPIKeySchemeContract`.
- **Revoke does not pretend.** There is no endpoint to revoke a static key at, so `Revoke` returns `ErrNotRevocable` and the key still works at the provider; whoever deletes the connection is told to revoke it there. Check: `SchemeContract`'s `TestRevokeSaysWhatItDid`.
- **Every hardcoded value cites its source** beside it: an RFC section or a line of the prototype at `cf62af0d`.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/schemes/apikey`. `TestAPIKeySchemeContract` runs `core/contracttest.SchemeContract` against a local TLS server that answers 200 only to the right header; `APIKeySuite` covers what is this scheme's own. No mocks (`.claude/skills/go-testing/SKILL.md`).
