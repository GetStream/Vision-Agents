# internal/connectors/fakeprovider

One `httptest` TLS server that plays an OAuth authorization server and an MCP endpoint, so scheme, source and resolver tests run against real HTTP with no vendor account. Personalities switch how it misbehaves. It is test code: import it only from `_test.go` files.

## Use

```go
srv := fakeprovider.New(s.T(), fakeprovider.RotatingRefreshWithGrace)
```

That is the whole setup. The server closes when the test ends (`t.Cleanup`). It gives you:

- `srv.URL`: issuer and origin of every endpoint (`Path*` constants): RFC 9728 and RFC 8414 metadata, RFC 7591 registration, authorize, token, RFC 7009 revoke, and `/mcp`.
- `srv.ClientID`, `srv.ClientSecret`: a preregistered confidential client (basic or post) redirecting to `fakeprovider.RedirectURI`; `srv.AllowRedirect(uri)` adds another redirect URI. Public and other clients register through `/register`.
- `srv.Consent(authorizeURL)`: the browser. It approves and returns the callback URL with `code`, `state`, `iss` (or `error`).
- `srv.Client()`: an `*http.Client` that trusts the server's certificate. Hand it to the code under test.
- `srv.FetchClientMetadataWith(c)`: the client `ClientMetadataDocuments` fetches a client_id URL with, such as the `Client()` of the test's own TLS server that serves the document. Fetching from loopback is the testing exception CIMD §8.6 allows; production code never does it.
- `srv.Use(...)` to switch personalities mid-test, `srv.Advance(d)` to move the server's clock past expiry or grace, `srv.Hits(path)` and `srv.Refreshes()` to count what reached it, `srv.RefreshScope()` for the `scope` the last refresh carried and whether it carried one.
- `TeamID`, `UserID`, `RealmID`, `Shop`: the synthetic identities the vendor-shaped personalities report.
- `srv.SwitchAccount()`: every later consent is by another user of the same workspace (its id is returned; `UserID` keeps the first). Grants already made keep their user. For an account-switch test (T17).

**Egress.** The server listens on loopback, which `egress.NewClient` refuses by design. Do not point an egress client at it and do not add a way around that: tests here use `srv.Client()`, and tests of the egress policy keep using egress's own seams (`internal/egress/AGENTS.md`).

## Baseline

With no personality the server is strict: PKCE with `S256` only (RFC 7636 §4.4.1, §4.6; RFC 9700 §2.1.1), `iss` in every authorization response (RFC 9207 §2), `resource` checked when sent (RFC 8707 §2), a code redeemed twice revokes its grant, even after it expired (RFC 6749 §4.1.2), refresh tokens rotate on every use and a replayed one revokes the grant (RFC 9700 §4.14.2), a refresh `scope` wider than the grant gets `invalid_scope` (RFC 6749 §6, §5.2), access tokens live `AccessTTL`. The MCP endpoint is dual-era: MCP 2026-07-28 (per-request `_meta`, `server/discover`, header validation) and 2025-11-25 (`initialize`), two tools (`echo`, `fail` with `isError`) on two `tools/list` pages.

## Personalities

| Personality | Behaviour | Source | Serves |
| --- | --- | --- | --- |
| `RotatingRefreshWithGrace` | A rotated refresh token keeps working for `Grace`, then gets `invalid_grant`; the grant and the token the rotation issued stay | oauth-jsclient README («previous refresh tokens expire 24 hours after you receive a new one») | T10 grace retry |
| `NonRotatingRefresh` | A refresh answers without `refresh_token`; the old one stays valid | RFC 6749 §6 | T10 |
| `NoRefreshToken` | The exchange returns an access token alone; it expires | RFC 6749 §5.1 | T10, T12 |
| `InvalidGrant` | Every refresh gets 400 `invalid_grant`; access tokens already issued work until they expire | RFC 6749 §5.2 | T10 InvalidGrant, T12 |
| `LostResponse` | A refresh rotates, then the connection closes with no answer; combines with `RotatingRefreshWithGrace` and `NonRotatingRefresh` | RFC 9700 §4.14.2 (why a replay then fails) | T8, T10 Uncertain and the single grace retry, T12 |
| `LostResponseOnce` | `LostResponse` for the next refresh only; the ones after it answer | as `LostResponse` | T10 grace retry that succeeds |
| `Unavailable` | The token endpoint answers 503 and changes nothing | RFC 9110 §15.6.4 | T10 Transient, T11 |
| `ServerError` | A refresh rotates, then answers 500 `server_error` (Slack's 200 `internal_error` under `CommaScopes`) | RFC 9110 §15.6.1; RFC 6749 §4.1.2.1; docs.slack.dev oauth.v2.access («It's possible some aspect of the operation succeeded») | T10 Uncertain |
| `CutOffRefusal` | A refresh gets 400 whose body is cut off (Content-Length promises more, the connection closes); nothing changes | not a vendor behaviour: a transport failure after the status line | T10 a refusal is not Uncertain |
| `AccessTokenNotRevocable` | Revoking an access token gets 400 `unsupported_token_type`; refresh tokens revoke as before | RFC 7009 §2, §2.2.1 | T10 Revoke |
| `InsufficientScope` | `tools/call` without `RequiredScope` gets 403 `insufficient_scope` with the scope to ask for | RFC 6750 §3.1; MCP 2025-11-25 «Scope Challenge Handling» | T10 ScopeRequired, T27 |
| `ClaimsChallenge` | MCP requests get 401 `insufficient_claims` with `claims` until the token came from a consent passing them | Microsoft «Claims challenges, claims requests and client capabilities» | T10 ScopeRequired, T27 |
| `RateLimited` | MCP requests and refresh grants get 429 with `Retry-After: 30`; a refused refresh spends nothing | RFC 6585 §4; RFC 9110 §10.2.3 | T10 RateLimited, T28 |
| `CommaScopes` | `scope` and `user_scope` split on commas; token response, refresh included, in `oauth.v2.access` shape with `team` and `authed_user` (the user token with its own `refresh_token`); access tokens live `SlackAccessTTL` (12 h); token errors as 200 `ok:false` with Slack's names | docs.slack.dev oauth.v2.access, using-token-rotation, installing-with-oauth, node-slack-sdk web-api | T9, T10 |
| `CallbackRealmID` | `realmId` in the callback | oauth-jsclient `src/OAuthClient.js` createToken | T9 |
| `SignedCallback` | `shop`, `host`, `timestamp` and an `hmac` over the callback; `Sign` recomputes it | shopify.dev «Authorization code grant» | the `shopify.callback_hmac` hook (architecture doc, stress-test row 4) |
| `ForeignIssuer` | The callback names `ForeignIssuerURL` as `iss` | RFC 9207 §2.4 | T9 |
| `ConsentDenied` | The callback carries `error=access_denied` | RFC 6749 §4.1.2.1 | T9, T17 |
| `ClientMetadataDocuments` | Metadata advertises `client_id_metadata_document_supported`; an https `client_id` the server does not know is fetched at authorize with the client `srv.FetchClientMetadataWith` set (no redirects, 5 KB cap) and accepted only if the document names that URL as `client_id`, lists the `redirect_uri`, has a `client_name` and no shared secret; the client is public (`none`) | draft-ietf-oauth-client-id-metadata-document-02 §4, §4.1, §4.2, §5, §6, §8.7; MCP 2025-11-25 «Client ID Metadata Documents» | T9 |

At most one of the token-endpoint personalities (`tokenEndpoint` in `fakeprovider.go`) can be on; `New` and `Use` fail the test otherwise. The rest combine. The two lost-response personalities change how a refresh is delivered, not what it does, so they sit outside that set.

## Adding a personality

- **Name it by behaviour**, not by vendor, unless the behaviour exists only at that vendor. One constant in `fakeprovider.go` whose comment says what it does and where the behaviour is specified.
- **Every hardcoded value cites its source** beside it: an RFC section, a spec page, or a vendor page or SDK file you opened. A value you could not check says `unverified` in the comment.
- **Synthetic values only.** Tokens, codes, secrets and ids come from `synthetic` or `syntheticDigits`, fresh per server. Never a real credential.
- **One self-test** in `fakeprovider_test.go`, named for the behaviour, that fails when the personality is broken. Break it once by hand and watch it fail before you commit.
- Add the row above, with the subtask it serves.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/...`. Testify suite in the outside package `fakeprovider_test`, so it uses only what another package's test can. No mocks (`.claude/skills/go-testing/SKILL.md`). The vendor-shaped tests run `core`'s fixture manifests (`../core/testdata/`) over the fake's answers, so the fake and those fixtures agree.
