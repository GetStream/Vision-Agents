# internal/connectors/schemes/oauth2code

The `oauth2_code` scheme: the OAuth 2.0 authorization code grant with PKCE S256 (RFC 6749 §4.1, RFC 7636), one implementation for every provider a manifest describes. `Begin` and `Complete` (part 1, AI-834) acquire the grant; `AccessCredential`, `Wrap`, `Classify` and `Revoke` (part 2, AI-836) renew it, apply it, read what a provider answered and end it. The lock, the checkpoint and the status a connection moves to are the resolver's (T12), not this package's.

## Flow

```
Begin(Ref, Manifest, RedirectURI)
  discover   manifest endpoints first; with endpoints.issuer, RFC 8414 metadata; with authorize or token
             missing (or client.from naming cimd/dcr) and no issuer, RFC 9728 metadata of endpoints.mcp, then RFC 8414.
             Each well-known URL is tried in MCP's order; one that fails (status, refused redirect, mismatched
             resource or issuer) is skipped, and only if none is usable is it an error naming every failure
  endpoints  every endpoint, pinned or discovered, passes Config.PublicEndpoint (egress) with its query removed
  PKCE       refuse a server listing methods without S256; refuse one found through RFC 9728 that lists none
  client     customer, operator (ClientLookup), cimd (Config.ClientMetadataURL), dcr (RFC 7591), in that order,
             each only if client.from names it
  URL        authorize endpoint + authorize_params + response_type, client_id, redirect_uri, state,
             code_challenge, S256, resource, scope joined by scopes.separator
  -> AuthorizeURL, State (JSON: state, verifier, redirect URI, client, issuer, token endpoint, resource,
     scopes, expiry). The core seals State; this package never stores it.

Complete(Ref, Manifest, State, Query)
  state      exactly one, equal in constant time, not expired (10 min), not spent in this process
  iss        RFC 9207: equal to the issuer, required when metadata says it is sent; checked before error
  error      AuthorizationError (access_denied and the rest)
  exchange   token endpoint, code + redirect_uri + code_verifier + resource + token_params, client auth
             none | client_secret_post | client_secret_basic; an "error" member is an error at any status
  id_token   iss, aud (this client only), azp (if present), exp checked when a capture rule reads it; the
             signature is not (OIDC Core §3.1.3.7, item 6)
  capture    ResolvedManifest.Apply on the callback and the raw token response: account id, metadata, unverified
  -> StoredCredentials{Scheme: oauth2_code, Version: 1, Payload: ref, client, endpoints, tokens, expiry,
     refresh expiry, scopes}, AccountInfo

AccessCredential(stored, m)
  due        expiry known and within refresh.margin (default 1 min, the prototype's); otherwise -> the token, stored as is
  none       no refresh token: still valid -> the token; expired -> InvalidGrant
  refresh    endpoints.refresh or the token endpoint, checkEndpoint; client secret looked up again by payload ref;
             grant_type, refresh_token, scope (granted scopes, only with scopes.send_on_refresh), resource
  classify   Classify on the answer; a 2xx without a readable token is Uncertain; an error member Classify does
             not name, or any other refusal, is Transient
  grace      Uncertain and refresh.grace > 0 and the window the first attempt opened still running -> the same
             refresh token once more (the retired one still works there); never without a grace
  -> AccessCredential, new StoredCredentials (new refresh token if one came, refresh_ttl expiry), or
     *core.OutcomeError and no StoredCredentials, with the old AccessCredential beside it while that has not
     expired; stored is never written to. A
     refresh token dying before the next refresh logs a warning

Classify(resp, body, err)
  err        no response: never written (errNotSent, failed dial) Transient, any other Uncertain. A response whose
             body was lost: status and headers decide as below, and a 1xx-3xx is Uncertain
  429        RateLimited, Retry-After (delay-seconds or HTTP-date)
  challenge  Bearer insufficient_scope -> ScopeRequired + scopes; 401 insufficient_claims -> ScopeRequired + decoded
             claims; invalid_token -> InvalidGrant. A token68 or unparseable text skips to the next comma
  body       invalid_grant, invalid_refresh_token -> InvalidGrant; temporarily_unavailable -> Transient;
             server_error, internal_error, fatal_error -> Uncertain; insufficient_scope -> ScopeRequired;
             invalid_client, unauthorized_client, unsupported_grant_type, invalid_scope -> Transient; any other
             code is a resource's own error and falls through
  status     503 Transient; other 5xx Uncertain; anything else OK

Wrap(base, AccessCredential) Authorization: Bearer on a clone of each request; no access token -> every request fails
Revoke(stored, m)           endpoints.revoke or the discovered revocation_endpoint (else ErrNoRevocationEndpoint),
                            RFC 7009 with the refresh token (else the access token); unsupported_token_type ->
                            ErrTokenTypeNotRevocable; 200 -> nil, not proof, and the access token may work until
                            it expires (§2.1 only if the server revokes access tokens)
```

## Rules

- **No provider names.** Behaviour comes from the `core.ResolvedManifest`: endpoints, `authorize_params`, `token_params`, `scopes.separator`, `client` (`client.from` among it), `capture`, `identity`. A provider that needs a branch here needs a manifest field or a hook instead (`internal/connectors/core/AGENTS.md`). No string literal in a non-test file is a provider name or a seeded connector id.
- **All outbound HTTP goes through `Config.HTTP`.** Discovery, registration, the code exchange, refresh and revocation use it and nothing else. The router passes `egress.NewClient(timeout, nil)`; tests pass the fake provider's `Client()`, because egress refuses loopback by design. Never build a client here, never replace its `Transport` or `CheckRedirect`.
- **Every authorization server endpoint passes `Config.PublicEndpoint`.** Nil means `egress.ValidatePublicHTTPSURL`. The authorize URL goes to the browser and never through `Config.HTTP`, so this is its only address check; the query is kept on the URL (RFC 6749 §3.1, §3.2) and removed only for the check. Tests that run on loopback pass a checker that lets a loopback IP literal through and nothing else; never pass one that returns nil for everything.
- **State is the core's to seal.** `BeginOutput.State` holds the PKCE verifier and, for a registered client, its secret; the core seals it into the attempt, whose one-use consumption is the replay guard across replicas. The in-process spent set refuses a replay within one process before its code reaches the provider, where a second redemption would revoke the grant (RFC 6749 §4.1.2).
- **A preregistered client's secret is never sealed.** It is looked up again at `Complete`, at refresh and at revocation, so a rotated secret lives in one place. `core.Scheme` hands `AccessCredential` and `Revoke` no `ConnectionRef`, so `Complete` keeps the one it ran for in the stored credentials; the core seals them bound to that connection. Only a client this scheme registered keeps its secret in State and StoredCredentials. The sealed client keeps the JSON name `owner` for its `Source`, so payloads stored before the `client.from` rename still read.
- **A rotated refresh token is never replayed outside a grace window.** A failure that may have taken effect is `Uncertain`, and `AccessCredential` sends the same refresh token again only when the manifest's `refresh.grace` says the provider still accepts it, once, inside the window. Anything else would be the replay RFC 9700 §4.14.2 revokes a grant for.
- **A failed `AccessCredential` or `Revoke` is a `*core.OutcomeError`** whose `Outcome` came from `Classify`, and leaves the caller's stored credentials as they were. The exceptions are `Revoke`'s two answers that no retry changes, `ErrNoRevocationEndpoint` and `ErrTokenTypeNotRevocable`. Nothing this package logs or returns names a token or a secret.
- **`private_key_jwt` is built, not offered.** `PrivateKeyJWT` (`privatekeyjwt.go`) makes the assertion (OIDC Core §9, RFC 7523 §2.2, §3); `supportedMethods` leaves the method out until a client record can hold a private key (T19).
- **The redirect URI is bound to the attempt.** `Complete` sends the one `Begin` used (RFC 6749 §4.1.3); `CompleteInput` has none to confuse it with.
- **Every hardcoded value cites its source** beside it: an RFC section, the CIMD draft (`draft-ietf-oauth-client-id-metadata-document-02`), MCP authorization 2025-11-25, a vendor page when no RFC defines the behaviour (the `claims` challenge is Microsoft's, `invalid_refresh_token`, `internal_error` and `fatal_error` are Slack's), or a line of the prototype `internal/mcp/oauth.go` or `internal/connectors/runtime.go` on `codex/connector-support` at `cf62af0d`.
- **The CIMD document is built here, served by the API.** `ClientMetadataDocument(clientID, redirectURIs)` is what T17 serves at `Config.ClientMetadataURL`.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/...`. `oauth2code_test.go` is an outside-package testify suite against `fakeprovider`, with `providers/*.yaml` and `core/testdata/manifests/*.yaml` resolved and their endpoints pointed at the fake. `TestTheSlackManifestBuildsThePrototypesAuthorizeURL` is the golden test for the prototype's Slack authorize URL. `credential_test.go` drives `AccessCredential`, `Wrap`, `Classify` and `Revoke` against the fake's token-endpoint and resource personalities, with the suite's clock; each outcome has one test over a table of answers that mean it. `idtoken_test.go` and `privatekeyjwt_test.go` are inside the package, since the fake issues no id_token and takes no assertion. No mocks (`.claude/skills/go-testing/SKILL.md`).
