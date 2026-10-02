# internal/connectors/schemes/oauth2code

The `oauth2_code` scheme: the OAuth 2.0 authorization code grant with PKCE S256 (RFC 6749 §4.1, RFC 7636), one implementation for every provider a manifest describes. This package is part 1 (AI-834): `Begin` and `Complete`. `Mint`, `Wrap`, `Classify` and `Revoke` are part 2 (AI-836); until then they return an error, and `Wrap` fails every request rather than send one without a credential.

## Flow

```
Begin(Ref, Profile, RedirectURI)
  discover   manifest endpoints first; with endpoints.issuer, RFC 8414 metadata; with authorize or token
             missing (or a cimd/dcr policy) and no issuer, RFC 9728 metadata of endpoints.mcp, then RFC 8414.
             Each well-known URL is tried in MCP's order; one that fails (status, refused redirect, mismatched
             resource or issuer) is skipped, and only if none is usable is it an error naming every failure
  endpoints  every endpoint, pinned or discovered, passes Config.PublicEndpoint (egress) with its query removed
  PKCE       refuse a server listing methods without S256; refuse one found through RFC 9728 that lists none
  client     customer, operator (ClientLookup), cimd (Config.ClientMetadataURL), dcr (RFC 7591), in that order,
             each only if client.policy names it
  URL        authorize endpoint + authorize_params + response_type, client_id, redirect_uri, state,
             code_challenge, S256, resource, scope joined by scopes.separator
  -> AuthorizeURL, State (JSON: state, verifier, redirect URI, client, issuer, token endpoint, resource,
     scopes, expiry). The core seals State; this package never stores it.

Complete(Ref, Profile, State, Query)
  state      exactly one, equal in constant time, not expired (10 min), not spent in this process
  iss        RFC 9207: equal to the issuer, required when metadata says it is sent; checked before error
  error      AuthorizationError (access_denied and the rest)
  exchange   token endpoint, code + redirect_uri + code_verifier + resource + token_params, client auth
             none | client_secret_post | client_secret_basic; an "error" member is an error at any status
  id_token   iss, aud (this client only), azp (if present), exp checked when a capture rule reads it; the
             signature is not (OIDC Core §3.1.3.7, item 6)
  capture    Profile.Apply on the callback and the raw token response: account id, metadata, unverified
  -> Material{Scheme: oauth2_code, Version: 1, Payload: client, endpoints, tokens, expiry, scopes}, Captured
```

## Rules

- **No provider names.** Behaviour comes from the `core.Profile`: endpoints, `authorize_params`, `token_params`, `scopes.separator`, `client`, `capture`, `identity`. A provider that needs a branch here needs a manifest field or a hook instead (`internal/connectors/core/AGENTS.md`). No string literal in a non-test file is a provider name or a seeded connector id.
- **All outbound HTTP goes through `Config.HTTP`.** Discovery, registration and the code exchange use it and nothing else. The router passes `egress.NewClient(timeout, nil)`; tests pass the fake provider's `Client()`, because egress refuses loopback by design. Never build a client here, never replace its `Transport` or `CheckRedirect`.
- **Every authorization server endpoint passes `Config.PublicEndpoint`.** Nil means `egress.ValidatePublicHTTPSURL`. The authorize URL goes to the browser and never through `Config.HTTP`, so this is its only address check; the query is kept on the URL (RFC 6749 §3.1, §3.2) and removed only for the check. Tests that run on loopback pass a checker that lets a loopback IP literal through and nothing else; never pass one that returns nil for everything.
- **State is the core's to seal.** `BeginOutput.State` holds the PKCE verifier and, for a registered client, its secret; the core seals it into the attempt, whose one-use consumption is the replay guard across replicas. The in-process spent set refuses a replay within one process before its code reaches the provider, where a second redemption would revoke the grant (RFC 6749 §4.1.2).
- **A preregistered client's secret is never sealed.** It is looked up again at `Complete` (and by part 2 at refresh), so a rotated secret lives in one place. Only a client this scheme registered keeps its secret in State and Material.
- **The redirect URI is bound to the attempt.** `Complete` sends the one `Begin` used (RFC 6749 §4.1.3); `CompleteInput` has none to confuse it with.
- **Every hardcoded value cites its source** beside it: an RFC section, the CIMD draft (`draft-ietf-oauth-client-id-metadata-document-02`), MCP authorization 2025-11-25, or a line of the prototype `internal/mcp/oauth.go` on `codex/connector-support` at `cf62af0d`.
- **The CIMD document is built here, served by the API.** `ClientMetadataDocument(clientID, redirectURIs)` is what T17 serves at `Config.ClientMetadataURL`.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/...`. `oauth2code_test.go` is an outside-package testify suite against `fakeprovider`, with `providers/*.yaml` and `core/testdata/manifests/*.yaml` resolved and their endpoints pointed at the fake. `TestTheSlackManifestBuildsThePrototypesAuthorizeURL` is the golden test for the prototype's Slack authorize URL. `idtoken_test.go` is inside the package, since the fake issues no id_token. No mocks (`.claude/skills/go-testing/SKILL.md`).
