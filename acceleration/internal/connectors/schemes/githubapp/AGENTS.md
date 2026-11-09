# internal/connectors/schemes/githubapp

The `github_app` scheme: a GitHub App acting as one of its installations (the bot), with an installation access token minted from the app's private key. Stress-test row 6 of the architecture doc (`core/testdata/manifests/github.yaml`). The built-in `providers/github.yaml` lists it from revision 12, beside `oauth2_code` and `bearer`.

## Flow

```
Begin(Manifest)              endpoints.api_base set -> Done
Complete(Manifest, Supplied) Supplied{app_id, private_key, installation_id, permissions?, repositories?}: any other key
                             refused; app_id letters, digits and dots, installation_id digits, private_key an RSA PEM
                             (PKCS#1 as GitHub downloads it, or PKCS#8), permissions a JSON object of name to
                             read|write|admin, repositories at most 500 names without their owner
  mint                       POST {api_base}/app/installations/{installation_id}/access_tokens, Bearer app JWT
                             (RS256; iat now-60s, exp now+9m, iss app_id), body {repositories, permissions} when set
  check                      GET {api_base}/installation/repositories?per_page=1 with the token -> 200
  -> StoredCredentials{github_app, v1, {app_id, private_key, installation_id, permissions, repositories, token,
     expires_at}}, AccountInfo{} (nothing captured)
Retrieve(stored, m, opts)    due within refresh.margin (default 5 min) of expires_at, at or before opts.ValidUntil, or
                             opts.Refused -> mint again with the stored key; otherwise the token, stored as is. No
                             checkpoint: a mint spends nothing
Wrap                         Authorization: Bearer <token> (oauth2code.Bearer)
Classify                     oauth2code.ClassifyStatic
Revoke                       DELETE {api_base}/installation/token with the token -> 204
Static                       core.Static: a key GitHub refuses is replaced with PUT .../credentials, never consented to
```

A refused request: 401 (a JWT GitHub cannot verify: wrong key or app id), 403 without a rate limit (a suspended installation, a permission not granted), 404 (an unknown installation) and 422 (a narrowing GitHub will not grant) are InvalidGrant; a 403 with `x-ratelimit-remaining: 0`, `retry-after` or a message naming a rate limit (a secondary limit may send neither header), and a 429, are RateLimited; anything else is Transient. The error says GitHub's status and message, with the JWT and token it may echo redacted.

## Rules

- **Never the key, the JWT or a token in a log, an error or a URL.** The key and token are in the sealed payload alone; GitHub's message is passed through `core.LoggedDescription` with what the request carried. Check: `go test -run 'TestGitHubAppSchemeContract|TestGitHubAppSuite/TestNoSecret' ./internal/connectors/schemes/githubapp`.
- **Static, so `oauth2_code` stays GitHub's default.** `api.defaultScheme` picks a connector's one scheme that is not `core.Static`; github_app is static, so a connection naming no scheme still consents. Check: `go test -run TestGitHubAppSuite/TestTheGitHubManifest ./internal/connectors/schemes/githubapp`.
- **The app JWT is `oauth2code.PrivateKeyJWT.Sign`**, with GitHub's three claims; no JWT code here.
- **All outbound HTTP goes through `Config.HTTP`**, and every URL passes `Config.PublicEndpoint` first (nil is `egress.ValidatePublicHTTPSURL`).
- **The installation id is sealed, not captured.** A connection's metadata comes only from a manifest's capture rules, and an input the manifest declares is required of every scheme, so neither can carry a value only github_app takes. It is checked as digits before it goes into a path.
- **Every hardcoded value cites its source** beside it, as the package comment's vendor pages.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/schemes/githubapp`. `TestGitHubAppSchemeContract` runs `core/contracttest.SchemeContract` against a local TLS GitHub (`github_test.go`) that verifies the app JWT with the app's public key and mints tokens that expire on the suite's clock; `GitHubAppSuite` covers the JWT, cache and renewal, refusals, narrowing, revoke, the sandbox token and redaction. Keys are generated per test; never use a real app's key. Live, with a test app's credentials from the environment only:

```bash
E2E_GITHUB_APP_ID=... E2E_GITHUB_APP_PRIVATE_KEY=path/to/key.pem E2E_GITHUB_APP_INSTALLATION_ID=... \
  go test -tags integration -run TestGitHubAppLiveSuite ./internal/connectors/providers
```
