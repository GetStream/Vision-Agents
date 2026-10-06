# internal/connectors/schemes/oauth2cc

The `oauth2_client_credentials` scheme: the OAuth 2.0 client credentials grant (RFC 6749 §4.4), one implementation for every provider a manifest describes. A connection the app owns gets its tokens from a client id and secret, with no person to consent: Salesforce's connected app with a «Run As» user is the first (`providers/salesforce.yaml`). T24 of AI-816 (AI-847). It is the second scheme added after the four the branch had, and it touches no file in `core/`.

## Flow

```
Begin(Manifest)              endpoints.token set, client.auth_method basic or post (or unset) -> Done
Complete(Manifest, Supplied) Supplied{client_id, client_secret}: any other key refused; both VSCHAR (RFC 6749
                             App. A.1, A.2), both required (§4.4: confidential clients only)
  mint                       POST endpoints.token: grant_type=client_credentials, resource (endpoints.resource,
                             RFC 8707), no scope; client_secret_basic unless the manifest says post
                             (oauth2code.TokenRequest, which holds the endpoint to Config.PublicEndpoint)
  capture                    ResolvedManifest.Apply(no callback, token response): account id, metadata
  -> StoredCredentials{oauth2_client_credentials, v1, {client_id, client_secret, access_token, expires_at}},
     AccountInfo (scopes from the response's scope, if any)
Retrieve(stored, m, opts)
  due                        expiry known and within refresh.margin (default 1 min) or at or before
                             opts.ValidUntil; otherwise -> the token, stored as is
  mint                       the same request with the stored client; no opts.Checkpoint (it spends nothing)
  -> AccessCredential, new StoredCredentials; or *core.OutcomeError and no StoredCredentials, with the old
     token beside it while that has not expired
  refusal                    oauth2code's Classify, then: invalid_client, unauthorized_client, invalid_client_id
                             or a 401 -> InvalidGrant (the client is the grant); RateLimited, InvalidGrant,
                             ScopeRequired stay; Uncertain, OK and the rest -> Transient
Wrap                         oauth2code.Bearer: Authorization: Bearer on a clone; a foreign credential fails
Classify                     oauth2code's Classify, for a resource's answer
Revoke(stored, m)            endpoints.revoke (else ErrNoRevocationEndpoint), RFC 7009 with the access token;
                             unsupported_token_type -> ErrTokenTypeNotRevocable; 200 -> nil. The client
                             lives on at the provider
```

## Rules

- **The client is the connection's, in its StoredCredentials.** The client id and secret are supplied at `Complete` and sealed with the token, not looked up in `connector_oauth_clients` (#758) or the operator's environment as `oauth2_code` does: a client credentials client acts as one tenant's «Run As» user, so it is one connection's credential, and deleting the connection forgets it. Check: `go test -run 'TestOAuth2CCSuite/TestCompleteGetsATokenWithTheSuppliedClient' ./internal/connectors/schemes/oauth2cc`.
- **No checkpoint.** A client credentials request spends nothing: no refresh token goes out (RFC 6749 §4.4.3: none is issued), the secret stays valid whatever the answer, and a second request mints another token. Committing `needs_reauthorization` before it would turn a dropped connection into a reconnect. So a lost answer, a 5xx and a 2xx without a token are Transient here, where a refresh's are Uncertain. Check: `go test -run 'TestOAuth2CCSuite/(TestReplacingATokenNeverCheckpoints|TestEachAnswerThatGaveNoTokenAndSpentNothingIsTransient|TestALostAnswer)' ./internal/connectors/schemes/oauth2cc`, and with the resolver: `go test -tags integration -run TestResolverSuite/TestASalesforceTokenEndpointThatIsDown ./internal/connectors/resolver`.
- **A refused client is InvalidGrant.** `invalid_client` and `unauthorized_client` (RFC 6749 §5.2), Salesforce's `invalid_client_id` and a 401 mean the stored client no longer works: only new client credentials help. `oauth2_code` makes the same codes Transient, because there the client is not the grant. Check: `go test -run TestOAuth2CCSuite/TestEachAnswerThatRefusesTheClientIsInvalidGrant ./internal/connectors/schemes/oauth2cc`.
- **The secret goes in the Authorization header only.** `client_secret_basic` by default (RFC 6749 §2.3.1, the method every server must support), `client_secret_post` in the body when the manifest says so; never in a URL or a log, and no error quotes it. Check: `go test -run 'TestOAuth2CCSuite/(TestTheSecretGoesInTheAuthorizationHeaderOnly|TestClientSecretPost)' ./internal/connectors/schemes/oauth2cc` and `TestOAuth2ClientCredentialsSchemeContract`.
- **Reuse oauth2code, add no OAuth plumbing here.** Token requests go through `oauth2code.TokenRequest`, bearer tokens through `oauth2code.Bearer`, and answers are read by an `oauth2code.Scheme`'s `Classify`. Check: `grep -n 'oauth2code\.' internal/connectors/schemes/oauth2cc/scheme.go`.
- **All outbound HTTP goes through `Config.HTTP`**, which the router builds with `egress.NewClient`, and every endpoint passes `Config.PublicEndpoint` first (nil is `egress.ValidatePublicHTTPSURL`). Check: `go test -run TestOAuth2CCSuite/TestATokenEndpointThatIsNotPublic ./internal/connectors/schemes/oauth2cc`.
- **No provider names in code.** What differs between providers is a `core.ResolvedManifest` field; the one vendor error code is cited beside it. Check: `go test ./internal/connectors/core -run TestGuardSuite/TestCoreNamesNoProvider` covers `core`; here, `grep -n -i 'salesforce' internal/connectors/schemes/oauth2cc/scheme.go` prints only comments.
- **Every hardcoded value cites its source** beside it: an RFC section, a vendor page with the date it was read, a live answer with its timestamp, or the prototype at `cf62af0d`.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/schemes/oauth2cc`. `TestOAuth2ClientCredentialsSchemeContract` runs `core/contracttest.SchemeContract` against `fakeprovider` with `ClientCredentials`; `OAuth2CCSuite` covers what is this scheme's own, including the built-in Salesforce manifest against the fake (`ClientCredentials`, `IdentityURL`). The resolver end to end, with Postgres: `go test -tags integration -run 'TestResolverSuite/(TestAnAppOwnedSalesforce|TestASalesforce)' ./internal/connectors/resolver`. No mocks (`.claude/skills/go-testing/SKILL.md`).
