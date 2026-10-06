# AI-816 connectors: PR-sized subtasks

Oct 1, 2026 · @Kanat Kiialbaev

Exported from Claude Docs on 2026-10-05 (https://claude.ai/code/artifact/e4d97114-1334-4181-beab-c5721909c201). The Claude Doc is the source of truth; this copy is a snapshot.

33 subtasks in 7 phases, each one PR, in the order they can land, plus 21 subtasks (T37 to T57) added on October 5 from the channel decisions, plus 4 subtasks (T58 to T61) added on October 6 to move plugins onto connectors. Status on October 6: 17 merged on accelerate (T1 to T11, T15 to T17, T20, T32, T37). They implement [Accelerate connectors: architecture design](architecture.md) on top of the code that exists today. Connector-layer subtasks are Linear sub-issues of [AI-816](https://linear.app/stream/issue/AI-816/basic-connectorsmcp-support); the channel bridge and the omni-channel conversation have their own parent issues, AI-866 and AI-867. Each subtask's Linear number follows its title.

## Ground rules

- **Base branch is `accelerate`.** Every PR targets it. The prototype branch `codex/connector-support` is a source of code to copy where the design says «keep», not a base: it changes 197 files at once, carries SDK regenerations for ten languages and an irreversible plugin migration, none of which is reviewable as one PR.
- **What `accelerate` has today** (checked October 1): `internal/plugins` with five catalog entries and endpoints in `api/legacy.yaml:1061-1177`; one-version AES-GCM sealer in `internal/auth/secret.go` (`KEKVersion = 1`); no `internal/egress`, no `internal/mcp`, no `internal/connectors`; Huma for new operations (`internal/api/policies.go` is the pattern); `cmd/openapi` renders the spec and a test fails when it is stale.
- **Layer order inside a phase:** migration and store, then core logic, then API, then session wiring, then SDKs. A PR touches one layer unless the layer is useless alone.
- **New endpoints use Huma, not `legacy.yaml`.** The prototype registered its connector endpoints through the generated oapi-codegen server (`ListConnectorsRequestObject` and friends in `internal/api/connectors.go`). AGENTS.md forbids adding to `legacy.yaml`, so the API phase ports the handlers to `huma.Register`.
- **Go first, SDKs later.** AGENTS.md: SDK changes start with Go; the other SDKs follow in their own PRs. Each API PR regenerates `openapi.yaml` and the Go SDK only.
- **Every PR keeps `go vet ./...` and `go test ./...` green** (`.github/workflows/ci.yml:78-98`), adds its own tests against real local servers and a temporary database, never mocks (AGENTS.md), and leaves the router startable with the feature off until the session wiring lands.
- **Nothing is deployed to production.** Router runs only on staging, where `agent_plugin_connections` had 0 rows on October 1 (competitor doc, «Where plugins run today»). Plugins grew after that, so their rows move onto connectors with a command (T61) before the removal (T23).
- **Channels are a later layer, not part of these 33 PRs.** Thierry on October 1: «the omni channel/connector concept will be important», naming Linq and Chatbase; later that day: «AI requires a good way to integrate slack, whatsapp, rcs, texting, imessage, maybe telegram». The transport is now decided as a proposal (channels doc, October 5): a channel bridge in the Router writes each external thread into its own thread channel in Stream Chat, one episode card goes to the person's omni-channel, and Router's existing message hook answers. T34 to T57 implement it: the connector-layer part in Phase 7 under AI-816, the bridge and the conversation under AI-866 and AI-867. One rule still holds now: T26's inbound endpoint and verifier registry are built so the channel bridge reuses them.
- **Each subtask lists:** title, description, scope, out of scope, dependencies, acceptance criteria. Titles are written for Linear.

## Order and dependencies

&#91;embedded content: PR waves · 53 PRs in 13 waves by dependency depth, 9 merged\]

A wave is the earliest point a PR can start: its depth in the dependency graph, computed from each subtask's Dependencies line. PRs in one wave do not depend on each other, so two people can take two of them. The waves are a conservative schedule; a PR may start as soon as its own dependencies merged, which is often earlier than its wave. The phases below group PRs by layer; the waves group them by time. T34 to T36 (channels) are drawn dashed: they have a wave by dependency (11 and 12) but no date until a channel is scheduled. T37 to T57 are not drawn yet; the table at the start of Phase 7 gives their waves.

## Phase 0: foundations and guardrails

Five PRs with no user-visible change. They give later PRs a place to put code, a test server to run against, and a CI rule that keeps provider names out of the core from day one.

### T1. Versioned KEK keyring in auth.Sealer · [AI-828](https://linear.app/stream/issue/AI-828)

**Status: merged** in [#708](https://github.com/GetStream/Vision-Agents/pull/708) (`b488194f`), October 1.

- **Description.** Extend `internal/auth/secret.go` from one key to a keyring: `ROUTER_AUTH_KEK_V<n>` plus `ROUTER_AUTH_KEK_VERSION`, with `ROUTER_AUTH_KEK` as the version-1 alias. Add `SealWithAAD`, `OpenWithAADVersion` and `CurrentVersion`. Copy from the branch (`internal/auth/secret.go`, 90-line diff). Build the sealer whenever connectors are enabled, in every auth mode: today `cmd/router/main.go:225` builds it only for `api_key`, and staging runs `proxy` (`internal/config/staging.yaml:3-4`), where `main.go:211-215` returns before a sealer exists.
- **Scope.** `internal/auth/secret.go`, its tests, the sealer construction in `cmd/router/main.go`, README variables. Existing `Seal` and `Open` keep working for API secrets.
- **Out of scope.** Rewrapping rows of any table; connector material (T8).
- **Dependencies.** None.
- **Acceptance.** A row sealed under version 1 opens after version 2 is added; a wrong AAD fails to open; missing keyring at startup fails fast with a clear message; with `auth.mode=proxy` and connectors enabled the router builds the sealer and fails fast without a keyring; with connectors off, `proxy` starts without a KEK as it does today; existing API-secret tests pass unchanged.
- **Before deploy.** Add `ROUTER_AUTH_KEK_V1` to staging's secrets, then redeploy. Staging had no KEK on October 1 (secret names listed, none is a KEK), which fits the chart's `router.authMode: proxy`. The chart loads the staging secrets into the pod's environment, so it needs no change; the deploy runbook in the infra repo shows how to add one key. Kanat has the permission to add it (checked October 1).

### T2. Egress policy for outbound connector traffic · [AI-829](https://linear.app/stream/issue/AI-829)

**Status: merged** in [#707](https://github.com/GetStream/Vision-Agents/pull/707) (`df8e1b58`), October 1.

- **Description.** Add `internal/egress` with `ValidatePublicHTTPSURL` (https only, no userinfo, no query string, public IPs after DNS resolution, no redirects across hosts) and `NewPublicHTTPClient`. Copy from the branch (`internal/egress/public.go`, 154 lines).
- **Scope.** The package and its tests. No caller yet.
- **Out of scope.** Wiring into any transport (T13).
- **Dependencies.** None.
- **Acceptance.** Private, loopback, link-local and metadata addresses are refused; a `javascript:` or `http:` URL is refused; a public https URL passes; the client refuses cross-host redirects.

### T3. Core contracts and the CI guard · [AI-830](https://linear.app/stream/issue/AI-830)

**Status: merged** in [#709](https://github.com/GetStream/Vision-Agents/pull/709) (`4020114e`), October 1.

- **Description.** Create `internal/connectors/core` with the interfaces from the design doc: `Scheme`, `ToolSource`, `Toolset`, `CredentialStore`, `Resolver`, `Verifier`, `Hook`, the `Registry`, and the value types `StoredCredentials`, `AccessCredential`, `Outcome`, `CredentialRequest`, `ConnectionRef`, `Signal`, `ResolvedBinding`. No implementation. Add the two guard tests: `TestCoreImportsNoAdapter` (`core` imports nothing under `schemes/`, `sources/`, `credentialstores/`, `signals/`, `providers/`) and `TestCoreNamesNoProvider` (no string literal in `core` equals a connector id or a name in the deny list).
- **Scope.** `internal/connectors/core/*.go`, doc comments, the two tests.
- **Out of scope.** Any adapter; the manifest model (T4).
- **Dependencies.** None.
- **Acceptance.** Package compiles with zero dependencies on other `internal/connectors` packages; both guard tests pass and a deliberate `case "slack"` added in a test build fails `TestCoreNamesNoProvider`.

### T4. Provider manifest model: parsing, validation, templates, capture and identity rules · [AI-832](https://linear.app/stream/issue/AI-832)

**Status: merged** as commit `e44b057d` («provider manifest model with 12 stress-test fixtures»), October 1.

- **Description.** The `Profile` type and the manifest schema (`inputs`, `vars`, `endpoints` as `{var}` templates, `schemes`, `client`, `scopes`, `identity`, `capture`, `refresh`, `rate_limit`, `sources`, `hooks`). Pure functions: parse YAML or JSON, validate (every `{var}` is a declared input, a `vars` entry or a captured name; every hook name is a string; enums closed), resolve endpoints for a set of inputs, apply `capture` and `identity` rules to a token response, a callback query or an `id_token`.
- **Scope.** `internal/connectors/core/manifest.go` and tests. Test fixtures: the 12 stress-test manifests from the design doc under `testdata/manifests/`, with recorded token responses and callbacks.
- **Out of scope.** Storing manifests (T6); seeding; hooks implementations.
- **Dependencies.** T3.
- **Acceptance.** All 12 fixture manifests load; Salesforce resolves production and sandbox hosts from `environment`; QuickBooks captures `realm_id` from a callback query; Google reads `sub` from an `id_token`; a manifest with an undeclared `{var}` is rejected with the variable named.

### T5. Fake provider server for tests · [AI-831](https://linear.app/stream/issue/AI-831)

**Status: merged** in [#715](https://github.com/GetStream/Vision-Agents/pull/715) (`a3382c90`), October 2.

- **Description.** `internal/connectors/fakeprovider`: one `httptest`-based server that plays an OAuth authorization server and an MCP endpoint, with switchable personalities: rotating refresh with a grace window, non-rotating refresh, no refresh token, `invalid_grant`, lost response, `insufficient_scope` on 403, a `claims` challenge on 401, 429 with `Retry-After`, comma scopes with `authed_user`, `realmId` in the callback, a signed callback, PKCE and `iss` checks. Start from the fake-provider tests on the branch (`connector-design.md:571`).
- **Scope.** The package and a self-test per personality.
- **Out of scope.** Any production code.
- **Dependencies.** None (T4 fixtures may reuse its recorded responses).
- **Acceptance.** Each personality has a test proving it behaves as named; the server is usable from any package test with one constructor call.

## Phase 1: storage

Three PRs: the schema and the store methods, nothing that calls them yet. Migrations follow the repo's goose files under `acceleration/migrations/` (latest today: `20260930120000_agent_config_speed.sql`).

### T6. Connector definitions table with manifest and revision, seeded from YAML · [AI-833](https://linear.app/stream/issue/AI-833)

**Status: merged** in [#716](https://github.com/GetStream/Vision-Agents/pull/716) (`96a59044`), with the migration-order fix [#728](https://github.com/GetStream/Vision-Agents/pull/728), October 2.

- **Description.** Migration `connector_definitions` (`customer_id`, `id`, `revision`, `name`, `category`, `description`, `manifest jsonb`, timestamps; primary key `customer_id, id, revision`; built-ins under a reserved customer id). Store: `CreateConnectorDefinition`, `ConnectorDefinition(id, revision)`, `LatestConnectorDefinition`, `ListConnectorDefinitions`. A startup seeder reads `internal/connectors/providers/*.yaml` and inserts a new revision only when the manifest changed. First two manifests: Slack and Linear, written from the branch's `connectors.yaml` plus the design's new fields.
- **Scope.** One migration, `internal/store/connectors.go` (definitions part), the seeder, Slack and Linear manifests, store integration tests.
- **Out of scope.** The API over definitions (T15); the other five providers (T32); custom definitions beyond the store method.
- **Dependencies.** T4.
- **Acceptance.** Router starts on an empty database and the two built-ins exist at revision 1; a changed YAML on restart creates revision 2 and leaves revision 1; a custom definition id must start with `custom_` and cannot shadow a built-in.

### T7. Connections, authorization attempts and the agent config column · [AI-835](https://linear.app/stream/issue/AI-835)

**Status: merged** in [#729](https://github.com/GetStream/Vision-Agents/pull/729) (`f955bc49`), October 2.

- **Description.** Migration `connector_connections` with the branch's columns (`20260929170000_connectors.sql`) plus the design's changes: `auth_scheme TEXT` and `tls_scheme TEXT NULL` instead of the `auth_type` CHECK, `definition_revision`, `inputs jsonb`, `metadata jsonb`, `credentials_sealed BYTEA`, `credentials_kek_version`. Migration `connector_authorization_attempts` as on the branch with `kind TEXT` and `kek_version`. `ALTER TABLE agent_configs ADD COLUMN connectors JSONB`. Store: create, get, list by owner, soft delete, `ConnectorConnectionReferenced`, and the attempt methods (create with cleanup, by state, by id, consume once).
- **Scope.** Two migrations, store methods and models, integration tests.
- **Out of scope.** The advisory lock and revisioned save (T8); any handler.
- **Dependencies.** T6, T1.
- **Acceptance.** `owner_type` is `app` with empty `owner_id` or `user` with a non-empty one, enforced by CHECK; `auth_scheme` is validated against the registry at write time, not by a CHECK; an attempt is consumed exactly once under concurrent callers; a soft-deleted connection is invisible to every read.

### T8. Locked grant backend: advisory lock, checkpoint, revision CAS, sealed material · [AI-839](https://linear.app/stream/issue/AI-839)

**Status: merged** in [#733](https://github.com/GetStream/Vision-Agents/pull/733) (`7a2759ee`), October 5.

- **Description.** `internal/connectors/credentialstores/pgsealed` implementing `core.CredentialStore`: `Update` with a session-level `pg_advisory_lock` on (tenant, connection), a `checkpoint` closure, and the revision compare-and-swap, copied from the branch (`store/connectors.go:198-347`). Seal and open `StoredCredentials` with AAD bound to tenant, connection id and revision (from `connectors/secrets.go`).
- **Scope.** The backend package, the `SaveConnectorConnectionAtRevision` store method, integration tests ported from the branch: one committed revision under concurrent rotation, lost response leaves `needs_reauthorization` and never replays a refresh token.
- **Out of scope.** Refresh itself (T10); the resolver (T12).
- **Dependencies.** T7, T1, T3.
- **Acceptance.** `TestConcurrentCredentialResolutionCommitsOneRotatedRefreshToken` and `TestRefreshOutcomeSurvivesLostResponsesAndCanceledWorkers` pass against the new backend with a fake rotation; a stale revision save returns `ErrConnectorConnectionChanged`; material sealed under KEK v1 is rewrapped to v2 on next successful use.

## Phase 2: schemes

Three PRs. The OAuth scheme is the branch's `internal/mcp/oauth.go` (961 lines) with the provider rules taken out; it is split in two so the consent half and the refresh half are reviewed apart.

### T9. Scheme oauth2\_code, part 1: discovery, client registration, consent and exchange · [AI-834](https://linear.app/stream/issue/AI-834)

**Status: merged** in [#730](https://github.com/GetStream/Vision-Agents/pull/730) (`06a90667`), October 2.

- **Description.** `internal/connectors/schemes/oauth2code` implementing `Begin` and `Complete`: RFC 9728 and 8414 discovery with manifest overrides, CIMD first then DCR, client auth `none`, `client_secret_post`, `client_secret_basic`, PKCE S256, `resource`, `state`, the authorize URL built from the `ResolvedManifest` (scope separator and extra params from the manifest, not from `connector.ID == "slack"`), code exchange with the `iss` check, then `capture` and `identity` rules from T4 on the token response and callback query. Provider branches in `oauth.go:265,823-830,832-887` and the Slack and Calendly fields of `tokenResponse` are deleted, not ported.
- **Scope.** The scheme package, the public client metadata document, tests against T5 for CIMD, DCR, confidential client, denial, replayed state, `iss` mismatch, comma scopes, `realmId` capture.
- **Out of scope.** Refresh, `Classify`, `Wrap`, `Revoke` (T10); the HTTP handlers (T17).
- **Dependencies.** T3, T4, T5, T2.
- **Acceptance.** All T5 consent personalities pass; `TestCoreNamesNoProvider` still passes; the Slack manifest with `separator: ","` produces the same authorize URL the branch produced for Slack.

### T10. Scheme oauth2\_code, part 2: mint, classify, wrap, revoke · [AI-836](https://linear.app/stream/issue/AI-836)

**Status: merged** in [#734](https://github.com/GetStream/Vision-Agents/pull/734) (`138c405e`), October 5.

- **Description.** `Retrieve` renews from `StoredCredentials` using `RefreshPolicy` (margin, `send_scope`, rotating with a grace retry inside the window, `token_ttl` warning), returns the new `StoredCredentials`. `Classify` maps responses to `Outcome`: `invalid_grant` and `invalid_refresh_token` to InvalidGrant, network and 5xx to Uncertain or Transient as the branch does (`oauth.go:539-595`), 403 `insufficient_scope` and a 401 `claims` challenge to ScopeRequired, 429 to RateLimited with `Retry-After`. `Wrap` sets the bearer header. `Revoke` calls the manifest's `revoke` endpoint when present and reports best effort.
- **Scope.** The same package; tests against T5 for every outcome; a `private_key_jwt` client auth method as one file, used by nothing yet.
- **Out of scope.** The lock and status transitions (T12).
- **Dependencies.** T9.
- **Acceptance.** A refresh under the grace window with the old token succeeds once; `scope` is sent only when the policy says; a lost response returns Uncertain and the caller's `StoredCredentials` is unchanged; each outcome has a table-driven test.

### T11. Static schemes and the scheme contract suite · [AI-840](https://linear.app/stream/issue/AI-840)

**Status: merged** in [#745](https://github.com/GetStream/Vision-Agents/pull/745) (`4f379ba6`), October 6. Open: the `api_key` header name comes from the caller (`Supplied["header"]`), so a built-in manifest with a fixed header needs a manifest field. T18 settles the `Supplied` key names.

- **Description.** `schemes/apikey`, `schemes/bearer`, `schemes/none`, each a few dozen lines: `Begin` returns Done, `Complete` seals the supplied value, `Retrieve` returns it with no expiry, `Wrap` sets the configured header (with the forbidden-header list from `api/connectors.go:1050-1057`). Plus `core/contracttest.SchemeContract`, a table-driven suite any scheme runs: round trip, concurrent Retrieve commits once, no secret in URL, log or error text, `Classify` covers the six outcomes, `Revoke` is honest.
- **Scope.** Three scheme packages, the contract package, and its application to all four schemes.
- **Out of scope.** `basic`, `mtls`, `aws_sigv4`, `github_app` (phase 6 and later).
- **Dependencies.** T3, T5, T10.
- **Acceptance.** All four schemes pass `SchemeContract`; adding a fifth scheme needs no change outside its own package, shown by T24 later.

## Phase 3: resolver and sources

Three PRs. After them a connection with a grant can be turned into an authorized MCP call from a Go test, with no HTTP API and no session yet.

### T12. Credential resolver · [AI-843](https://linear.app/stream/issue/AI-843)

- **Description.** `core.Resolver` implementation: `Resolve(ref, need)` loads the connection, checks status, opens `StoredCredentials through the credential store`, calls `Scheme.Retrieve` on a detached context with its own deadline under the backend lock with the checkpoint before any refresh, persists rotated material at the next revision, maps `Outcome` to status and `last_error` (`needs_reauthorization`, `connected`, temporary), and caches the fast path by (connection, revision) with an explicit maximum age. `Invalidate(ref, why)` moves the connection to `needs_reauthorization` and drops the cache entry. This replaces the branch's `connectors.ResolveCredentials` (`runtime.go:26-153`), whose refresh ran on the tool call's context.
- **Scope.** `internal/connectors/core/resolver.go`, integration tests: refresh race, lost response, interruption during refresh does not change status, disconnect blocks a new resolve within the cache window.
- **Out of scope.** Request wrapping (T13); rate limiting (T28).
- **Dependencies.** T8, T10, T11.
- **Acceptance.** Cancelling the caller's context during a refresh leaves the connection `connected` or rotated, never `needs_reauthorization`; p50 and p95 of the fast path are printed by a benchmark test (no target asserted yet, per the design's spike 4).

### T13. Transport composition and the wrapping order · [AI-845](https://linear.app/stream/issue/AI-845)

- **Description.** `core.ResolvedBinding.Transport`: builds the outbound client with `egress.NewClient(timeout, scheme.Wrap)`. `Scheme.Wrap` (and `tls_scheme` when set) is the outer layer and sees the final request; the egress transport sits under it, checks the URL and dials only a checked public IP. Egress is last on purpose: it must judge the request that actually leaves, after the scheme has finished with it. One transport per connection, cached, closed on disconnect. A `RoundTrip` test proves a signing scheme sees the final headers and body and that a private destination is refused before anything leaves the router.
- **Egress (from [AI-829](https://linear.app/stream/issue/AI-829), PR #707).** Build every connector client with `egress.NewClient(timeout, scheme.Wrap)` and nothing else; never replace the returned client's `Transport`. It owns the redirect policy (same origin only, no method change), the URL check before and after `wrap`, and the dial-time public-IP check, so the order is fixed by construction. Close a connection's client with `CloseIdleConnections`, which reaches the inner transport through the wrapper. The dial-time check runs after `wrap`, so a scheme that retrieves or refreshes a token per request does so before a name that resolves to a private address is refused; the token never leaves the process. If Retrieve itself must not run, call `egress.ValidatePublicHTTPSURL` on the endpoint first.
- **Scope.** `internal/connectors/core/transport.go`, tests with a recording scheme.
- **Out of scope.** `mtls` itself (later). Network-level egress isolation ([AI-864](https://linear.app/stream/issue/AI-864)).
- **Dependencies.** T12, T2.
- **Acceptance.** Order is enforced by construction: a source cannot obtain a transport without egress; a test with a fake signing scheme sees `Content-Length` and the body hash of the final request; a cross-host redirect never carries the scheme's credential.

### T14. Source mcp · [AI-849](https://linear.app/stream/issue/AI-849)

- **Description.** `internal/connectors/sources/mcp` implementing `core.ToolSource` from the branch's `internal/mcp/mcp.go`: `Discover` with the official Go SDK, paginated `tools/list`, `ToolSchemaDigest` over name, description and input schema; `Open` returns a `Toolset` with the allowlist, digest check, JSON Schema validation of arguments, prefixed names and collision detection, 4 MiB response and 32 KiB result caps, `isError` kept as an error. Plus `core/contracttest.SourceContract`.
- **Scope.** The source package, the contract suite, tests against a local MCP server from T5.
- **Out of scope.** The dispatcher and envelope (T21); `http` and `openapi` sources (T25).
- **Dependencies.** T3, T13, T5.
- **Acceptance.** `SourceContract` passes: stable digest, ungranted tool never dispatched, changed schema hidden, oversized result cut with the marker, non-text `isError` still an error; `startupTimeout` of 10 s bounds `Discover`.

## Phase 4: API with Huma

Five PRs. Each registers its operations with `huma.Register` beside Go request and response structs, runs `go run ./cmd/openapi`, regenerates the Go SDK and nothing else. The branch's handlers in `internal/api/connectors.go` are the behavior to keep; their generated oapi-codegen wrapper types are not ported.

### T15. Connector definitions endpoints · [AI-837](https://linear.app/stream/issue/AI-837)

**Status: merged** in [#727](https://github.com/GetStream/Vision-Agents/pull/727) (`c72ad203`), October 2.

- **Description.** `GET /v1/agents/connectors` (search built-ins and the app's custom definitions), `GET /v1/agents/connectors/{id}`, `POST /v1/agents/connectors` for a custom MCP definition (id `custom_*`, public https endpoint, scheme from the registry). Responses expose the non-secret manifest: schemes, inputs, scopes, client registration methods.
- **Scope.** `internal/api/connectors.go` (new file on `accelerate`), OpenAPI regen, Go SDK regen, handler tests.
- **Out of scope.** Connections (T16); editing a built-in.
- **Dependencies.** T6.
- **Acceptance.** Operations are server-side only unless marked `x-client-accessible`; the OpenAPI freshness test passes; a custom id that shadows a built-in returns 400 in the API's `{"error": ...}` shape.

### T16. Connections CRUD with owner checks · [AI-841](https://linear.app/stream/issue/AI-841)

**Status: merged** in [#731](https://github.com/GetStream/Vision-Agents/pull/731) (`3c0beee2`), October 5. Its paths `/v1/agents/connections/{id}` are where T44 and T45 add `proxy` and `token`.

- **Description.** `POST /v1/agents/connections` (connector id, owner `app` or `user`, inputs validated by the manifest, label), `GET /v1/agents/connections` with filters, `GET` and `DELETE /v1/agents/connections/{id}`. Owner rule from the branch: a user-owned connection is created only by the server-side backend and only for the verified user it acts for; reads of another user's connection return not-found. Delete is the soft delete from T7 and refuses a connection still bound by a fixed binding unless `force=true` (`ConnectorConnectionReferenced`).
- **Scope.** Handlers, OpenAPI and Go SDK regen, tests for cross-user and anonymous callers.
- **Out of scope.** Credentials (T18); OAuth (T17).
- **Dependencies.** T7, T15.
- **Acceptance.** Alice cannot read, delete or authorize Bob's connection; an anonymous or guest caller gets 400 on a user-owned create; inputs missing from the manifest return 400 naming the input.

### T17. OAuth consent flow: authorizations, launch page, handoff, callback · [AI-844](https://linear.app/stream/issue/AI-844)

**Status: merged** in [#747](https://github.com/GetStream/Vision-Agents/pull/747) (`7390f97d`), October 6. `core.CredentialState` gained `AccountID`, `Metadata` and `Scopes`. Running it on staging needs [AI-896](https://linear.app/stream/issue/AI-896): the staging gateway answers 401 on every no-auth route, and staging has no KEK and connectors off.

- **Description.** `POST /v1/agents/connections/{id}/authorizations` creates a sealed `Attempt{Kind: consent | reconnect}` with a 10-minute lifetime and returns the router-hosted launch URL and a handoff token; the launch page, the origin-checked handoff that sets the HttpOnly callback cookie, `GET /v1/agents/connectors/oauth/callback` (state consumed once, cookie must match, `iss` check, exchange through `Scheme.Complete`, account-switch rejection), and `/.well-known/oauth-client-metadata`. All copied from the branch's handlers (`api/connectors.go:294-488,760-878`), with the `BeforeComplete` hook point called from the callback.
- **Scope.** Handlers, the launch HTML, cookie helpers, OpenAPI and Go SDK regen, tests against T5 for success, denial, replay, wrong browser, account switch.
- **Out of scope.** `step_up` and `admin_consent` kinds (T27); OAuth client records (T19).
- **Dependencies.** T9, T16.
- **Acceptance.** A consent completed in a different browser is refused; a replayed callback is refused; a callback for a different provider account leaves the old grant intact and reports it; `ROUTER_PUBLIC_URL` unset returns 400 at authorize, not at callback.

### T18. Credentials write, validate and tools · [AI-850](https://linear.app/stream/issue/AI-850)

- **Description.** `PUT /v1/agents/connections/{id}/credentials` with `expected_revision`: a static value for `api_key` and `bearer`, activation for `none`, and an OAuth grant import that takes no endpoints from the caller (port of `OAuthClientForImport`). `POST /v1/agents/connections/{id}/validate` runs `Resolver.Resolve` then `Source.Discover` and stores `cached_tools` with digests. `GET /v1/agents/connections/{id}/tools` returns the cache.
- **Scope.** Handlers, OpenAPI and Go SDK regen, tests.
- **Out of scope.** Scope check against tool needs (T31).
- **Dependencies.** T16, T10, T14.
- **Acceptance.** Credential material is write-only: no response ever carries it; a stale `expected_revision` returns 409; validate on a `needs_reauthorization` connection returns that status without opening MCP.

### T19. OAuth client records per app and connector · [AI-846](https://linear.app/stream/issue/AI-846)

- **Description.** Table `connector_oauth_clients` (`customer_id`, `connector_id`, `client_id`, sealed secret, auth method, `registration: operator | customer`) with `PUT` and `DELETE /v1/agents/connectors/{id}/oauth-client`. T17's authorize reads the client from this record (customer BYO) or from the operator environment (`<ID>_MCP_CLIENT_ID`), instead of per-request `oauth_client_id` and `oauth_client_secret` sealed into each grant.
- **Scope.** Migration, store, handlers, the change in T17's lookup, OpenAPI and Go SDK regen.
- **Out of scope.** White-label redirects; token export (T45); the provider app id and the managed registration method for an app the Router creates (T40).
- **Dependencies.** T16, T17.
- **Acceptance.** Rotating a customer secret touches one row and the next refresh of every connection of that connector uses it; a connector whose manifest says `registration: [operator]` refuses a customer client.

## Phase 5: agent config and session

Four PRs. After T21 an agent on staging can call a Slack or Linear tool; after T23 the old plugins are gone.

### T20. Agent config connector bindings · [AI-842](https://linear.app/stream/issue/AI-842)

**Status: merged** in [#735](https://github.com/GetStream/Vision-Agents/pull/735) (`5db5d199`), October 5. The unforced connection delete became one `UPDATE … AND NOT EXISTS` in [#746](https://github.com/GetStream/Vision-Agents/pull/746) (`8b75d33a`), October 6. Still open in [AI-889](https://linear.app/stream/issue/AI-889): a config save racing the delete, and a bind that commits while the delete waits on the row lock.

- **Description.** The `connectors[]` field on `AgentConfigRequest`, `AgentConfig` and `SyncAgentRequest`: `name` (alias), `connector_id`, `connection {type: fixed | session, connection_id}`, `tools[{name, schema_digest}]`, `required`, `timeout_ms`. Validation from the branch's `connectorBindingsComplaint` (`api/connectors.go:1147-1192`): alias pattern, no `__`, unique aliases, fixed needs a connection id, session must not carry one, digest is 64 hex chars, timeout 1 to 30,000 ms. Omission on update leaves bindings unchanged; `[]` clears them; a sync replaces them.
- **Scope.** `internal/api/configs.go`, `config_patch.go`, `sync.go`, store read and write of the column from T7, OpenAPI and Go SDK regen, tests.
- **Out of scope.** Using the bindings (T21); the `policy` object (T30).
- **Dependencies.** T7, T15.
- **Acceptance.** A config with a binding to a non-existent connector id returns 400 naming it; `GET` returns bindings exactly as written; a `fixed` binding to a user-owned connection is refused.

### T21. Session attach and the dispatcher · [AI-851](https://linear.app/stream/issue/AI-851)

- **Description.** `session.attachConnectors` from the branch (`connector_tools.go:32-227`): resolve each binding against its connection, check tenant, provider, owner (`fixed` needs `app`; `session` needs `user` equal to the verified `Spec.Caller`, never anonymous or guest), status, and the grant list; required failures stop the session, optional ones produce `connector_unavailable` events. The multi-person rule: a session with more than one verified participant exposes app-owned bindings only. Then the Dispatcher replacing `connectorToolRunner` (`mcp_tools.go`): the authority map from exposed name to (binding, connection, source, tool), the envelope (timeout from the binding, cancel on interruption with `notifications/cancelled`, size cap, `outcome_unknown` for a timed-out call), and the dispatch-time recheck of the current config and connection (`connector_tools.go:229-288`). Wire it into the `ToolRunner` chain in `manager.go:354-370`.
- **Scope.** `internal/session/connector_tools.go`, `dispatcher.go`, `spec.go` (`ConnectorBindings`, `ConnectorSelections`), `manager.go`, `session.go` events, integration tests with two users and two accounts, required versus optional, disconnect during an open session, grant removed on an open session.
- **Out of scope.** The API field on session creation (T22); plugins removal (T23).
- **Dependencies.** T12, T14, T20.
- **Acceptance.** Alice's session cannot call through Bob's connection by any input; a tool not in the grant is not exposed and not callable; a required connector that cannot open fails session creation with a clear error; the branch's `connector_tools_integration_test.go` cases pass.

### T22. Session creation with connector selections and fork revalidation · [AI-853](https://linear.app/stream/issue/AI-853)

- **Description.** `connector_bindings[{name, connection_id}]` on `POST /v1/agents/sessions`: only aliases declared as `session` may be supplied; fixed aliases cannot be overridden; the selection must belong to the verified caller. Migration `agent_sessions.connector_selections jsonb` (branch `20260929200000`) so a fork of a closed session re-resolves selections against the current config and principal.
- **Scope.** `internal/api/sessions.go`, `session/spec.go`, the migration, store, OpenAPI and Go SDK regen, the Python plugin's `SessionOptions` (`plugins/stream`), tests.
- **Out of scope.** Other SDKs (T33).
- **Dependencies.** T21.
- **Acceptance.** Supplying a fixed alias returns 400; a selection for an alias the config no longer declares is dropped on fork with an event; a stored selection never contains a credential.

### T23. Remove the plugin system · [AI-859](https://linear.app/stream/issue/AI-859)

- **Rewritten October 6.** Plugins move onto connectors first (architecture doc, «Plugins move onto connectors»), so this is the last step, not a plain drop.
- **Description.** Delete `internal/plugins`, `internal/pluginevents`, `session/plugin_tools.go`, `session/plugin_clients.go` and `attachPlugins`, the plugin operations (`listPlugins`, `listConfigPlugins`, `authorizePlugin`, `disconnectPlugin`, the plugin client operations, `getPluginLogo`, `pluginOAuthCallback`, `receivePluginEvent`) and their handlers in `api/plugins.go` and `api/handwritten.go`, the `plugin_authorization` attachment once T59 replaces it, and the SDK surfaces that reference them. One migration drops `agent_plugin_connections`, `agent_plugin_clients`, `agent_plugin_event_subscriptions`, `agent_plugin_event_deliveries` and the columns `agent_configs.agent_plugins`, `user_plugins`, `plugin_events`.
- **Scope.** Deletions, one migration, OpenAPI and Go SDK regen, the Python plugin's `folder.py` and `config.py` plugin fields, `.claude/skills/plugin/SKILL.md`.
- **Out of scope.** Any new behavior; moving rows (T61).
- **Dependencies.** T21, T22 (an agent always has a tool path), T59, T60, T61 (rows moved), and Volt on connector endpoints.
- **Acceptance.** `grep -rn plugin_id acceleration/` finds nothing outside the migration's down block; the OpenAPI freshness test passes; `router plugins migrate` reported 0 unmapped rows on staging before the drop.

### T58. Manifests for the plugins with no connector

- **Description.** Built-in manifests for the 7 catalog plugins with no connector: shopify, sentry, hubspot, google_calendar, google_drive, google_docs, gmail. Each copies the plugin's endpoints, scopes, setup steps and client env from `internal/plugins/plugins.yaml` and checks them against the vendor page, as T32 did. Google's three share one OAuth client env.
- **Scope.** `internal/connectors/providers/*.yaml`, the consent test per manifest.
- **Dependencies.** T4, T32 (pattern).
- **Acceptance.** Every plugin id in `plugins.yaml` has a connector id; each new manifest loads and its consent test passes against the fake provider.

### T59. Login in the chat for connectors

- **Description.** When a tool call needs the person's own connection (a `session` binding with none connected, or one in `needs_reauthorization`), the reply carries an authorization attachment with T17's launch URL, as the plugin's `plugin_authorization` does today (`conversation/authorizations.go`). After the callback the agent carries on with the request, as `8e450afd` does for plugins.
- **Scope.** `conversation`, `session`, the attachment schema, OpenAPI and Go SDK regen.
- **Dependencies.** T17, T21, T22.
- **Acceptance.** «Tell Nash a joke on Slack» with no Slack connection shows the attachment; after consent the agent sends the message with no second ask.

### T60. MCP Events over connections

- **Description.** Port the plugin MCP Events client (`internal/plugins/events.go`, `internal/pluginevents`): subscriptions keyed by connection, deliveries checked with the subscription's own Standard Webhooks secret, on the MCP source (T14). The deliveries endpoint stays separate from provider events (T26), which verify with the provider app's secret.
- **Scope.** The events client, store tables keyed by connection, the deliveries route, tests.
- **Dependencies.** T14, T21.
- **Acceptance.** An event from a connected MCP server reaches the agent; a delivery signed with another subscription's secret is refused.

### T61. Move plugin rows onto connectors

- **Description.** `router plugins migrate`, a Go command, not SQL: sealing needs the keyring, and the AAD binds connection id and revision (T8). It writes each `agent_plugin_connections` row as an `oauth2_code` connection (owner `app` when `user_id` is empty, else `user`) with its tokens sealed; each `agent_plugin_clients` row as a `connector_oauth_clients` record, the most recently updated one when two configs of one app differ; each `agent_plugins` entry as a `fixed` binding and each `user_plugins` entry as a `session` binding. It is idempotent and reports every row it cannot map. Event subscriptions are not moved; T60 re-creates them.
- **Scope.** The command, a dry-run flag that prints the plan, integration tests on a copy of plugin rows.
- **Dependencies.** T58, T20, T19, T40, T8.
- **Acceptance.** A dry run on staging lists every row and its target; a real run then leaves every migrated agent's tools working through connectors with no new login; a second run changes nothing. Staging row counts are `unverified` now (0 on October 1); count them first (architecture doc, «Plugins move onto connectors»).

### T24. Scheme oauth2\_client\_credentials · [AI-847](https://linear.app/stream/issue/AI-847)

- **Description.** One package: `Begin` is non-interactive, `StoredCredentials hold client id and sealed secret, Retrieve` posts `grant_type=client_credentials` to the manifest's token endpoint and caches until expiry, `Classify` and `Wrap` reuse the OAuth helpers. Add the Salesforce manifest's `oauth2_client_credentials` entry and a fake-provider personality.
- **Scope.** `schemes/oauth2cc`, its `SchemeContract` run, the Salesforce manifest line.
- **Out of scope.** `oauth2_jwt_bearer`.
- **Dependencies.** T11, T12.
- **Acceptance.** The diff touches nothing under `core/`, `api/` or `session/`; the contract suite passes; an app-owned Salesforce connection resolves a token without a browser.

### T25. Source http: operations defined as data · [AI-852](https://linear.app/stream/issue/AI-852)

- **Description.** `sources/http`: a connection's manifest or custom definition lists operations (`name`, `description`, `method`, `path` template, parameter mapping to path, query, header or body, `body: json | form`, response filter); `Discover` returns them as `ToolSpec` with digests; `Open` runs them through `ResolvedBinding.Transport` with the same envelope. A Twilio-shaped `POST .../Messages.json` with a form body is the test case.
- **Scope.** The source package, `SourceContract` run, manifest schema extension for `operations`.
- **Out of scope.** `openapi` source; `provided_arguments`.
- **Dependencies.** T14, T13.
- **Acceptance.** No change in `core/`; a private base URL is refused; a form-encoded body reaches the fake server with the credential applied last.

### T26. Events endpoint, verifier registry and the first verifier · [AI-848](https://linear.app/stream/issue/AI-848)

- **Description.** `POST /v1/agents/connectors/events/{connector_id}` with no API auth: the one inbound handler for any provider event. The `Verifier` registry; `signals/hmacheader` (header name, algorithm, encoding and secret source from the manifest). A verifier returns a VerifiedEvent (T37): grant signals, messages, and a URL-verification challenge that the endpoint answers with 200. Grant signals (`Revoked`, `Uninstalled`, `Rotated`) go to `Resolver.Invalidate` by account id; a message goes to the channel bridge hook, which logs and drops it until T57 lands. T38 adds a second route to the same handler, `POST /v1/connectors/events/{provider_app_id}`, for connectors with a provider app for each customer. First mapping: Slack `tokens_revoked`.
- **Scope.** Handler, registry, one verifier, Slack manifest `signals` block, tests with a signed and an unsigned payload.
- **Out of scope.** `jwt_set` (Google RISC), `twilio_signature`; the route for each provider app (T38); the channel bridge (T57).
- **Dependencies.** T12, T16, T37.
- **Acceptance.** An unsigned or stale payload returns 401 and changes nothing; a valid `tokens_revoked` moves the matching connection to `needs_reauthorization` and the next resolve fails fast; a validly signed message event is accepted and handed to the bridge hook, so the bridge can register without a change to the handler.

### T27. Step-up and admin consent attempts · [AI-854](https://linear.app/stream/issue/AI-854)

- **Description.** `Attempt{Kind: step_up}` created when a dispatch returns `ScopeRequired`, carrying the scope union or the claims challenge; a `connector_scope_required` session event with the authorize URL (the code today never returns it; design doc `connector-design.md:389`); `Attempt{Kind: admin_consent}` for Microsoft. Re-consent keeps the old grant until the new one succeeds.
- **Scope.** Resolver outcome handling, attempt kinds, session event, T17 handler branch, tests against T5's 403 and claims personalities.
- **Out of scope.** Interactive per-call approval.
- **Dependencies.** T17, T21.
- **Acceptance.** A 403 `insufficient_scope` during a session produces one event and no retry storm; after consent the same call succeeds without a new session.

### T28. Rate limits from the provider · [AI-855](https://linear.app/stream/issue/AI-855)

- **Description.** `RateLimitRule{per}` from the manifest, `Classify` already returns `RateLimited` with `RetryAfter`; add a `Limiter` keyed as the rule says (app, workspace, org, user) backed by Redis, a `connector_rate_limited` tool result for the model, and no blind retry.
- **Scope.** `core/limiter.go`, dispatcher branch, tests.
- **Out of scope.** Accelerate's own quotas.
- **Dependencies.** T12, T21.
- **Acceptance.** After a 429 with `Retry-After: 30`, calls on the same key are refused locally for 30 s with the reason, and other keys are unaffected.

### T29. Invocation log and dependents · [AI-856](https://linear.app/stream/issue/AI-856)

- **Description.** Two PRs if the first grows past a screenful. (a) An `connector_invocations` row per call: binding, connection, tool, latency, `error_type` from the set `customer_auth`, `external_server`, `client_timeout`, `outcome_unknown`, `denied`, and `GET /v1/agents/connections/{id}/invocations` paged by cursor (read the `pagination` skill first). (b) `used_by` on the connection response and `DELETE /v1/agents/users/{id}/connections` for offboarding and GDPR, hard-deleting `owner_id` and `account_id`.
- **Scope.** Migration, store, dispatcher hook, handlers, OpenAPI and Go SDK regen.
- **Out of scope.** Audit of grants and consents (a later PR).
- **Dependencies.** T21, T16.
- **Acceptance.** Every dispatched call leaves exactly one row; incognito sessions leave no arguments or results; the user delete removes all of a user's connections and the next session for that user attaches none.

### T30. Policy envelope on the binding · [AI-857](https://linear.app/stream/issue/AI-857)

- **Description.** Additive `policy` object on a binding: `pre_speech`, `on_interrupt: cancel | wait`, `async`, `cancellable`, `read_only`. The dispatcher honours `on_interrupt` and `cancellable`; the agent speaks `pre_speech` through the existing tool-started path.
- **Scope.** Config schema, dispatcher, agent hook, OpenAPI and Go SDK regen, tests.
- **Out of scope.** Per-tool policy (keep it per binding first).
- **Dependencies.** T21.
- **Acceptance.** A `wait` binding finishes its call after an interruption, as LiveKit does; a `cancel` binding sends the cancel, as today.

### T31. Scope check on connect · [AI-858](https://linear.app/stream/issue/AI-858)

- **Description.** `needs_scopes` on a `ToolSpec` (from the manifest per tool, or a provider's `_meta` when present); validate compares `granted_scopes` with the union over granted tools and reports `connector_scope_required` with the missing scopes.
- **Scope.** Manifest field, validate handler, tests.
- **Dependencies.** T18.
- **Acceptance.** A Slack connection granted `chat:write` only, with a tool that needs `channels:read`, validates as `needs_scopes` and names the scope.

### T32. Built-in manifests for the other five providers · [AI-838](https://linear.app/stream/issue/AI-838)

**Status: merged** in [#732](https://github.com/GetStream/Vision-Agents/pull/732) (`6b309464`), October 5.

- **Description.** Calendly, Cal.com, GitHub, Gong and Salesforce as YAML under `providers/`, from the branch's `connectors.yaml` and the design's stress-test findings; the Shopify callback hook `shopify.callback_hmac` if Shopify returns.
- **Scope.** YAML, fixtures, `ManifestContract` runs. No Go outside `providers/`.
- **Dependencies.** T6, T9.
- **Acceptance.** All load at startup as revision 1; `TestHooksAreRegistered` passes; each has a recorded fake-provider consent test.

### T33. SDK parity · [AI-861](https://linear.app/stream/issue/AI-861)

- **Description.** One PR per SDK (JS, Python plugin, Swift, Kotlin, Dart, .NET, Ruby, Rust, PHP) regenerating types from the spec and exposing `connectors` on config and `connector_bindings` on session creation. Per AGENTS.md this runs periodically after Go is stable; each SDK's own skill says how.
- **Scope.** Generated types plus the thin wrappers; language test suites.
- **Dependencies.** T22, T23.
- **Acceptance.** `npm run types --check` and the equivalent parity checks pass; no SDK exposes a credential field.

## Phase 7: connector layer for channels and direct calls

Eleven PRs in the connector layer, layer 1 of the [channels doc](channels.md): the manifest `channel` block, the verifier result, the events endpoint for each provider app, provider app records, the proxy, token export, raw event forwarding and audit. They stay sub-issues of [AI-816](https://linear.app/stream/issue/AI-816/basic-connectorsmcp-support). The layers above them, the channel bridge and the omni-channel conversation, have their own parent issues in the next two sections. **Required** marks what the first channel cannot ship without; **Proposal** marks the rest. The design behind them is «Decisions, 2026-10-05» in the architecture doc, a proposal until Thierry confirms it.

| Wave | Connector layer (AI-816) | Channel bridge ([AI-866](https://linear.app/stream/issue/AI-866)) | Omni-channel conversation ([AI-867](https://linear.app/stream/issue/AI-867)) |
| --- | --- | --- | --- |
| 1 |  |  | T49 |
| 2 | T37 |  |  |
| 3 | T34 |  |  |
| 5 | T39 |  | T43 |
| 8 | T40 |  |  |
| 9 | T38, T54 | T57 |  |
| 10 | T46 |  | T41 |
| 11 |  | T35, T36, T51, T52, T53 | T48, T55, T56 |
| 12 | T47 |  | T42 |
| 13 | T44, T45 |  |  |
| 14 | T50 |  |  |

Waves follow the same rule as the chart above: one more than the deepest dependency, with T26 now after T37. T37 depends on T3 only, so it can start now.

### T37. Verifier result carries a message · [AI-868](https://linear.app/stream/issue/AI-868)

**Status: merged** in [#744](https://github.com/GetStream/Vision-Agents/pull/744) (`65449473`), October 6. `Verify` returns `VerifiedEvent{Signals, Messages, Challenge}`. Each `InboundMessage` carries a `ProviderMessageID`, by which the channel bridge drops retried deliveries (T57).

- **Required, do first.** Description. Change `core.Verifier.Verify` to return grant signals and messages: a result with `[]Signal` and an optional inbound message (provider unit id, thread key, author, text, raw body). Today `Verify` returns `[]core.Signal` only (`[core/signal.go:5-27](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/signal.go#L5-L27)`). No implementation exists yet, so the change costs least now.
- Scope. `core/signal.go`, doc comments, `core/AGENTS.md`, a contract test.
- Out of scope. The endpoint (T26); any verifier; the manifest `channel` block (T34).
- Dependencies. T3 (merged).
- Unblocks. T26, T34.
- Acceptance. `core` compiles; `TestCoreNamesNoProvider` still passes; a fake verifier in a test returns one signal and one message.

### T34. Manifest channel block · [AI-860](https://linear.app/stream/issue/AI-860)

- **Required.** Description. `core.Manifest` gets a `channel` block beside `sources`. It holds the verifier kind and parameters (header, algorithm, signed bytes: the body, or the timestamp and the body; or a shared-secret header), the routing key path (for example `phone_number_id`), the thread key path, the author and text paths, the reply endpoint and body template, and the outbound policy (the WhatsApp 24-hour window and its template path). A connector has `sources`, `channel` or both: Linear has only `sources`, Telegram and Linq only `channel`, Slack and WhatsApp both. Parsing, validation and template checks follow T4. The bridge that uses the block is T57.
- Scope. `core/manifest.go`, validation, fixtures in `core/testdata/manifests` for the Slack bot, Linq, Telegram and WhatsApp shapes, `core/AGENTS.md`.
- Out of scope. The channel bridge (T57); real provider manifests (T35, T36, T51 to T53).
- Dependencies. T4 (merged), T37.
- Unblocks. T39, T57, T35, T36, T51 to T53.
- Acceptance. A manifest with only a `channel` block loads; an unknown verifier kind, or a reply template that names an undeclared input, is refused with the field named; `TestCoreNamesNoProvider` still passes.

### T38. Events endpoint for each provider app · [AI-869](https://linear.app/stream/issue/AI-869)

- **Required.** Description. `POST /v1/connectors/events/{provider_app_id}` (proposal), a second route into T26's handler. The URL names the customer's provider app, and with it the tenant and the signing secret, so the Router needs no global lookup by account. Token signals go to `Resolver.Invalidate`; messages go to the channel bridge (T57). Slack sends `tokens_revoked` and `app_uninstalled` to the app's Request URL ([tokens\_revoked](https://docs.slack.dev/reference/events/tokens_revoked), [app\_uninstalled](https://docs.slack.dev/reference/events/app_uninstalled)), so one URL serves signals and messages. T26's route for each connector stays for connectors without a provider app for each customer.
- Scope. Route, provider-app lookup, verifier with that app's secret, tests.
- Out of scope. Shared webhooks (T39).
- Dependencies. T26, T40.
- Unblocks. T35, T36, T52.
- Acceptance. An event signed with app A's secret on app B's URL is refused; a valid event reaches the bridge with app A's customer; a valid `tokens_revoked` on the same URL invalidates the connection.

### T39. Routing index for shared webhooks · [AI-870](https://linear.app/stream/issue/AI-870)

- **Required for WhatsApp.** Description. When the provider fixes one app for all customers (Meta Tech Provider), the Router finds the customer by a routing key from the event. Add an index on `(connector_id, account_id)` and a lookup by the `channel` block's routing key path, for example `phone_number_id`. Today the only index is `connector_connections_owner_idx` (`[migrations/20261002193000_connector_connections.sql:68-70](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/migrations/20261002193000_connector_connections.sql#L68-L70)`).
- Scope. Migration, store query, tests.
- Dependencies. T7 (merged), T34.
- Unblocks. T51.
- Acceptance. A lookup by routing key returns one connection; a second connection with the same key for another customer is refused at write time.

### T40. Provider app record for each customer · [AI-871](https://linear.app/stream/issue/AI-871)

- **Required for Slack.** Description. One record for each (`app_pk`, connector): provider app id, `client_id`, owner `stream` or `customer`, and who created it. It extends T19's `connector_oauth_clients` with the provider app id and a new client registration method, `managed`: an app the Router created for this customer (T54). The secrets, `client_secret` and `signing_secret`, live in a `store.ConnectorConnection` with owner `app` and are read through `core.Resolver`, sealed with the same `auth.Sealer` and keyring as every connector secret (architecture doc, decision 7). The shared Stream app (registration `operator`) serves only Stream's own agents, so Athena's first Slack channel runs on a Stream-owned record before T54 exists.
- Scope. Migration or T19 extension, store, the `managed` registration method, tests.
- Out of scope. Creating the app at the provider (T54).
- Dependencies. T19, T11.
- Unblocks. T35, T38, T45, T46, T54.
- Acceptance. A second record for the same (`app_pk`, connector) is refused; an event signed with record A's signing secret verifies only on A; a connector whose manifest lists only `operator` refuses a `managed` record.

### T54. Slack app for each customer with apps.manifest.create · [AI-872](https://linear.app/stream/issue/AI-872)

- **Proposal.** Description. The Router creates the customer's Slack app with [`apps.manifest.create`](https://docs.slack.dev/reference/methods/apps.manifest.create) from a manifest template: scopes, events, `request_url` = the T38 URL, `token_rotation_enabled: true`, optional `allowed_ip_address_ranges` (at most 10). It needs an app configuration token that a workspace admin gives once; `tooling.tokens.rotate` keeps it alive. The app stays in the customer's workspace without public distribution, so Slack's non-Marketplace rate limit does not apply. The built-in `slack.yaml` gets a new revision with `managed` in `client.registration`; today, at revision 3, it lists `[operator]` only.
- Scope. The Slack create and delete calls, config-token storage and rotation, `slack.yaml` revision 2, tests against a fake Slack.
- Out of scope. The Stream-owned app for Stream's own agents (T40).
- Dependencies. T40.
- Unblocks. Slack for customers in integration modes B and C.
- Acceptance. Connecting Slack for `app_pk` X creates one Slack app named for X; its install uses its own client; deleting the connector calls `apps.manifest.delete`.

### T44. Proxy for direct calls · [AI-873](https://linear.app/stream/issue/AI-873)

- **Proposal, the default direct-call path.** Description. `ANY /v1/agents/connections/{id}/proxy/{path}`, server-side only, beside T16's connection endpoints. Base URL and allowed hosts come from the manifest. The Router resolves the credential (T12), wraps it (T13), forwards the request unchanged and returns the response unchanged. On a 401 it calls `Invalidate` and retries once. A rate limit for each customer; the provider's `429` and `Retry-After` pass through. A user connection works only in that user's session (`Binding.Selection: session`). The customer uses the provider's official SDK with its base URL set to the proxy (`base_url` in `slack_sdk`, `slackApiUrl` in `@slack/web-api`). The Router implements no provider method.
- Scope. Handler, rate limiter, OpenAPI and Go SDK regen, tests with the fake provider.
- Out of scope. Token export (T45).
- Dependencies. T12, T13, T16, T47.
- Unblocks. Integration modes B and C.
- Acceptance. A host outside the manifest is refused; the provider receives the credential and the original body; a user connection outside its session is refused.

### T45. Token export · [AI-874](https://linear.app/stream/issue/AI-874)

- **Proposal, opt-in.** Description. An optional `Export` on `core.Scheme`, for bearer schemes only: `core.AccessCredential` keeps its secret private today (`[core/scheme.go:115-130](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/connectors/core/scheme.go#L115-L130)`). `POST /v1/agents/connections/{id}/token`, server-side only. Only the customer's own provider unit exports; export is off by default for each connector and refused when the app restricts tokens to the Router IP ranges. A bot token for `Binding.Selection: fixed`; a user token only in that user's own session. One audit row for each export.
- Scope. Scheme extension, `oauth2code` and `apikey` export, handler, OpenAPI and Go SDK regen.
- Dependencies. T12, T40, T47.
- Acceptance. A Stream-owned app refuses export; each export writes one audit row; no response ever contains a refresh token.

### T46. Raw event forwarding · [AI-875](https://linear.app/stream/issue/AI-875)

- **Proposal.** Description. Event destinations for each connector. The bridge forwards the events it does not handle (buttons, reactions, modals), or every event in integration mode C. Each forward is signed with a key of that customer or destination, not with the deployment secret. The raw provider body and headers stay as they are, so Slack Bolt can verify with the app's signing secret when the customer owns the app.
- Scope. Destination records, signer, retries on 5xx, tests.
- Dependencies. T57, T40.
- Acceptance. A forward signed for customer A does not verify with customer B's key; a 503 is retried.

### T47. Audit of token requests, proxy calls and grants · [AI-876](https://linear.app/stream/issue/AI-876)

- **Required before T44 and T45.** Description. One row for each proxy call, token export, and grant created, refreshed or revoked, with correlation ids, like the Observability tab of Vercel Connect. T29 lists «Audit of grants and consents (a later PR)» as out of scope; this is that PR.
- Scope. Migration, writes from the resolver and the proxy, a paged read endpoint (pagination skill first).
- Dependencies. T29.
- Acceptance. Each proxy call and each export leaves exactly one row; incognito sessions leave no arguments.

### T50. SDK parity for direct-call and event endpoints · [AI-877](https://linear.app/stream/issue/AI-877)

- **Required by AGENTS.md.** Description. T38, T44, T45 and T46 regenerate `openapi.yaml` and the Go SDK in their own PRs. The other SDKs follow in parity PRs, as in T33. No SDK wraps a provider SDK: customers use the provider's official SDK pointed at the proxy.
- Dependencies. T44, T45, T46.

## Channel bridge and inbound channels

Six PRs in the Router above the connector layer: the channel bridge and the first channels. Parent issue: [AI-866](https://linear.app/stream/issue/AI-866). The bridge writes each external thread into its own thread channel in Stream Chat and Router's existing message hook (`internal/api/messagehooks.go`) wakes the session; replies go back through the connector layer. Each customer has its own provider unit: a Slack app, a WABA with a number, a bot, a vendor account. Slack comes first: Thierry on September 17, «For our own sovereign ai we would need good slack integration».

### T57. Channel bridge core · [AI-878](https://linear.app/stream/issue/AI-878)

- **Required.** Description. The inbound half maps a verified message (T26, T37) to a provider unit, an external thread and an author, and writes it to the thread channel as the person, without `source`. Router's message hook then wakes or starts the session. The outbound half takes a reply (`source: agent`) in a linked thread channel, which the message hook hands over, resolves the credential (T12) and sends it through the reply endpoint and body template of the manifest `channel` block (T34). Store `channel_threads`: external thread ↔ thread channel cid, with `stream_app_pk`. Retried deliveries (same ProviderMessageID, T37) and the bot's own messages are dropped. An external author maps to a Stream Chat user.
- Scope. The bridge package, the store table, the message-hook hand-off, a fake-provider channel personality, tests.
- Out of scope. Any real provider (T35, T36, T51 to T53); episode cards (T41).
- Dependencies. T26, T34, T12.
- Unblocks. T35, T36, T41, T46, T51 to T53.
- Acceptance. The fake provider posts a message event; the bridge writes it to a new thread channel without `source`; the message hook wakes a text session; the session's reply reaches the bridge and leaves with the connection's credential; a second event on the same thread uses the same thread channel.

### T35. Slack channel · [AI-862](https://linear.app/stream/issue/AI-862)

- **Required.** Description. Slack Events API on the customer's provider app (T40; for Athena the Stream-owned app): acknowledge within 3 seconds, honour `x-slack-retry-num`, channel + `thread_ts` as the thread key, write into the thread channel, reply with `chat.postMessage` and the bot token. A built-in manifest for the bot token (owner `app`, identity = the Slack team) sits beside the user-token tool manifest `slack.yaml`, unless the hosted Slack MCP accepts a bot token (`unverified`). Athena's first scenario is this channel.
- Scope. The Slack bot manifest with its `channel` block, Slack verifier parameters, tests with recorded Slack events.
- Out of scope. Slack as a tool (T9 to T21); personal tokens in a shared channel; creating the customer's Slack app (T54).
- Dependencies. T57, T38, T40, T41.
- Unblocks. The first channel in a real workspace.
- Acceptance. A mention in a channel with three people produces one thread channel and one episode card, and the session calls only app-owned tools; a retried delivery is deduplicated; the reply lands in the same thread.

### T36. Linq channel (iMessage, BYO account: unverified) · [AI-863](https://linear.app/stream/issue/AI-863)

- **Proposal.** Description. iMessage through the customer's own Linq account. The Linq webhook points to the events endpoint of the customer's provider app (T38). The verifier is HMAC-SHA256 over `{timestamp}.{rawBody}` with the subscription's signing secret, compared with the `X-Webhook-Signature` header ([Linq webhooks](https://docs.linqapp.com/guides/webhooks/index.md)). The bridge writes each conversation into its own thread channel and one episode card into the person's omni-channel (T41); number ↔ number is the thread key, and the contact map joins it by E.164. The reply goes out through the send endpoint from the manifest `channel` block with the `api_key` scheme (T11). BYO is still `unverified` as a decision: asked on October 1 whether the customer brings their own Linq key, Thierry answered «well thats we have to figure out». No voice: Linq's API places no calls.
- Scope. Linq manifest (`api_key` scheme, `channel` block with the verifier parameters, the thread key and the reply template), tests against the fake provider.
- Out of scope. Apple Messages for Business, open only through an Apple-approved MSP; Stream-operated lines.
- Dependencies. T57, T38, T41, T11.
- Unblocks. iMessage as an inbound channel.
- Acceptance. An inbound iMessage opens or continues one thread channel and one episode card; the reply is sent on the customer's line; a payload with a bad signature or a stale timestamp is refused before the bridge runs.

### T51. WhatsApp channel (Meta Cloud API) · [AI-879](https://linear.app/stream/issue/AI-879)

- **Proposal.** Description. One Stream Meta app as Tech Provider. The customer onboards its WABA and number with Embedded Signup; the Router exchanges the code for a business token. Verifier: `X-Hub-Signature-256` with our app secret. Routing by `phone_number_id` (T39). Outbound policy: free text inside the 24-hour window, approved templates outside it; the WhatsApp idle period is shorter than 24 hours.
- Dependencies. T57, T39, T41, T43.

### T52. Telegram channel · [AI-880](https://linear.app/stream/issue/AI-880)

- **Proposal.** Description. The customer creates a bot in BotFather and gives its token (scheme `apikey`). `setWebhook` to the T38 URL with `secret_token`; verifier on `X-Telegram-Bot-Api-Secret-Token`; `chat_id` as the thread key; account linking for identity.
- Dependencies. T11, T57, T38, T41.

### T53. SMS channel (Twilio, Telnyx) · [AI-881](https://linear.app/stream/issue/AI-881)

- **Proposal.** Description. The customer's number or account at the vendor; scheme `apikey`; the vendor's signature verifier; number ↔ number as the thread key; contact map by E.164.
- Dependencies. T11, T57, T41, T43.

## Omni-channel conversation

Seven PRs that give the agent one history across channels. Parent issue: [AI-867](https://linear.app/stream/issue/AI-867). Each external thread keeps its messages word for word in a thread channel, a call keeps its call channel, and the person's omni-channel gets one episode card for each call or text thread. The agent reads its own thread word for word and other episodes through their cards.

### T43. Contact map · [AI-882](https://linear.app/stream/issue/AI-882)

- **Required for phone-number channels.** Description. Table (customer, agent, E.164 number) → omni-channel cid. Channel ids hold no raw number. SMS, WhatsApp, iMessage and calls look it up and set `ConversationID`; the session API accepts `conversation_id` (`[api/sessions.go:513](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/sessions.go#L513)`). Slack and Telegram need account linking (later).
- Scope. Migration, store, E.164 normalization, lookups in the bridge and the call path.
- Dependencies. T7 (merged).
- Unblocks. T41, T51, T53.
- Acceptance. The same number from SMS and from a call maps to one omni-channel; a new number creates one row.

### T41. Thread channel and episode card · [AI-883](https://linear.app/stream/issue/AI-883)

- **Required for channels.** Description. On the first message of an episode the bridge creates one episode card in the person's omni-channel: `source` (`sms`, `whatsapp`, `slack`, `imessage`), `status: in_progress`, `started_at`, `thread_channel`. A call writes a card with `source: call` and `call_id` when its session starts, linked to the call channel `agent:<call id>`. The card stays one message, so an episode takes one place in the 200-message history window (`[chatlog/reader.go:13-16](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/reader.go#L13-L16)`).
- Scope. Card writes from the bridge and the call path, the episode store, tests.
- Out of scope. Closing and summarizing (T55); reading cards (T56); the contact map (T43).
- Dependencies. T57, T43.
- Unblocks. T35, T36, T48, T51 to T53, T55, T56.
- Acceptance. Three SMS in a row make one card and one thread channel; a call makes one card with `source: call` and a link to its call channel; a card write starts no session, because the card has `source`.

### T55. Episode close, summary and memory · [AI-884](https://linear.app/stream/issue/AI-884)

- **Required for channels.** Description. A text episode closes after an idle period, a setting (shorter than 24 hours for WhatsApp); a call episode closes on `call.session_ended`. The Router sets `status: ended` with `UpdateMessagePartial`, which sends no webhook (`[chatlog.go:551-556](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/chatlog/chatlog.go#L551-L556)` uses the same call). The LLM writes the summary into the card (`status: summarized`, or `summary_failed`), and the facts go to memory, scoped by the person from the contact map.
- Scope. Idle timer, close trigger, summary job, memory write, tests.
- Dependencies. T41.
- Acceptance. After the idle period the card holds a summary; a failed summary leaves `summary_failed` and the raw thread intact; a card update triggers no message hook.

### T56. Session reads the episode cards · [AI-885](https://linear.app/stream/issue/AI-885)

- **Required for channels.** Description. A text session reads its own thread channel word for word and the person's episode cards as extra context: `summarized` → the summary; `in_progress`, `ended` or `summary_failed` → the last lines of `thread_channel`. This covers an SMS that arrives seconds after a call, before its summary.
- Scope. Session context build, tests.
- Dependencies. T41.
- Unblocks. T42.
- Acceptance. A session on a new SMS thread sees the card of an earlier Slack thread; an SMS 10 seconds after a call reads the call's last raw lines while its summary is not ready.

### T42. Voice session reads the history · [AI-886](https://linear.app/stream/issue/AI-886)

- **Required for calls in the omni-channel.** Description. Today only a persistent text conversation loads history; voice is refused with «persistent conversations require text mode» (`[session/manager.go:229-231](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/session/manager.go#L229-L231)`). A voice session reads the person's episode cards as T56 does.
- Scope. Session manager, context build, tests.
- Dependencies. T56.
- Acceptance. A call after an SMS thread starts with the SMS card in its context.

### T48. Tenancy: app pins and a message hook in each customer app · [AI-887](https://linear.app/stream/issue/AI-887)

- **Required after nash/project-tenancy merges.** Description. The omni-channel and the thread channels live in the customer's Stream app. The episode, the thread link and the provider app keep `stream_app_pk`, as the branch does for sessions and calls ([`migrations/20261002120000_stream_app_pins.sql`](https://github.com/GetStream/Vision-Agents/blob/8436b0b36c6848fdd539044a991dbba7dae524bc/acceleration/migrations/20261002120000_stream_app_pins.sql) on `nash/project-tenancy` @ `8436b0b3`). The Router sets its message hook in each customer app: on the branch `PointMessageHook` runs only for the deployment app ([`cmd/phone/main.go:568`](https://github.com/GetStream/Vision-Agents/blob/8436b0b36c6848fdd539044a991dbba7dae524bc/acceleration/cmd/phone/main.go#L568)). The customer's Stream app key stays in the `stream_apps` table of nash/project-tenancy; the customer's Slack app secrets stay in `store.ConnectorConnection` (owner `app`). Both seal with the same `auth.Sealer` and keyring.
- Note. The branch's six migrations are older than the nine on accelerate, three of them connectors. Renumber at merge.
- Dependencies. nash/project-tenancy merged, T40, T41.
- Acceptance. A card written after the idle period lands in the app pinned at the first message; a message in a customer app's thread channel wakes the Router.

### T49. Session API: history from the caller · [AI-888](https://linear.app/stream/issue/AI-888)

- **Proposal, integration mode C.** Description. A field to pass the caller's own history when a thread outlives a session. Neither `CreateSessionRequest` nor `CreateResponseRequest` has one today (`[api/generated.go:2027,2040](https://github.com/GetStream/Vision-Agents/blob/ead4a273f4d3623fff2a2286d5422725aa0af2e2/acceleration/internal/api/generated.go#L2027)`). Until then mode C uses a text session with `incognito: true`.
- Dependencies. None.

## Sources

- [Accelerate connectors: architecture design](architecture.md): the keep, change and add lists, the one-way doors, the validation plan this decomposition follows.
- [Voice-agent connectors: competitor analysis](competitor-analysis.md): Work plan P0 and P1 items, «Where plugins run today» (0 plugin rows on staging).
- Branch [`codex/connector-support`](https://github.com/GetStream/Vision-Agents/tree/codex/connector-support) at `cf62af0d`: the code copied in T1, T2, T8, T9, T14, T17, T21 and the migrations named by date.
- `accelerate` at `8771b8bb`, checked October 1: `internal/auth/secret.go:14` (`KEKVersion = 1`), `internal/plugins/`, `api/legacy.yaml:1061-1177`, `internal/api/policies.go` (Huma pattern), `cmd/openapi/main.go`, `.github/workflows/ci.yml:78-98`.
- `AGENTS.md`: Huma for new operations, no additions to `legacy.yaml`, Go first for SDK changes, no mocks, the `pagination` and `go-testing` skills.
- [Connectors: inbound channels, tools and the omni-channel conversation](channels.md) — the source for T34 to T53: channel bridge, thread channel and episode card, integration modes, proxy and token export, other channels, tenancy. Read on October 5.
- [Vercel Connect и наши connectors: в чём разница](https://claude.ai/code/artifact/b1467e6d-f4f7-4f5e-bb77-3f862c19e608) — the comparison behind T40, T44, T45 and T47: an app for each customer, `getToken`, triggers, observability. Read on October 5.
