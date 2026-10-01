# AI-816 connectors: PR-sized subtasks

Oct 1, 2026 · @Kanat Kiialbaev

Exported from Claude Docs on 2026-10-01 (https://claude.ai/code/artifact/e4d97114-1334-4181-beab-c5721909c201). The Claude Doc is the source of truth; this copy is a snapshot.

33 subtasks in 7 phases, each one PR, in the order they can land. They implement [Accelerate connectors: architecture design](architecture.md) on top of the code that exists today. Each subtask is a Linear sub-issue of [AI-816](https://linear.app/stream/issue/AI-816/basic-connectorsmcp-support); its number follows the title.

## Ground rules

- **Base branch is `accelerate`.** Every PR targets it. The prototype branch `codex/connector-support` is a source of code to copy where the design says «keep», not a base: it changes 197 files at once, carries SDK regenerations for ten languages and an irreversible plugin migration, none of which is reviewable as one PR.
- **What `accelerate` has today** (checked October 1): `internal/plugins` with five catalog entries and endpoints in `api/legacy.yaml:1061-1177`; one-version AES-GCM sealer in `internal/auth/secret.go` (`KEKVersion = 1`); no `internal/egress`, no `internal/mcp`, no `internal/connectors`; Huma for new operations (`internal/api/policies.go` is the pattern); `cmd/openapi` renders the spec and a test fails when it is stale.
- **Layer order inside a phase:** migration and store, then core logic, then API, then session wiring, then SDKs. A PR touches one layer unless the layer is useless alone.
- **New endpoints use Huma, not `legacy.yaml`.** The prototype registered its connector endpoints through the generated oapi-codegen server (`ListConnectorsRequestObject` and friends in `internal/api/connectors.go`). AGENTS.md forbids adding to `legacy.yaml`, so the API phase ports the handlers to `huma.Register`.
- **Go first, SDKs later.** AGENTS.md: SDK changes start with Go; the other SDKs follow in their own PRs. Each API PR regenerates `openapi.yaml` and the Go SDK only.
- **Every PR keeps `go vet ./...` and `go test ./...` green** (`.github/workflows/ci.yml:78-98`), adds its own tests against real local servers and a temporary database, never mocks (AGENTS.md), and leaves the router startable with the feature off until the session wiring lands.
- **Nothing is deployed to production.** Router runs only on staging, where `agent_plugin_connections` has 0 rows (competitor doc, «Where plugins run today»). So there is no plugin import and the plugin removal is a plain drop.
- **Channels are a later layer, not part of these 33 PRs.** Thierry on October 1: «the omni channel/connector concept will be important», naming Linq and Chatbase; later that day: «AI requires a good way to integrate slack, whatsapp, rcs, texting, imessage, maybe telegram». Where a channel's transport lives is not decided (architecture doc, two-way door «Where a channel's transport lives»): adapters in Accelerate on top of connections, which is what T34 to T36 describe, or a bridge into Stream Chat where Router's existing message hook answers and Accelerate changes nothing. In this plan that means one cheap rule now: T26's inbound endpoint and verifier registry are built so a channel adapter can reuse them. T34 to T36 wait for that decision.
- **Each subtask lists:** title, description, scope, out of scope, dependencies, acceptance criteria. Titles are written for Linear.

## Order and dependencies

&#91;embedded content: PR waves · 33 scheduled PRs and 3 channel PRs in 12 waves by dependency depth\]

A wave is the earliest point a PR can start: its depth in the dependency graph, computed from each subtask's Dependencies line. PRs in one wave do not depend on each other, so two people can take two of them. The waves are a conservative schedule; a PR may start as soon as its own dependencies merged, which is often earlier than its wave. The phases below group PRs by layer; the waves group them by time. T34 to T36 (channels) are drawn dashed: they have a wave by dependency (11 and 12) but no date until a channel is scheduled and the approach (adapters in Accelerate or a Stream Chat bridge) is decided.

## Phase 0: foundations and guardrails

Five PRs with no user-visible change. They give later PRs a place to put code, a test server to run against, and a CI rule that keeps provider names out of the core from day one.

### T1. Versioned KEK keyring in auth.Sealer · [AI-828](https://linear.app/stream/issue/AI-828)

- **Description.** Extend `internal/auth/secret.go` from one key to a keyring: `ROUTER_AUTH_KEK_V<n>` plus `ROUTER_AUTH_KEK_VERSION`, with `ROUTER_AUTH_KEK` as the version-1 alias. Add `SealWithAAD`, `OpenWithAADVersion` and `CurrentVersion`. Copy from the branch (`internal/auth/secret.go`, 90-line diff). Build the sealer whenever connectors are enabled, in every auth mode: today `cmd/router/main.go:225` builds it only for `api_key`, and staging runs `proxy` (`internal/config/staging.yaml:3-4`), where `main.go:211-215` returns before a sealer exists.
- **Scope.** `internal/auth/secret.go`, its tests, the sealer construction in `cmd/router/main.go`, README variables. Existing `Seal` and `Open` keep working for API secrets.
- **Out of scope.** Rewrapping rows of any table; connector material (T8).
- **Dependencies.** None.
- **Acceptance.** A row sealed under version 1 opens after version 2 is added; a wrong AAD fails to open; missing keyring at startup fails fast with a clear message; with `auth.mode=proxy` and connectors enabled the router builds the sealer and fails fast without a keyring; with connectors off, `proxy` starts without a KEK as it does today; existing API-secret tests pass unchanged.
- **Before deploy.** Add `ROUTER_AUTH_KEK_V1` to staging's secrets, then redeploy. Staging has no KEK today: on October 1 `gcloud secrets list` over the staging prefix showed 28 secrets (names only) and none is a KEK, which fits the chart's `router.authMode: proxy`. The chart loads every secret under that prefix into the pod's environment, so it needs no change. Whoever deploys T1 runs one `the deploy CLI's secrets command --key <staging secret for ROUTER_AUTH_KEK_V1> --value-file=-`, as the launch runbook `the launch runbook` does for the other keys. Kanat can: `testIamPermissions` on the project returned `secretmanager.secrets.create` and `secretmanager.versions.add` on October 1.

### T2. Egress policy for outbound connector traffic · [AI-829](https://linear.app/stream/issue/AI-829)

- **Description.** Add `internal/egress` with `ValidatePublicHTTPSURL` (https only, no userinfo, no query string, public IPs after DNS resolution, no redirects across hosts) and `NewPublicHTTPClient`. Copy from the branch (`internal/egress/public.go`, 154 lines).
- **Scope.** The package and its tests. No caller yet.
- **Out of scope.** Wiring into any transport (T13).
- **Dependencies.** None.
- **Acceptance.** Private, loopback, link-local and metadata addresses are refused; a `javascript:` or `http:` URL is refused; a public https URL passes; the client refuses cross-host redirects.

### T3. Core contracts and the CI guard · [AI-830](https://linear.app/stream/issue/AI-830)

- **Description.** Create `internal/connectors/core` with the interfaces from the design doc: `Scheme`, `Source`, `Runtime`, `Backend`, `Resolver`, `Verifier`, `Hook`, the `Registry`, and the value types `Material`, `Credential`, `Outcome`, `Need`, `ConnectionRef`, `Signal`, `Bound`. No implementation. Add the two guard tests: `TestCoreImportsNoAdapter` (`core` imports nothing under `schemes/`, `sources/`, `backends/`, `signals/`, `providers/`) and `TestCoreNamesNoProvider` (no string literal in `core` equals a connector id or a name in the deny list).
- **Scope.** `internal/connectors/core/*.go`, doc comments, the two tests.
- **Out of scope.** Any adapter; the manifest model (T4).
- **Dependencies.** None.
- **Acceptance.** Package compiles with zero dependencies on other `internal/connectors` packages; both guard tests pass and a deliberate `case "slack"` added in a test build fails `TestCoreNamesNoProvider`.

### T4. Provider manifest model: parsing, validation, templates, capture and identity rules · [AI-832](https://linear.app/stream/issue/AI-832)

- **Description.** The `Profile` type and the manifest schema (`inputs`, `vars`, `endpoints` as `{var}` templates, `schemes`, `client`, `scopes`, `identity`, `capture`, `refresh`, `rate_limit`, `sources`, `hooks`). Pure functions: parse YAML or JSON, validate (every `{var}` is a declared input, a `vars` entry or a captured name; every hook name is a string; enums closed), resolve endpoints for a set of inputs, apply `capture` and `identity` rules to a token response, a callback query or an `id_token`.
- **Scope.** `internal/connectors/core/manifest.go` and tests. Test fixtures: the 12 stress-test manifests from the design doc under `testdata/manifests/`, with recorded token responses and callbacks.
- **Out of scope.** Storing manifests (T6); seeding; hooks implementations.
- **Dependencies.** T3.
- **Acceptance.** All 12 fixture manifests load; Salesforce resolves production and sandbox hosts from `environment`; QuickBooks captures `realm_id` from a callback query; Google reads `sub` from an `id_token`; a manifest with an undeclared `{var}` is rejected with the variable named.

### T5. Fake provider server for tests · [AI-831](https://linear.app/stream/issue/AI-831)

- **Description.** `internal/connectors/fakeprovider`: one `httptest`-based server that plays an OAuth authorization server and an MCP endpoint, with switchable personalities: rotating refresh with a grace window, non-rotating refresh, no refresh token, `invalid_grant`, lost response, `insufficient_scope` on 403, a `claims` challenge on 401, 429 with `Retry-After`, comma scopes with `authed_user`, `realmId` in the callback, a signed callback, PKCE and `iss` checks. Start from the fake-provider tests on the branch (`connector-design.md:571`).
- **Scope.** The package and a self-test per personality.
- **Out of scope.** Any production code.
- **Dependencies.** None (T4 fixtures may reuse its recorded responses).
- **Acceptance.** Each personality has a test proving it behaves as named; the server is usable from any package test with one constructor call.

## Phase 1: storage

Three PRs: the schema and the store methods, nothing that calls them yet. Migrations follow the repo's goose files under `acceleration/migrations/` (latest today: `20260930120000_agent_config_speed.sql`).

### T6. Connector definitions table with manifest and revision, seeded from YAML · [AI-833](https://linear.app/stream/issue/AI-833)

- **Description.** Migration `connector_definitions` (`customer_id`, `id`, `revision`, `name`, `category`, `description`, `manifest jsonb`, timestamps; primary key `customer_id, id, revision`; built-ins under a reserved customer id). Store: `CreateConnectorDefinition`, `ConnectorDefinition(id, revision)`, `LatestConnectorDefinition`, `ListConnectorDefinitions`. A startup seeder reads `internal/connectors/providers/*.yaml` and inserts a new revision only when the manifest changed. First two manifests: Slack and Linear, written from the branch's `connectors.yaml` plus the design's new fields.
- **Scope.** One migration, `internal/store/connectors.go` (definitions part), the seeder, Slack and Linear manifests, store integration tests.
- **Out of scope.** The API over definitions (T15); the other five providers (T32); custom definitions beyond the store method.
- **Dependencies.** T4.
- **Acceptance.** Router starts on an empty database and the two built-ins exist at revision 1; a changed YAML on restart creates revision 2 and leaves revision 1; a custom definition id must start with `custom_` and cannot shadow a built-in.

### T7. Connections, authorization attempts and the agent config column · [AI-835](https://linear.app/stream/issue/AI-835)

- **Description.** Migration `connector_connections` with the branch's columns (`20260929170000_connectors.sql`) plus the design's changes: `auth_scheme TEXT` and `tls_scheme TEXT NULL` instead of the `auth_type` CHECK, `definition_revision`, `inputs jsonb`, `metadata jsonb`, `material_sealed BYTEA`, `material_kek_version`. Migration `connector_authorization_attempts` as on the branch with `kind TEXT` and `kek_version`. `ALTER TABLE agent_configs ADD COLUMN connectors JSONB`. Store: create, get, list by owner, soft delete, `ConnectorConnectionReferenced`, and the attempt methods (create with cleanup, by state, by id, consume once).
- **Scope.** Two migrations, store methods and models, integration tests.
- **Out of scope.** The advisory lock and revisioned save (T8); any handler.
- **Dependencies.** T6, T1.
- **Acceptance.** `owner_type` is `app` with empty `owner_id` or `user` with a non-empty one, enforced by CHECK; `auth_scheme` is validated against the registry at write time, not by a CHECK; an attempt is consumed exactly once under concurrent callers; a soft-deleted connection is invisible to every read.

### T8. Locked grant backend: advisory lock, checkpoint, revision CAS, sealed material · [AI-839](https://linear.app/stream/issue/AI-839)

- **Description.** `internal/connectors/backends/pgsealed` implementing `core.Backend`: `WithLocked` with a session-level `pg_advisory_lock` on (tenant, connection), a `checkpoint` closure, and the revision compare-and-swap, copied from the branch (`store/connectors.go:198-347`). Seal and open `Material` with AAD bound to tenant, connection id and revision (from `connectors/secrets.go`).
- **Scope.** The backend package, the `SaveConnectorConnectionAtRevision` store method, integration tests ported from the branch: one committed revision under concurrent rotation, lost response leaves `needs_reauthorization` and never replays a refresh token.
- **Out of scope.** Refresh itself (T10); the resolver (T12).
- **Dependencies.** T7, T1, T3.
- **Acceptance.** `TestConcurrentCredentialResolutionCommitsOneRotatedRefreshToken` and `TestRefreshOutcomeSurvivesLostResponsesAndCanceledWorkers` pass against the new backend with a fake rotation; a stale revision save returns `ErrConnectorConnectionChanged`; material sealed under KEK v1 is rewrapped to v2 on next successful use.

## Phase 2: schemes

Three PRs. The OAuth scheme is the branch's `internal/mcp/oauth.go` (961 lines) with the provider rules taken out; it is split in two so the consent half and the refresh half are reviewed apart.

### T9. Scheme oauth2\_code, part 1: discovery, client registration, consent and exchange · [AI-834](https://linear.app/stream/issue/AI-834)

- **Description.** `internal/connectors/schemes/oauth2code` implementing `Begin` and `Complete`: RFC 9728 and 8414 discovery with manifest overrides, CIMD first then DCR, client auth `none`, `client_secret_post`, `client_secret_basic`, PKCE S256, `resource`, `state`, the authorize URL built from the `Profile` (scope separator and extra params from the manifest, not from `connector.ID == "slack"`), code exchange with the `iss` check, then `capture` and `identity` rules from T4 on the token response and callback query. Provider branches in `oauth.go:265,823-830,832-887` and the Slack and Calendly fields of `tokenResponse` are deleted, not ported.
- **Scope.** The scheme package, the public client metadata document, tests against T5 for CIMD, DCR, confidential client, denial, replayed state, `iss` mismatch, comma scopes, `realmId` capture.
- **Out of scope.** Refresh, `Classify`, `Wrap`, `Revoke` (T10); the HTTP handlers (T17).
- **Dependencies.** T3, T4, T5, T2.
- **Acceptance.** All T5 consent personalities pass; `TestCoreNamesNoProvider` still passes; the Slack manifest with `separator: ","` produces the same authorize URL the branch produced for Slack.

### T10. Scheme oauth2\_code, part 2: mint, classify, wrap, revoke · [AI-836](https://linear.app/stream/issue/AI-836)

- **Description.** `Mint` renews from `Material` using `RefreshPolicy` (margin, `send_scope`, rotating with a grace retry inside the window, `token_ttl` warning), returns the new `Material`. `Classify` maps responses to `Outcome`: `invalid_grant` and `invalid_refresh_token` to InvalidGrant, network and 5xx to Uncertain or Transient as the branch does (`oauth.go:539-595`), 403 `insufficient_scope` and a 401 `claims` challenge to ScopeRequired, 429 to RateLimited with `Retry-After`. `Wrap` sets the bearer header. `Revoke` calls the manifest's `revoke` endpoint when present and reports best effort.
- **Scope.** The same package; tests against T5 for every outcome; a `private_key_jwt` client auth method as one file, used by nothing yet.
- **Out of scope.** The lock and status transitions (T12).
- **Dependencies.** T9.
- **Acceptance.** A refresh under the grace window with the old token succeeds once; `scope` is sent only when the policy says; a lost response returns Uncertain and the caller's `Material` is unchanged; each outcome has a table-driven test.

### T11. Static schemes and the scheme contract suite · [AI-840](https://linear.app/stream/issue/AI-840)

- **Description.** `schemes/apikey`, `schemes/bearer`, `schemes/none`, each a few dozen lines: `Begin` returns Done, `Complete` seals the supplied value, `Mint` returns it with no expiry, `Wrap` sets the configured header (with the forbidden-header list from `api/connectors.go:1050-1057`). Plus `core/contracttest.SchemeContract`, a table-driven suite any scheme runs: round trip, concurrent mint commits once, no secret in URL, log or error text, `Classify` covers the six outcomes, `Revoke` is honest.
- **Scope.** Three scheme packages, the contract package, and its application to all four schemes.
- **Out of scope.** `basic`, `mtls`, `aws_sigv4`, `github_app` (phase 6 and later).
- **Dependencies.** T3, T5, T10.
- **Acceptance.** All four schemes pass `SchemeContract`; adding a fifth scheme needs no change outside its own package, shown by T24 later.

## Phase 3: resolver and sources

Three PRs. After them a connection with a grant can be turned into an authorized MCP call from a Go test, with no HTTP API and no session yet.

### T12. Credential resolver · [AI-843](https://linear.app/stream/issue/AI-843)

- **Description.** `core.Resolver` implementation: `Resolve(ref, need)` loads the connection, checks status, opens `Material` through the backend, calls `Scheme.Mint` on a detached context with its own deadline under the backend lock with the checkpoint before any refresh, persists rotated material at the next revision, maps `Outcome` to status and `last_error` (`needs_reauthorization`, `connected`, temporary), and caches the fast path by (connection, revision) with an explicit maximum age. `Invalidate(ref, why)` moves the connection to `needs_reauthorization` and drops the cache entry. This replaces the branch's `connectors.ResolveCredentials` (`runtime.go:26-153`), whose refresh ran on the tool call's context.
- **Scope.** `internal/connectors/core/resolver.go`, integration tests: refresh race, lost response, interruption during refresh does not change status, disconnect blocks a new resolve within the cache window.
- **Out of scope.** Request wrapping (T13); rate limiting (T28).
- **Dependencies.** T8, T10, T11.
- **Acceptance.** Cancelling the caller's context during a refresh leaves the connection `connected` or rotated, never `needs_reauthorization`; p50 and p95 of the fast path are printed by a benchmark test (no target asserted yet, per the design's spike 4).

### T13. Transport composition and the wrapping order · [AI-845](https://linear.app/stream/issue/AI-845)

- **Description.** `core.Bound.Transport`: builds the outbound `http.RoundTripper` as egress policy, then `Scheme.Wrap` (and `tls_scheme` when set), then the base transport. One transport per connection, cached, closed on disconnect. A `RoundTrip` test proves a signing scheme sees the final headers and body and that a private destination is refused before any credential is applied.
- **Scope.** `internal/connectors/core/transport.go`, tests with a recording scheme.
- **Out of scope.** `mtls` itself (later).
- **Dependencies.** T12, T2.
- **Acceptance.** Order is enforced by construction: a source cannot obtain a transport without egress; a test with a fake signing scheme sees `Content-Length` and the body hash of the final request.

### T14. Source mcp · [AI-849](https://linear.app/stream/issue/AI-849)

- **Description.** `internal/connectors/sources/mcp` implementing `core.Source` from the branch's `internal/mcp/mcp.go`: `Discover` with the official Go SDK, paginated `tools/list`, `ToolSchemaDigest` over name, description and input schema; `Open` returns a `Runtime` with the allowlist, digest check, JSON Schema validation of arguments, prefixed names and collision detection, 4 MiB response and 32 KiB result caps, `isError` kept as an error. Plus `core/contracttest.SourceContract`.
- **Scope.** The source package, the contract suite, tests against a local MCP server from T5.
- **Out of scope.** The dispatcher and envelope (T21); `http` and `openapi` sources (T25).
- **Dependencies.** T3, T13, T5.
- **Acceptance.** `SourceContract` passes: stable digest, ungranted tool never dispatched, changed schema hidden, oversized result cut with the marker, non-text `isError` still an error; `startupTimeout` of 10 s bounds `Discover`.

## Phase 4: API with Huma

Five PRs. Each registers its operations with `huma.Register` beside Go request and response structs, runs `go run ./cmd/openapi`, regenerates the Go SDK and nothing else. The branch's handlers in `internal/api/connectors.go` are the behavior to keep; their generated oapi-codegen wrapper types are not ported.

### T15. Connector definitions endpoints · [AI-837](https://linear.app/stream/issue/AI-837)

- **Description.** `GET /v1/agents/connectors` (search built-ins and the app's custom definitions), `GET /v1/agents/connectors/{id}`, `POST /v1/agents/connectors` for a custom MCP definition (id `custom_*`, public https endpoint, scheme from the registry). Responses expose the non-secret manifest: schemes, inputs, scopes, client policy.
- **Scope.** `internal/api/connectors.go` (new file on `accelerate`), OpenAPI regen, Go SDK regen, handler tests.
- **Out of scope.** Connections (T16); editing a built-in.
- **Dependencies.** T6.
- **Acceptance.** Operations are server-side only unless marked `x-client-accessible`; the OpenAPI freshness test passes; a custom id that shadows a built-in returns 400 in the API's `{"error": ...}` shape.

### T16. Connections CRUD with owner checks · [AI-841](https://linear.app/stream/issue/AI-841)

- **Description.** `POST /v1/agents/connections` (connector id, owner `app` or `user`, inputs validated by the manifest, label), `GET /v1/agents/connections` with filters, `GET` and `DELETE /v1/agents/connections/{id}`. Owner rule from the branch: a user-owned connection is created only by the server-side backend and only for the verified user it acts for; reads of another user's connection return not-found. Delete is the soft delete from T7 and refuses a connection still bound by a fixed binding unless `force=true` (`ConnectorConnectionReferenced`).
- **Scope.** Handlers, OpenAPI and Go SDK regen, tests for cross-user and anonymous callers.
- **Out of scope.** Credentials (T18); OAuth (T17).
- **Dependencies.** T7, T15.
- **Acceptance.** Alice cannot read, delete or authorize Bob's connection; an anonymous or guest caller gets 400 on a user-owned create; inputs missing from the manifest return 400 naming the input.

### T17. OAuth consent flow: authorizations, launch page, handoff, callback · [AI-844](https://linear.app/stream/issue/AI-844)

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

- **Description.** Table `connector_oauth_clients` (`customer_id`, `connector_id`, `client_id`, sealed secret, auth method, `policy: operator | customer`) with `PUT` and `DELETE /v1/agents/connectors/{id}/oauth-client`. T17's authorize reads the client from this record (customer BYO) or from the operator environment (`<ID>_MCP_CLIENT_ID`), instead of per-request `oauth_client_id` and `oauth_client_secret` sealed into each grant.
- **Scope.** Migration, store, handlers, the change in T17's lookup, OpenAPI and Go SDK regen.
- **Out of scope.** White-label redirects; token export.
- **Dependencies.** T16, T17.
- **Acceptance.** Rotating a customer secret touches one row and the next refresh of every connection of that connector uses it; a connector whose manifest says `policy: [operator]` refuses a customer client.

## Phase 5: agent config and session

Four PRs. After T21 an agent on staging can call a Slack or Linear tool; after T23 the old plugins are gone.

### T20. Agent config connector bindings · [AI-842](https://linear.app/stream/issue/AI-842)

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

- **Description.** Delete `internal/plugins`, `session/plugin_tools.go` and `attachPlugins`, the five plugin operations from `api/legacy.yaml:1061-1177` and their handlers in `api/plugins.go`, the `plugins` fields of the config schemas, and the SDK surfaces that reference them. Migration drops `agent_plugin_connections` and `agent_configs.plugins` (branch `20260929210000`). No import: staging has 0 rows.
- **Scope.** Deletions, one migration, OpenAPI and Go SDK regen, the Python plugin's `folder.py` and `config.py` plugin fields.
- **Out of scope.** Any new behavior.
- **Dependencies.** T21 (so an agent always has a tool path), T22.
- **Acceptance.** `grep -rn plugin_id acceleration/` finds nothing outside the migration's down block; `legacy.yaml` is smaller and the OpenAPI freshness test passes; the dashboard's plugin calls, if any remain, get 404 and are tracked in the Volt repo.

## Phase 6: proof of the design and hardening

Ten PRs that can run in any order once their dependencies are in. The first two are the proof the design asked for: a second scheme and a second source that touch no file in `core`.

### T24. Scheme oauth2\_client\_credentials · [AI-847](https://linear.app/stream/issue/AI-847)

- **Description.** One package: `Begin` is non-interactive, `Material` holds client id and sealed secret, `Mint` posts `grant_type=client_credentials` to the manifest's token endpoint and caches until expiry, `Classify` and `Wrap` reuse the OAuth helpers. Add the Salesforce manifest's `oauth2_client_credentials` entry and a fake-provider personality.
- **Scope.** `schemes/oauth2cc`, its `SchemeContract` run, the Salesforce manifest line.
- **Out of scope.** `oauth2_jwt_bearer`.
- **Dependencies.** T11, T12.
- **Acceptance.** The diff touches nothing under `core/`, `api/` or `session/`; the contract suite passes; an app-owned Salesforce connection resolves a token without a browser.

### T25. Source http: operations defined as data · [AI-852](https://linear.app/stream/issue/AI-852)

- **Description.** `sources/http`: a connection's manifest or custom definition lists operations (`name`, `description`, `method`, `path` template, parameter mapping to path, query, header or body, `body: json | form`, response filter); `Discover` returns them as `ToolSpec` with digests; `Open` runs them through `Bound.Transport` with the same envelope. A Twilio-shaped `POST .../Messages.json` with a form body is the test case.
- **Scope.** The source package, `SourceContract` run, manifest schema extension for `operations`.
- **Out of scope.** `openapi` source; `provided_arguments`.
- **Dependencies.** T14, T13.
- **Acceptance.** No change in `core/`; a private base URL is refused; a form-encoded body reaches the fake server with the credential applied last.

### T26. Signals endpoint and the first verifier · [AI-848](https://linear.app/stream/issue/AI-848)

- **Description.** `POST /v1/agents/connectors/events/{connector_id}` with no API auth: the one inbound door for any provider event. The `Verifier` registry; `signals/hmacheader` (header name, algorithm, encoding and secret source from the manifest). A verifier returns typed events; grant events (`Revoked`, `Uninstalled`, `Rotated`) go to `Resolver.Invalidate` by account id; every other event is handed to an adapter hook that today logs and drops it. That hook is where a channel adapter plugs in later (ground rules). First mapping: Slack `tokens_revoked`.
- **Scope.** Handler, registry, one verifier, Slack manifest `signals` block, tests with a signed and an unsigned payload.
- **Out of scope.** `jwt_set` (Google RISC), `twilio_signature`; routing a message event to a session, which is the channel adapter (see «Later: channels»).
- **Dependencies.** T12, T16.
- **Acceptance.** An unsigned or stale payload returns 401 and changes nothing; a valid `tokens_revoked` moves the matching connection to `needs_reauthorization` and the next resolve fails fast; a validly signed non-grant event is accepted, logged and dropped through the adapter hook, so a channel adapter can register without a change to the handler.

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

- **Description.** Calendly, Cal.com, GitHub, Gong and Salesforce as YAML under `providers/`, from the branch's `connectors.yaml` and the design's stress-test findings; the Shopify callback hook `shopify.callback_hmac` if Shopify returns.
- **Scope.** YAML, fixtures, `ManifestContract` runs. No Go outside `providers/`.
- **Dependencies.** T6, T9.
- **Acceptance.** All load at startup as revision 1; `TestHooksAreRegistered` passes; each has a recorded fake-provider consent test.

### T33. SDK parity · [AI-861](https://linear.app/stream/issue/AI-861)

- **Description.** One PR per SDK (JS, Python plugin, Swift, Kotlin, Dart, .NET, Ruby, Rust, PHP) regenerating types from the spec and exposing `connectors` on config and `connector_bindings` on session creation. Per AGENTS.md this runs periodically after Go is stable; each SDK's own skill says how.
- **Scope.** Generated types plus the thin wrappers; language test suites.
- **Dependencies.** T22, T23.
- **Acceptance.** `npm run types --check` and the equivalent parity checks pass; no SDK exposes a credential field.

## Later: channels, not scheduled

Three PRs that would make a channel out of the same connection layer, if channels are built as adapters in Accelerate (option A in the architecture doc). The alternative under consideration, option B, is a bridge into Stream Chat: an external channel arrives as a Stream Chat channel and Router's existing hook (`internal/api/messagehooks.go`) answers, so none of these three PRs is needed in Accelerate. They are outside the 12 waves and start only when a channel is asked for and the approach is decided. Slack comes first under either option: Thierry on September 17, «For our own sovereign ai we would need good slack integration». Shape follows the architecture doc's two-way door «Where a channel's transport lives».

### T34. Channel adapter contract and the manifest channels block · [AI-860](https://linear.app/stream/issue/AI-860)

- **Description.** `core.Channel` interface: map a verified inbound event (from T26) to a connection, a conversation key and a message; reply through an `http` source operation named in the manifest. Manifest `channels` block: event names, the reply operation, the conversation key path. Store `channel_conversations` (provider conversation id to session or Stream Chat channel). The multi-person rule from T21 applies: a channel session with several people uses app-owned connections only.
- **Scope.** Interface, manifest fields, the store table, the adapter hook wiring in T26's handler, a fake-provider channel personality.
- **Out of scope.** Any real provider.
- **Dependencies.** T26, T25, T22.
- **Acceptance.** The fake provider posts a message event; a text session receives it and its reply leaves through `Bound.Transport` with the connection's credential; an event for an unknown conversation opens a new session under the configured agent.

### T35. Slack channel adapter · [AI-862](https://linear.app/stream/issue/AI-862)

- **Description.** Slack Events API on an app-owned bot connection: acknowledge within 3 seconds, honour `x-slack-retry-num`, thread id as the conversation key, reply through a `chat.postMessage` operation in the Slack manifest. Athena's first scenario is this channel.
- **Scope.** Adapter package under `signals/` or `channels/slack`, Slack manifest `channels` block, tests with recorded Slack events.
- **Out of scope.** Slack as a tool (already T9 to T21); personal tokens in a shared channel.
- **Dependencies.** T34.
- **Acceptance.** A mention in a channel with three people produces one session that calls only app-owned tools; a retried delivery is deduplicated; the reply lands in the same thread.

### T36. Linq channel adapter (iMessage, BYO account: unverified) · [AI-863](https://linear.app/stream/issue/AI-863)

- **Description.** Linq webhooks with HMAC verification into T26, reply through the Linq send operation defined as data for the `http` source, on a customer-owned Linq account (BYO, unverified as a decision: asked on October 1 whether the customer brings their own Linq key, Thierry answered «well thats we have to figure out»). No voice: Linq's API places no calls.
- **Scope.** Linq manifest (`api_key` scheme, `channels` block, `operations`), adapter, tests against the fake provider.
- **Out of scope.** Apple Messages for Business; Stream-operated lines.
- **Dependencies.** T34, T25.
- **Acceptance.** An inbound iMessage opens or continues a text session; the reply is sent on the customer's line; a payload with a bad signature is refused by T26 before the adapter runs.

## Sources

- [Accelerate connectors: architecture design](architecture.md): the keep, change and add lists, the one-way doors, the validation plan this decomposition follows.
- [Voice-agent connectors: competitor analysis](competitor-analysis.md): Work plan P0 and P1 items, «Where plugins run today» (0 plugin rows on staging).
- Branch [`codex/connector-support`](https://github.com/GetStream/Vision-Agents/tree/codex/connector-support) at `cf62af0d`: the code copied in T1, T2, T8, T9, T14, T17, T21 and the migrations named by date.
- `accelerate` at `8771b8bb`, checked October 1: `internal/auth/secret.go:14` (`KEKVersion = 1`), `internal/plugins/`, `api/legacy.yaml:1061-1177`, `internal/api/policies.go` (Huma pattern), `cmd/openapi/main.go`, `.github/workflows/ci.yml:78-98`.
- `AGENTS.md`: Huma for new operations, no additions to `legacy.yaml`, Go first for SDK changes, no mocks, the `pagination` and `go-testing` skills.
