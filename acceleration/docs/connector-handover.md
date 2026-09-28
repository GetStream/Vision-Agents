# Accelerate connectors: engineering handover

Updated 28 September 2026. This is a working implementation with live Linear and
Slack demonstrations, not a production-complete release. Start here; the longer
[design and research](connector-design.md) explains the API decisions and the
[Eve study](eve-connectors-research.md) covers its MCP/connector approach.

## Branches and integration

- Backend and SDKs: [Vision-Agents, codex/connector-support](https://github.com/GetStream/Vision-Agents/tree/codex/connector-support).
- Dashboard: [volt-dashboard, codex/connector-support-handover](https://github.com/GetStream/volt-dashboard/tree/codex/connector-support-handover). A separate name preserves an older local connector worktree with uncommitted changes.
- Original backend base: `89d1193e69e182be8c68850713c96d82aa77be53` on `accelerate`.
- Original Volt base: `436f1b3cd` on `ai-team/agent-dashboard`.
- Both upstream branches advanced during development. At handover, the observed
  upstream tips were backend `38f7869c5bc135054c13dd387b78f69914af4471` and Volt
  `f0a325cf1e08bd63da2321879b6b19cb27bf265e`. Integrating those changes and rerunning
  checks is still required. These feature branches preserve the tested snapshot.

Changes are grouped into backend/runtime, SDK migration, dashboard connections,
Playground fixes, and documentation commits. No local environments, OAuth tokens,
encryption keys, tunnel configuration, or conversation outbox are deliverables.

## Mental model and request flow

| Object | Responsibility |
| --- | --- |
| Connector definition | Describes the provider or tenant-owned custom MCP endpoint, authentication mode, and provider policy. |
| Connection | One reusable authorized account, owned by the app or a verified application user. Holds encrypted credentials independently of any agent. |
| Agent binding | Gives an alias an account-selection rule and exact reviewed tool grants. Connecting an account alone grants the agent nothing. |

An app-owned account can be fixed in agent configuration. Personal accounts are
selected at session creation for an alias declared with `connection.type=session`.
The caller's authenticated identity supplies ownership; the agent's call user ID,
tool arguments, and conversation text cannot select a different owner.

1. A trusted backend creates a connection. Volt currently creates app-owned ones.
2. OAuth authorization returns a router launch URL and a short-lived browser
   handoff token. Volt opens a popup and sends that token using origin-checked
   `postMessage`; it is never placed in a URL.
3. The router sets its own HttpOnly callback cookie, performs provider consent
   with PKCE S256, and consumes the expiring one-use authorization state.
4. The callback seals the grant and redirects to the configured Volt Connections
   page. Access and refresh tokens are not returned by the management API.
5. Discovery/validation lists available tools. An administrator grants exact
   names and their reviewed schema digests to an agent alias.
6. Session startup validates ownership, provider, connection status, and tool
   schema. Tools appear to the model as `alias__tool_name`.
7. Each outbound MCP request resolves current credentials. The model supplies
   tool arguments, never credentials or an alternative connection identity.

Required unavailable bindings fail startup. Optional unavailable bindings are
omitted and produce replayable `connector_unavailable` events. Changed names,
descriptions, or input schemas require grant review rather than silently expanding
permissions. Forked sessions re-resolve current config and caller authority.

## API and code map

The source of truth is [api/openapi.yaml](../api/openapi.yaml).

| Endpoint | Purpose |
| --- | --- |
| `GET/POST /v1/agents/connectors` | List definitions or register a tenant-owned custom definition. |
| `GET/POST /v1/agents/connections` | List connection metadata or create an account connection. |
| `GET/DELETE /v1/agents/connections/{id}` | Read metadata or disconnect locally. |
| `POST /v1/agents/connections/{id}/authorizations` | Begin consent, including reauthorization. |
| `PUT /v1/agents/connections/{id}/credentials` | Write static credentials, activate no-auth, or import an OAuth grant through the backend. |
| `GET /v1/agents/connections/{id}/tools` | Discover tools and digests for review. |
| `POST /v1/agents/connections/{id}/validate` | Check connection health and discovered tools. |
| `GET /v1/agents/connectors/oauth/callback` | Complete browser consent. |

Agent config uses `connectors`; session creation uses `connector_bindings`.
For example, a config binding for personal Linear access is:

```json
{
  "name": "linear",
  "connector_id": "linear",
  "connection": { "type": "session" },
  "tools": [
    { "name": "list_teams", "schema_digest": "<64-character digest returned by discovery>" }
  ],
  "required": true,
  "timeout_ms": 10000
}
```

The digest above is a placeholder, not an accepted literal. The session request
then supplies `"connector_bindings": [{"name":"linear","connection_id":"<owned-connection-id>"}]`.
A fixed app-owned binding instead uses
`"connection": {"type":"fixed","connection_id":"<app-connection-id>"}`.
Session inputs cannot override it or add tool grants.

| Location | What to read |
| --- | --- |
| `internal/api/connectors.go` | Ownership checks, management handlers, OAuth browser handoff and callback. |
| `internal/mcp/` | Catalog, OAuth discovery/exchange/refresh, official MCP SDK transport, tool discovery. |
| `internal/connectors/` | Encrypted credential bundles, dispatch-time resolution and refresh coordination. |
| `internal/store/connectors.go` | Connections, authorization attempts, revisions and database locking. |
| `internal/session/connector_tools.go`, `mcp_tools.go` | Binding resolution, schema grants, required/optional behavior and tool execution. |
| `internal/egress/` | Outbound destination protections used by remote MCP/OAuth. |
| `internal/connectorimport/`, `migrations/20260924*`, `migrations/20260925*` | Old-account transfer and new storage schema. |
| `plugins/stream/`, `sdks/` | Config bindings and session account selection; removal of obsolete plugin surfaces. |
| Volt `src/api/agents.ts`, `agent-connections*.tsx` | Connection setup, credentials/consent, account and exact tool selection. |
| Volt `agent-playground.tsx`, `src/utils/agents/session.ts` | Text/voice session handling and reply reconciliation. |

## Authentication, refresh, and disconnect

Supported modes are anonymous, bearer token, API key, and OAuth2. OAuth supports
public/confidential clients, validated RFC 8414/OIDC metadata, client metadata
documents where advertised, and dynamic registration as a compatibility fallback.
Provider-specific client policies remain in the catalog/OAuth implementation.

Credentials are AES-GCM encrypted, bound to tenant, connection and revision.
Persisted credentials and pending attempts carry KEK versions. Use a persistent
operator-managed keyring and retain old versions until old credentials and
authorization attempts no longer reference them. Successful use lazily rewraps
credentials with the current key. Losing the keys loses access to stored grants.

Refresh is serialized across router replicas with a PostgreSQL advisory lock.
Before sending the refresh token, the router durably checkpoints the account as
requiring reauthorization. Success persists the rotated encrypted grant and
restores connected status. A lost response, canceled worker, or uncertain provider
outcome leaves reconnect required, preventing replay of a potentially consumed
refresh token. Known temporary rejections may fall back to an access token only
while it remains valid. Explicit invalid grants require reconnect.

Disconnect removes local encrypted credentials and stops subsequent dispatches,
including through an already-open runtime. It cannot undo a request already sent.
Provider-side token revocation is not implemented; local disconnect must not be
presented as proof that the provider revoked its tokens. Side-effecting tools must
not be blindly retried after an unknown outcome.

## Running the demo

Follow [the dashboard development skill](../development_skills/dashboard_skill.md)
for the full Volt/example setup and the [router README](../README.md) for variables.
Use an ignored repo-root `.env` and a disposable test database for automated tests.

- Configure a persistent high-entropy `ROUTER_AUTH_KEK` or versioned
  `ROUTER_AUTH_KEK_V1`, `ROUTER_AUTH_KEK_VERSION` keyring.
- Set `ROUTER_PUBLIC_URL` to a stable HTTPS origin and register
  `<ROUTER_PUBLIC_URL>/v1/agents/connectors/oauth/callback` with providers.
- Set `DASHBOARD_BASE_URL` to the complete scoped Volt Connections URL, including
  organization/app path and config query. Callback status is appended to it.
- Slack needs `SLACK_MCP_CLIENT_ID` and `SLACK_MCP_CLIENT_SECRET`. Enable MCP and
  PKCE in the Slack app. Workspace installation approval and user consent are
  separate. The tested scopes were `channels:read`, `users:read`, `im:write`, and
  `chat:write`; remove unwanted default scopes after enabling MCP. Its user OAuth
  endpoint uses the `scope` parameter, not `user_scope`.
- Local `noauth` is only for a private development setup. If tunneling it, expose
  only the OAuth launch/callback and client metadata paths, never management APIs.
- Rebuild/restart the router after source or environment changes, then start
  `simple_voice_ai`, connect the account in Volt, grant one tool, and use Playground.

At handover, the live demo binary includes the Slack consent and callback fixes
but predates the latest durable refresh checkpoint changes. Those source changes
passed automated tests. End active sessions before rebuilding/restarting the demo.
After the test Slack secret is rotated, update the ignored environment, restart,
and reauthorize affected connections because stored grants retain client metadata.

## Removal of the old connector implementation

The old per-agent plugin routes, implementation, config fields, and SDK surfaces
are removed. There is one connector model. Startup transfers supported connected
Slack, Calendly, Cal.com, and Salesforce rows before dropping the old plaintext
table and config column. The importer keeps IDs, encrypts preserved credentials,
checks conflicts, and creates fixed bindings only for previously selected entries.

Imported connections require reauthorization and have no granted tools: old
records lack sufficient trusted OAuth/account metadata. Unsupported providers
(including Shopify) and inactive logins are not transferred. Missing encryption or
a transfer failure stops startup before the destructive removal migration. Review
backup/restore and rehearse this upgrade against a representative database before
deployment; restarting an old binary is not a sufficient rollback after the drop.

## What has actually been demonstrated

| Provider | Evidence and limits |
| --- | --- |
| Linear | Live read-only OAuth, discovery, explicit `list_teams` grant, and an agent response counting 48 teams. Live expiry/reconnect/revocation not tested. |
| Slack | Live OAuth, messaging-tool discovery, explicit `slack_send_message` grant, and an approved self-DM visibly delivered. User also reported the voice test worked. Channel sending was intentionally skipped. |
| Gong | Catalog/manual and automatic registration paths implemented; live interoperability unverified, including manual token client-auth method. |
| Salesforce | External Client App/PKCE configuration implemented; live account/tool/refresh cycle unverified. |
| Calendly | Published metadata and public-client registration flow studied; live lifecycle unverified. |
| Cal.com | Hosted MCP/OAuth definition implemented; live registration/scopes/lifecycle unverified. |
| GitHub | Discovery/registration and optional configured public client supported; live lifecycle unverified. |

The Slack DM included an adjacent approval sentence from the test prompt. Delivery
was verified; exact-message fidelity was not. Delimit the payload in the next test
and inspect the tool arguments. The live Slack and Linear connections did not
provide a stable account ID through the current token mapping; do not claim that
wrong-account reconnect prevention has been proven for those live responses.

## Validation and remaining release work

The targeted Go checks on 25 September and Volt checks rerun on 28 September passed:

```sh
# From acceleration/
go test ./internal/mcp ./internal/connectors ./internal/store ./internal/api
go test -tags=integration ./internal/api -run 'TestAPIIntegrationSuite/Test(ConcurrentCredentialResolutionCommitsOneRotatedRefreshToken|RefreshOutcomeSurvivesLostResponsesAndCanceledWorkers)$' -count=1 -timeout=90s
go test -tags=integration ./internal/store -run 'TestStoreSuite/TestConnector' -count=1 -timeout=90s

# From Volt: 28 tests passed across these four files
bun run vitest run --project unit tests/unit/agents/connector-settings-flow.test.tsx tests/unit/agents/session.test.ts tests/unit/agents/design-review.test.tsx tests/unit/agents/playground-flow.test.tsx
```

Run database suites serially with `ROUTER_POSTGRES_DSN` pointing at a disposable
`_test` database, never the live demo database: store suites can reset the schema.
Volt `bun lint` passed its TypeScript stage but fails on the unchanged
`src/utils/agents/voices.ts:35` sort-comparator rule. Full lint is not green.

Before publication, the changed files and all five new commits were checked for
the credential values in the local environments, with no matches. Gitleaks found
no Volt-history issues; its backend-history finding was reviewed as a false
positive in the generated compressed OpenAPI schema. Local environment files and
the conversation outbox remain ignored and untracked.

Earlier in this work, broader Go/SDK and database acceptance checks passed as
recorded in the design document. Full Go tests also had baseline session-attribution
and five simulation failures reproduced on the original base. Those earlier SDK
results do not certify the final snapshot. Final SDK regeneration and complete
parity checks were deliberately deferred until agent validation is finished.

Work required to finish stages 1–3 of the plan:

1. Integrate the advanced upstream branches, resolve conflicts, and rerun checks.
2. Rebuild the router with the tested refresh changes. Exercise live token expiry,
   rotation, reconnect, and disconnect during an open session for Slack and Linear.
3. Complete the other five providers' consent → tool → expiry → reconnect/disconnect
   flows using approved accounts. Catalog presence alone is not verified support.
4. Establish stable provider account identity, especially for Slack/Linear responses
   currently lacking an ID. Validate same-account reconnect and account-switch
   rejection against real responses. Do not infer identity from tool descriptions.
5. Decide and implement provider-side revocation semantics; document local-only
   disconnect until then. Test provider outages and unknown write outcomes.
6. Finish the product flow for user-owned account consent/selection. API ownership
   and per-session selection are tested; Volt currently manages app-owned accounts.
   Validate the deployed trusted-backend/auth configuration, not only local noauth.
7. Rehearse migration/restore and encryption-key rotation/recovery. Review outbound
   destination policy and operational error/latency telemetry for production.
8. Measure startup, cold/cached discovery, warm calls, refresh and outages under
   concurrent voice sessions; no latency/capacity SLO has yet been established.
9. Once API and agent behavior are stable, regenerate every SDK from the spec,
   check generated parity (including Volt), run per-language checks and full CI,
   and address or track baseline failures before merging.

Stage 4 remains optional future scope: interactive per-invocation approval,
backend-supplied arguments, delegated-task connector access, native/OpenAPI
adapters, tool search, and private networking. Today's tool grant authorizes the
tool; it is not an interactive approval step for each send or write.
