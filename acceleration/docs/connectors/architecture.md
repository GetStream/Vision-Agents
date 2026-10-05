# Router connectors: architecture design

Oct 1, 2026 · @Kanat Kiialbaev

Exported from Claude Docs on 2026-10-05 (https://claude.ai/code/artifact/b2e88660-5435-4b00-89f6-0e56da8b0bfa). The Claude Doc is the source of truth; this copy is a snapshot.

## Decisions, 2026-10-05

These decisions close the open channel questions of this doc. They are the target design. They are not in the code. They are a proposal from the connectors work: Thierry has not confirmed them yet (channels doc, «To decide with Thierry»). Details and diagrams: [Connectors: inbound channels, tools and the omni-channel conversation](channels.md). Comparison with Vercel: [Vercel Connect и наши connectors](https://claude.ai/code/artifact/b1467e6d-f4f7-4f5e-bb77-3f862c19e608).

1. **Channel transport.** The channel bridge runs in the Router. It writes the inbound message to a Stream Chat channel. Router's existing message hook wakes `session.Session` ([`internal/api/messagehooks.go:61`](https://github.com/GetStream/Vision-Agents/blob/99b39e1efed159d29d2b5c9e6c5b71fcd3c115c1/acceleration/internal/api/messagehooks.go#L61)). Rejected: a bridge that calls the session directly (no shared history), and a bridge in a separate service (a second token store, a second contact map and its own Stream Chat webhook).
2. **The connector layer serves both sides.** Tools use the `sources` block of `core.Manifest`. Inbound channels use a new `channel` block: scheme, verifier parameters, event routing, thread key, reply endpoint and outbound policy, for example the WhatsApp 24-hour window. An inbound channel is still never a kind of tool (one-way door 9). One events endpoint for each provider app, `/v1/connectors/events/{provider_app_id}` (proposal), is a second route into the same handler as the per-connector endpoint, which stays for connectors with no provider app for each customer. `core.Verifier` checks the request and returns either token signals or a message. Reason: Slack sends `tokens_revoked` and `app_uninstalled` as Events API events to the app's Request URL ([tokens\_revoked](https://docs.slack.dev/reference/events/tokens_revoked), [app\_uninstalled](https://docs.slack.dev/reference/events/app_uninstalled)).
3. **Episodes.** Each external thread has a thread channel with the messages word for word. The omni-channel, the person's agent channel, keeps one episode card for each call or text thread. An episode closes at call end or after an idle period. On close, the summary goes into the card and the facts go to memory. The agent reads its own thread word for word and other channels through the cards.
4. **One provider unit for each customer.** Slack: one Slack app for each `app_pk`, created by the Router with `apps.manifest.create`, or the customer's own app (`client.policy: customer`). WhatsApp: the customer's WABA and number under one Stream Tech Provider app. Telegram: the customer's bot. SMS and iMessage: the customer's account or number. The shared Stream app serves only Stream's own agents. Each Slack app has its own events URL, stays in the customer's workspace without public distribution, has token rotation on, and can restrict its tokens to the Router IP ranges (at most 10 in the manifest).
5. **Direct calls.** The proxy is the default. The Router adds the token and forwards the request unchanged. It implements no provider method; the customer uses the provider's official SDK with its base URL set to the proxy. Token export is opt-in and only for the customer's own unit: a bot token for a connection in the agent's `core.Binding` (`fixed`), a user token only in that user's own session (`session`), and never when tokens are restricted to the Router IP ranges. Static keys (`api_key`, bot tokens) do not rotate, so the proxy stays their default. Raw event forwarding signs each request with a key of that customer.
6. **Integration modes.** A: full platform, agent config only. B: platform plus customer code for what the provider MCP server does not offer. C: pass-through, the connector and agent layers only; the customer keeps the history and uses the proxy, token export and event forwarding.
7. **Tenancy.** The tenant of every record is the customer's Stream app (`app_pk`). In the Stream platform, `app_pk` separates all data and `org_id` is only an attribute of the app: in the live `chat` database on October 5, 124 of 131 tables have `app_pk` or `app_id`, and one has `org_id` as a non-key column. Rules from the branch `nash/project-tenancy`: channels live in the customer's app; episodes, thread links and provider apps keep a `stream_app_pk` pin; the Router sets its message hook in each customer app (which code does this: `unverified`); the customer's Stream app key stays in the `stream_apps` table of `nash/project-tenancy`, and the customer's Slack app secrets stay in `store.ConnectorConnection` (owner `app`), read through `core.Resolver`; both tables seal with the same `auth.Sealer` and keyring. No org level now.
8. **Industry.** Vercel Connect creates a fresh Slack app for each connector, gives tokens to customer code with `getToken` and forwards verified events as triggers. Nango offers a proxy and credential retrieval. Pipedream Connect gives credentials only with the customer's own OAuth client.

## TL;DR

Keep the AI-816 data model (Connector → Connection → Binding → Grant) and the credential resolver boundary. Replace everything that is provider-specific Go code with two things: a **provider manifest** (data, stored in Postgres with revisions) and four small **adapter registries** (auth schemes, tool sources, store backends, event verifiers). A new provider becomes a manifest row. A new auth method, tool type or token store becomes one Go file that implements one interface. The core never learns a provider's name.

**What the stress test showed.** Of 12 awkward providers, 11 can be expressed with manifest data plus the adapters below. One (Shopify) needs a single named hook for its callback signature. None needs a core change once the model below is in place. Today's branch fails 9 of the 12 without editing core files: provider names are hardcoded in `internal/mcp/catalog.go:111`, `internal/mcp/oauth.go:265,824,834`, the auth type is a database CHECK with four values, and the only executor is MCP over HTTP.

**One-way doors to fix now** (details in «One-way doors»):

1. A connection's identity and its reuse on reconnect: (app, connector, owner, account) is one row; reconnect never creates a second grant.
2. Credential material is an opaque, scheme-tagged sealed blob plus public metadata. Not fixed columns.
3. Connector definitions live in the database with revisions, not in an embedded YAML compiled into the binary.
4. One resolver door for every executor, with the refresh running on its own context and deadline.
5. The egress policy wraps every outbound transport last, for every tool source.
6. Tool authority is the internal map, never the exposed name. Grants stay explicit and digest-pinned.
7. Owner identity comes only from the trusted principal; a session with several verified people uses app-owned connections by default.

**Two-way doors to postpone:** which store backend (sealed Postgres, KMS, a broker), the HTTP and OpenAPI tool sources, the voice policy fields, step-up and approvals, provider-side revocation, rate limiting, tool search.

**Scope of the first build.** Registries and contract tests now; ship the same four schemes and the MCP source the prototype has today. The proof that the design holds is a second scheme (`oauth2_client_credentials`) and a second source (`http`) added without touching the core, each covered by the same contract tests.

## Goal, constraints and what counts as done

**Goal.** In six months a new service must not need a core change or a hack. Adding a provider, an auth method, a tool type or a token store is new data or one adapter. Secondary: little provider-specific code and manual work; many tenants, users and accounts per provider; no effect on voice call latency.

**Constraints, from the brief and the competitor doc.**

- Both scenarios stay: app-owned and user-owned connections, voice and text. This is the decision recorded in «TL;DR» of the competitor doc: «We support both scenarios, and the developer picks the owner», with app-owned first. Since October 1 the competitor doc names Athena's first scenario: a Slack session with several people, which runs on app-owned connections on Stream's internal app; personal connections come after it.
- A broker may come later and must not be the core. The doc's «Build our own layer or use a broker» puts a broker only behind the credential resolver boundary (`connector-design.md:409`) and decides on data after a month of Slack and Linear in production.
- Channels are out of scope. «iMessage: a channel, not a connector» and «Decide separately» make this split. Inbound provider events about a grant (a revoked token) are not a channel and are in scope. Where a channel's transport lives was decided on October 5: the channel bridge in the Router writes to Stream Chat («Decisions, 2026-10-05», one-way door 9 and the two-way door «Where a channel's transport lives»).
- The AI-816 prototype on `codex/connector-support` is an example, not an accepted design. It can be redone. Where this doc cites its code, the path is on that branch under `acceleration/`.
- Router has no production deployment, only staging. Plugins have zero rows there («Where plugins run today» in the competitor doc). So the plugin import code and the irreversible migration are not needed.

**Sources used.** The competitor doc as a whole, with its «Appendix: fact check» as the source of truth where the text disagrees. The branch code: `internal/mcp/{catalog.go,connectors.yaml,oauth.go,mcp.go}`, `internal/connectors/{runtime.go,secrets.go}`, `internal/store/connectors.go`, `internal/session/{connector_tools.go,mcp_tools.go}`, `internal/api/connectors.go`, and the five migrations dated 2026-09-29. The two design docs on the branch, `connector-design.md` and `connector-handover.md`. Facts that come from outside these are marked **outside the document** with a way to verify them.

**Done means:**

- Every one of the 12 stress-test providers is expressed as a manifest plus registered adapters, with zero provider names in the core packages. A CI test enforces this.
- A second auth scheme and a second tool source are added as single files and pass the same contract tests as the first ones.
- The resolver fast path (no refresh) is measured under concurrent voice sessions before launch. No number is claimed here: `connector-handover.md` item 8 says none exists yet.

## Layers and interfaces

The core is six stable things. Everything that varies by provider sits behind four registries and one data file. The arrow of dependency points inward: adapters import the core, the core imports no adapter.

&#91;embedded content: connector layers · 6 stable parts, 4 registries, 1 data file\]

A tool call goes down the left column: the session resolves bindings, the Dispatcher maps the name, a Source runs the call, and the Resolver hands it a credential that a Scheme applies before the egress policy lets it out. The manifest feeds the Resolver and the Schemes with data. Attempts write finished grants; the events endpoint invalidates them on a token signal.

**Stable core** (package `internal/connectors/core`):

| Part | What it owns | Where it is today on the branch |
| --- | --- | --- |
| Model | Connector definition (revisioned), Connection (owner, account, inputs, metadata, status), Binding (alias, selection rule, grants, policy), Grant (tool name + schema digest) | `store/models.go:359-435`, migrations `20260929170000`, `20260929180000` |
| Resolver | The one door to a credential: `Resolve(ref, need)`. Lock, revision CAS, checkpoint, mint on its own context, cache of the fast path | `connectors/runtime.go:26-153`, `store/connectors.go:237-291` |
| Dispatcher | Maps an exposed tool name to (binding, connection, source, tool). Applies the envelope: timeout, cancel on interruption, size cap, outcome. Rechecks the grant at call time | `session/mcp_tools.go:11-26`, `mcp/mcp.go:272-331`, `session/connector_tools.go:229-288` |
| Attempt | One-use, expiring, sealed state for any interactive acquisition: consent, reconnect, step-up, admin consent | `store/connectors.go:349-445`, `api/connectors.go:294-401` |
| Policy | Owner checks, grant checks, multi-person session rule, egress policy, data-use flags | `session/connector_tools.go:119-139`, `egress/public.go` |
| Records | Invocation log with error type, audit of grants and consents, dependents of a connection | Not on the branch: `connector-handover.md` item 7, competitor doc P1 items 4 and 5 |

**Pluggable, by registry:**

| Registry | One adapter per | First members | Later members (each one file) |
| --- | --- | --- | --- |
| `Scheme` | way to acquire, mint and apply a credential | `oauth2_code`, `api_key`, `bearer`, `none` (the four on the branch) | `oauth2_client_credentials`, `oauth2_jwt_bearer`, `basic`, `mtls`, `aws_sigv4`, `github_app`, `host_supplied` |
| `Source` | way to discover and run tools | `mcp` | `http` (operations defined as data), `openapi` (pinned spec), `caller` (existing bridge) |
| `Backend` | place grants live | `pg_sealed` (AES-GCM, KEK keyring) | `kms`, `broker` (Nango, Vercel Connect) |
| `Verifier` | way to check an inbound provider event | none | `hmac_header` (Slack, Shopify, Linq), `jwt_set` (Google RISC), `twilio_signature` |
| `Hook` | escape hatch, by name, called at 3 fixed points | none | `shopify.callback_hmac` |

**Data, not code:** the provider manifest. One per connector, stored in `connector_definitions` with a revision, seeded from YAML at startup for the built-ins. It carries endpoints as templates, the allowed schemes, scope format, where the account id comes from, what to capture from the token response or callback, refresh policy, rate-limit keying, the OAuth client policy and the hook names. A connection pins the definition revision it was created from.

**Go signatures of the key interfaces.** Names follow the branch (`harness.Tool`, `llm.ToolCall`, `store.ConnectorBinding`).

```go
// Scheme is one way to acquire, mint, apply and revoke a credential.
// One implementation per scheme. Provider quirks are Profile fields, never a new scheme.
type Scheme interface {
	Name() string
	// Begin starts acquisition. Interactive schemes return an authorize URL and sealed
	// attempt state. Non-interactive schemes (api_key, client_credentials) return Done.
	Begin(ctx context.Context, in BeginInput) (BeginOutput, error)
	// Complete turns a callback or a static input into long-lived Material and the
	// public values the manifest asked to capture (instance_url, realm_id, team_id).
	Complete(ctx context.Context, in CompleteInput) (Material, Captured, error)
	// Mint produces a short-lived Credential from Material. It may rotate Material;
	// the resolver persists what comes back under the lock.
	Mint(ctx context.Context, m Material, p Profile) (Credential, Material, error)
	// Wrap applies the credential to every outbound request: a header, a signature
	// or a TLS client certificate. The egress policy wraps the result, outermost.
	Wrap(base http.RoundTripper, c Credential) http.RoundTripper
	// Classify maps a provider response to the one outcome the core acts on.
	Classify(resp *http.Response, body []byte, err error) Outcome
	// Revoke is best effort. A nil error is not proof of provider-side revocation.
	Revoke(ctx context.Context, m Material, p Profile) error
}

// Material is scheme-private and sealed as one blob, bound to tenant, connection
// and revision (today's AAD in connectors/secrets.go:113).
type Material struct {
	Scheme  string          `json:"scheme"`
	Version int             `json:"version"`
	Payload json.RawMessage `json:"payload"`
}

// Credential is what one request needs. It never reaches a log or the model.
type Credential struct {
	Scheme    string
	ExpiresAt time.Time
	secret    json.RawMessage // read only by the scheme that minted it
}

type Outcome struct {
	Kind       OutcomeKind // OK, InvalidGrant, Transient, Uncertain, ScopeRequired, RateLimited
	Scopes     []string    // for ScopeRequired: the union to ask for
	Claims     string      // for ScopeRequired: a claims challenge (Microsoft CAE)
	RetryAfter time.Duration
}

// Profile is the manifest, resolved for one connection's inputs and captured values.
type Profile struct {
	ConnectorID string
	Revision    int
	Scheme      string
	Endpoints   map[string]string // authorize, token, refresh, revoke, issuer, api_base
	Inputs      map[string]string // shop, instance, region, tenant
	Metadata    map[string]string // captured: instance_url, realm_id, team_id
	Scopes      ScopePolicy       // list, separator, send_on_refresh, step_up_union
	Refresh     RefreshPolicy     // margin, grace window, rotating, token_ttl
	Identity    IdentityRule      // from token_response | id_token | profile_request; path
	Capture     []CaptureRule     // from callback_query | token_response | id_token; name; path
	Client      ClientPolicy      // operator | customer | dcr | cimd; token auth method
	RateLimit   RateLimitRule     // keyed per app | workspace | user; retry_after
	Hooks       map[string]string // point -> registered hook name
}

// Resolver is the only door to a credential. Every Source uses it; none opens the store.
type Resolver interface {
	Resolve(ctx context.Context, ref ConnectionRef, need Need) (Credential, error)
	Invalidate(ctx context.Context, ref ConnectionRef, why Outcome) error
}

type Need struct {
	Audience string
	Scopes   []string
	Deadline time.Time // the call's budget; the mint itself runs on a detached context
}

// Backend is the locked, revisioned storage behind the resolver. The branch's
// WithLockedConnectorConnection (store/connectors.go:237) is the first implementation.
type Backend interface {
	WithLocked(ctx context.Context, ref ConnectionRef,
		fn func(g *Grant, checkpoint func() error) (changed bool, err error)) error
}

// Source discovers and runs tools of one kind: mcp, http, openapi, caller.
type Source interface {
	Kind() string
	Discover(ctx context.Context, b Bound) ([]ToolSpec, error) // each with a schema digest
	Open(ctx context.Context, b Bound, grants []store.ToolGrant) (Runtime, error)
}

type Runtime interface {
	Tools() []harness.Tool
	Call(ctx context.Context, call llm.ToolCall) (Result, error)
	Close()
}

// Bound is one binding resolved against one connection. Transport already carries the
// resolver and the egress policy, so a Source cannot get the order wrong.
type Bound struct {
	Binding    store.ConnectorBinding
	Connection store.ConnectorConnection
	Profile    Profile
	Transport  func(context.Context) (http.RoundTripper, error)
}

// Verifier checks one inbound provider event and names the grants it is about.
type Verifier interface {
	Name() string
	Verify(r *http.Request, body []byte, p Profile) ([]Signal, error)
}

type Signal struct {
	ConnectorID string
	AccountID   string
	Kind        SignalKind // Revoked, Uninstalled, Rotated
}

// Hook is the escape hatch. Registered by name, referenced from a manifest, called at
// exactly three points: BeforeAuthorize, BeforeComplete, AfterToken.
type Hook func(ctx context.Context, hc *HookContext) error

// Registry is how adapters are found. The core holds one of each; adapters register in init.
type Registry struct {
	Schemes   map[string]Scheme
	Sources   map[string]Source
	Backends  map[string]Backend
	Verifiers map[string]Verifier
	Hooks     map[string]Hook
}
```

**Order of wrapping on the way out** (a rule, enforced by `Bound.Transport`): egress policy, then `Scheme.Wrap`, then the base transport. A signing scheme sees the final headers and body. The egress check sees the final destination. This is what `mcp/mcp.go:186-208` does for MCP today, made the rule for every source.

**The session** keeps the shape it has on the branch: `attachConnectors` (`session/connector_tools.go:32`) resolves bindings, and the result is one more `agent.ToolRunner` in the chain (`session/manager.go:354-370`). The change is that the runner is the Dispatcher over all sources, not a runner over one MCP runtime.

## Axes where providers differ

Twenty axes. For each: what the competitor doc and the branch show, whether AI-816 has an extension point or a constant, and what the design needs. «Constant» means a Go switch on a provider name, a fixed struct field or a database CHECK. «Data» means a manifest field. «Adapter» means one registered implementation.

| # | Axis | Examples from the doc | AI-816 today | Needed |
| --- | --- | --- | --- | --- |
| 1 | How a credential is acquired | Auth code + PKCE (Slack, Linear); client credentials (Salesforce at Retell and ElevenLabs, PolyAI APIs tab); JWT bearer and private-key JWT (ElevenLabs `oauth2_jwt`); API key or PAT (Cal.com, HubSpot); basic (Freshdesk, Jira at ElevenLabs); mTLS (ElevenLabs, Dialogflow); IAM AssumeRole (PolyAI Amazon Connect); host supplies the token (Vapi, ElevenLabs `secret__*`); token exchange (AWS, Claude EMA) | Constant. `auth_type IN ('oauth2','none','bearer','api_key')` in migration `20260929180000:18`; same list in `store/connectors.go:72-83` and `connectors/runtime.go:14-19`. No client credentials: `grep client_credentials internal/` finds only `phone/sinch` (competitor doc, «What this means for Accelerate») | Adapter: `Scheme` registry. Column becomes a free string checked against the registry |
| 2 | What is stored | Access + refresh; API key; client id + secret; private key; cert + key; role ARN; extras from the token response: Salesforce `instance_url`, QuickBooks `realmId`, Slack `team.id` (Nango quirks, «Brokers and iPaaS») | Constant. `connectors.Credentials` has 11 fixed fields (`secrets.go:14-27`). `tokenResponse` hardcodes Slack `authed_user` and `team`, Calendly `owner`, Salesforce `id` (`oauth.go:95-116`) | Data + adapter: scheme-private `Material.Payload`; public `metadata jsonb` on the connection filled by `Capture` rules |
| 3 | How a short-lived token is minted | Refresh grant; client credentials re-mint (no refresh token, Retell Salesforce); JWT assertion; GitHub App installation token; STS temporary creds; per-request signing; scheduled fetch (Bland refresh secret); broker `getToken` (Vercel Connect) | Constant. Only `refresh_token` in `ResolveCredentials` (`runtime.go:78-139`), called on the tool call's context («Incidents and provider quirks», «Temporary error during refresh») | Adapter: `Scheme.Mint`. Core runs it on a detached context with its own deadline, under the lock |
| 4 | How the credential is applied to a request | Bearer header; custom header (`X-Api-Key`); basic; query parameter (PolyAI MCP, Retell `query_params`); request signature; TLS client cert | Constant. `AuthorizeRequest` switch on 4 types (`runtime.go:187-204`); api\_key header name with a forbidden list (`api/connectors.go:1050-1057`); query forbidden by design (`connector-design.md:250`) | Adapter: `Scheme.Wrap` on the RoundTripper. Query auth stays forbidden unless a reviewed scheme allows it |
| 5 | Connection parameters and endpoint resolution | Salesforce production vs sandbox hosts and `instance_url`; Shopify `{shop}`; Zoho data center; QuickBooks `realmId`; Microsoft `{tenant}` issuer; 417 of 1,029 Nango entries put connection parameters into the URL | Constant. `Endpoint()` has `if connector.ID == "salesforce"` (`catalog.go:111-120`); `providerOAuthEndpoints` hardcodes the sandbox hosts (`oauth.go:823-830`); only `{instance}` is a template | Data: declared `inputs` on the manifest, `{var}` templates in every endpoint, `inputs jsonb` on the connection |
| 6 | Account identity | Slack team + user; Linear `viewer` query; Google `sub`; Salesforce identity URL; Microsoft `oid` + `tid`; GitHub installation id; Shopify shop domain. Live Slack and Linear returned no id (`connector-handover.md:190-192`) | Constant. `providerAccountID` switch on connector id (`oauth.go:832-844`). No profile request by design (`connector-design.md:159`) | Data: `Identity{From: token_response \| id_token \| profile_request, Path}`. A profile request is allowed only when the manifest says the token's audience permits it |
| 7 | Source of tool schemas | MCP `tools/list` (dynamic, can differ per account); OpenAPI (Eve, Google CES); hand-written operations (ElevenLabs webhook tool, PolyAI APIs tab, Retell custom function); customer backend functions | Constant. Only MCP `tools/list` at session start (`mcp.go:78,210-229`); `cached_tools` on the connection | Adapter: `Source.Discover`. Every source returns the same `ToolSpec` with a digest |
| 8 | Execution mode | MCP over Streamable HTTP; MCP over stdio (Sendblue, Linq; not for a cloud router); plain HTTP with the connection's credential; GraphQL; caller-executed over the WebSocket bridge; sandbox; broker-executed (Composio, Arcade) | Constant. `connectorToolRunner` runs MCP, hands the rest to the caller bridge (`session/mcp_tools.go:17-26`). `tool_timeout_ms` does not bound MCP (`connector-design.md:64`) | Adapter: `Source.Open` returns a `Runtime`; the Dispatcher applies one envelope to all |
| 9 | Inbound events about a grant | Slack `tokens_revoked`; Google RISC `token-revoked`; Microsoft CAE claim challenge; Nango webhook on failed refresh; Shopify `app/uninstalled`; GitHub `installation.deleted` («Incidents and provider quirks») | Absent. The only hook is Stream Chat (`chat/hooks.go`) | Adapter: one endpoint, `Verifier` registry, maps to `Resolver.Invalidate` |
| 10 | Who owns the OAuth client app | Operator (Stream); customer BYO; DCR; CIMD; broker-managed. Rules per provider: Slack MCP only for Marketplace or internal apps; Google CASA; Microsoft publisher verification («Whose OAuth app») | Half an extension point. `oauth_mode` is a string with four values (`oauth.go:147-157`); client from `<ID>_MCP_CLIENT_ID` env (`oauth.go:794-797`); customer id and secret per attempt, sealed with the grant (`api/connectors.go:319-353,843-853`) | Data: `ClientPolicy` on the manifest; an OAuth client record per (app, connector) so BYO is a row, not a per-attempt body field |
| 11 | Connection owner | App; user; agent-owned (ChatGPT, Glean); org-level install (Slack Enterprise Grid, GitHub App installation) | Constant but right. `owner_type IN ('app','user')` (`20260929170000:11,30`). Agent-owned is `app` (competitor doc, «Assistants and IDEs») | Keep two owners. The installation is the account, not a third owner |
| 12 | Scope model | Space-separated; comma-separated (Slack); bot vs user token (`scope` vs `user_scope`, the live Slack bug); GitHub App permissions instead of scopes; Microsoft `.default`; step-up union (MCP 2026-07-28); scope check against tools (Retell misses it) | Constant. `if connector.ID == "slack"` joins with commas (`oauth.go:265-267`). `granted_scopes` stored, never compared («Appendix: fact check») | Data: `ScopePolicy{Separator, SendOnRefresh, StepUpUnion}`; a per-tool `needs_scopes` on the `ToolSpec` so the check is possible |
| 13 | Token endpoint client auth | `none` + PKCE; `client_secret_post`; `client_secret_basic`; `private_key_jwt` (Microsoft, Okta, Apple); `tls_client_auth` | Constant. Three methods (`oauth.go:790-792`, `597-624`) | Adapter detail inside `oauth2_code`: a `ClientAuth` sub-registry; `private_key_jwt` is one file |
| 14 | Discovery of the auth server | RFC 9728 + 8414 (MCP); OIDC; static endpoints; tenant-dependent issuer (Microsoft); region-dependent issuer (Zoho) | Extension point for the MCP case: discovery with catalog overrides (`oauth.go:159-180`). Not for tenant or region templates | Data: issuer and endpoints as templates over `inputs` |
| 15 | Refresh semantics | Rotating with grace (Linear 30 min, Slack short); non-rotating (Google, HubSpot); no refresh token (client credentials); refresh token expiry (Google testing 7 days); needs `scope` (Microsoft); limits (Google 100 per account per client, Salesforce 5 approvals) | One policy: refresh if under 1 minute to expiry, checkpoint to `needs_reauthorization`, no `scope`, no grace retry (`runtime.go:78-139`, `oauth.go:543-548`) | Data: `RefreshPolicy{Margin, Grace, Rotating, SendScope, TokenTTL}` read by the scheme. Reconnect reuses the connection row (axis 6) |
| 16 | Rate limits | Per app (Google, Graph); per workspace (Slack); per org (Salesforce); per user (Linear); `Retry-After` | Absent. `429` in the design is the Router's own quota («Edge cases», last row) | Data: `RateLimitRule{Per}`; `Classify` returns `RateLimited` with `RetryAfter`; a `Limiter` keyed as the rule says |
| 17 | Revocation at the provider | Slack `auth.revoke`; Linear, Calendly, GitHub, Salesforce revoke endpoints (`connector-design.md:157`) | Absent. Local soft delete only (`store/connectors.go:294-321`, `connector-handover.md:127-131`) | Adapter: `Scheme.Revoke`, optional, best effort, reported honestly |
| 18 | Connection test | Zapier requires `test`; Nango `proxy.verification`; Retell `test-app-auth` | MCP-only: `validate` opens MCP and lists tools (`api/connectors.go:669-758`) | Adapter: `Source.Discover` is the test for tool sources; a scheme may add a `Probe` request from the manifest |
| 19 | Call policy (voice envelope) | `response_timeout_secs`, `pre_tool_speech`, `interruption_mode`, `execution_mode` (ElevenLabs); `CANCELLABLE` (LiveKit); `cancel_on_interruption` (Pipecat); no blind retries anywhere | Only `timeout_ms` on the binding (`store/models.go:365`), 5 s default (`connector_tools.go:19`); cancel on interruption in `agent.go:1182-1188` | Core: a `Policy` object on the binding or grant, provider-independent, additive |
| 20 | Consent flow shape | Browser popup bound by cookie (branch); session-bound one-time link for voice («Personal token in a voice channel»); admin consent (Microsoft); install flow with a signed callback (Shopify); GitHub App install then optional user auth; device code (banned) | Constant. One flow: launch page, popup, cookie, callback (`api/connectors.go:31-52,403-488,760-878`) | Core: `Attempt{Kind}` with kinds consent, reconnect, step\_up, admin\_consent; `Scheme.Begin` picks the parameters; `BeforeComplete` hook for a signed callback |

**Reading the table.** Thirteen axes are constants today (1–8, 12, 13, 15, 20 and the identity switch in 6). Three are absent (9, 16, 17). Two are already right (11, 14 for MCP). The pattern is the same each time: a provider name or a fixed list sits in a core Go file. The fix is also the same each time: move the variation into a manifest field or a registered adapter.

## Stress test: 12 awkward providers

Each row says how the provider is expressed in the design above, and whether a hook is needed. «Data» means manifest fields only. «Adapter» means a scheme, source or verifier that already exists in the registry list, or one new file. The last column is the honest answer to «can the branch do this without a core edit». Facts marked **verified** were checked against the vendor's own page or SDK source on October 1, 2026; the pages are listed in «Sources and evidence».

| # | Provider | Quirks | How it is expressed | Hook? | Branch today |
| --- | --- | --- | --- | --- | --- |
| 1 | Salesforce | Production vs sandbox login hosts; `instance_url` in the token response; External Client App; client credentials with a «Run As» user (Retell, ElevenLabs, PolyAI); 5 approvals per user, the oldest is revoked («Incidents and provider quirks»); identity URL `id` | Data: input `environment ∈ {production, sandbox}` drives `authorize` and `token` templates; `capture: instance_url from token_response`; `identity: token_response $.id`; schemes allowed `oauth2_code`, `oauth2_client_credentials`, `oauth2_jwt_bearer`. Reconnect reuses the row, so the 5-approval limit is not hit | No | Core edits: `catalog.go:111`, `oauth.go:824`, `oauth.go:874`. No client credentials |
| 2 | QuickBooks Online | `realmId` arrives as a callback parameter, not in the token body: Intuit's own SDK reads it from the redirect parameters (`oauth-jsclient/src/OAuthClient.js:217`); API bases `sandbox-quickbooks.api.intuit.com` and `quickbooks.api.intuit.com` with `/v3/company/{realm_id}` (`OAuthClient.js:108-109`); access token 3600 s; the refresh token rotates and «previous refresh tokens expire 24 hours after you receive a new one» (README:539-542); the token response carries `x_refresh_token_expires_in`; in November 2025 Intuit replaced the rolling 100-day refresh token with an absolute five-year maximum and a new expiry field (**verified**) | Data: `capture: realm_id from callback_query key=realmId`; `api_base` template with `{realm_id}`; `refresh: rotating true, grace 24h`; `capture: refresh_expires_at from token_response $.x_refresh_token_expires_in` so the connection warns before the grant dies | No | Core edit: the callback reads only `state`, `iss`, `error`, `code` (`api/connectors.go:762-817`) |
| 3 | Zoho | Eight data centers, each with its own accounts host: `accounts.zoho.com`, `.eu`, `.in`, `.com.au`, `.jp`, `zohocloud.ca`, `.sa`, `.uk`; the callback carries `location` and `accounts-server`; the token response carries `api_domain`, «which is where you will need to make the service API requests» (**verified**) | Data: input `region` (enum of 8) in the issuer and endpoint templates; `capture: api_domain from token_response`, `location` and `accounts_server from callback_query` | No | Core edit: no templates beyond `{instance}` (`catalog.go:121-135`) |
| 4 | Shopify | Authorize at `https://{shop}/admin/oauth/authorize`, the shop host must match Shopify's `*.myshopify.com` pattern; callback check: drop `hmac`, sort the other parameters, HMAC-SHA256 with the client secret, constant-time compare, and `state` must equal the stored nonce; scopes are comma-separated; offline tokens for new public apps must be requested with `expiring=1` and come with `expires_in`, a `refresh_token` and `refresh_token_expires_in` of 90 days; online tokens via `grant_options[]=per-user` carry `associated_user` and have no refresh token; header `X-Shopify-Access-Token`; webhooks signed in `X-Shopify-Hmac-SHA256` as base64 HMAC-SHA256 of the raw body with the client secret; mandatory topics `customers/data_request`, `customers/redact`, `shop/redact` (48 hours after uninstall) and `app/uninstalled`; REST bucket 40 requests, leak 2 per second (Plus: 400 and 20), 429 with `Retry-After` (**verified**) | Data: input `shop` with the pattern; `scopes.separator: ","`; `authorize_params: {expiring: 1}`; `refresh: rotating true, token_ttl 90d`; token kinds `offline` and `online` with `identity` from `$.associated_user.id` for online; `api_key` scheme with the header for custom apps (the branch's `auth_header` already does this); `rate_limit: bucket 40, leak 2/s`; `Verifier hmac_header` (base64, raw body) for `app/uninstalled` → `Uninstalled` and the three compliance topics. The callback HMAC is a **hook** at `BeforeComplete`: `shopify.callback_hmac` | **Yes**, one | Core edits: templates, callback, no signals. Shopify was dropped from the catalog (`connector-design.md:56`) |
| 5 | Slack, Enterprise Grid | Org-ready apps install on the org, not on workspaces («For Athena»); user vs bot token (`scope` vs `user_scope`, the live bug in `connector-design.md:32`); comma scopes; rotation every 12 hours with a short grace; `tokens_revoked` event; MCP only for Marketplace or internal apps; limits per workspace per app | Data: `scopes.separator: ","`; two connector entries or one with `token_kind ∈ {bot, user}` selecting the token path (`$.authed_user` vs top level); `identity: $.team.id + $.authed_user.id`, `capture: enterprise_id`; `refresh: rotating, grace short`; `rate_limit.per: workspace`; `Verifier hmac_header` (Slack signing secret) for `tokens_revoked` | No | Core edits: `oauth.go:265` (commas), `oauth.go:106-116,626-633` (`authed_user`), no signals |
| 6 | GitHub App | Three credential kinds: an app JWT signed with the private key, RS256, «no more than 10 minutes into the future»; an installation token from `POST /app/installations/{installation_id}/access_tokens` that «will expire after 1 hour» with no refresh; an optional user-to-server token that lives 8 hours with a 6-month rotating refresh token (expiry is opt-out per app). GitHub appends `installation_id` to the setup URL redirect but says not to trust it: confirm the installation through a user token and the installations API. The `installation` webhook has a `deleted` action; webhooks are signed in `X-Hub-Signature-256` (**verified**). The hosted MCP uses OAuth 2.1 (`connector-design.md:148`) | Adapter: scheme `github_app` where `Material` = private key + app id + installation id and `Mint` signs the JWT and exchanges it; `capture: installation_id from callback_query, verify: profile_request`; `identity` = installation id; `Verifier hmac_header`. The MCP path stays `oauth2_code` | No, one scheme file | Not expressible: no scheme other than refresh |
| 7 | Google Workspace | Restricted scopes need CASA yearly; refresh token in Testing status lives 7 days; 100 refresh tokens per account per client; RISC `token-revoked` events; non-rotating refresh (competitor doc). `access_type=offline` is needed to get a refresh token, `prompt=consent` because «the refresh\_token is only returned on the first authorization», `include_granted_scopes=true` for incremental authorization; the `id_token` claim `sub` is «unique among all Google Accounts and never reused» and Google says to use `sub`, not `email`, as the key; `hd` names the Workspace domain (**verified**); quota per Cloud project | Data: `authorize_params: {access_type: offline, prompt: consent, include_granted_scopes: true}`; `identity: id_token $.sub`; `capture: hd from id_token`; `refresh: rotating false`; `rate_limit.per: app` (so BYO apps shard the quota); scheme `oauth2_jwt_bearer` with a `subject` input for domain-wide delegation; `Verifier jwt_set` for RISC | No | Core edits: no extra authorize params, no `id_token` parsing, no signals |
| 8 | Microsoft 365 | Issuer per tenant `login.microsoftonline.com/{tenant}`; separate admin consent endpoint and risk-based consent («Incidents and provider quirks»); CAE claim challenge on 401 with `claims`; refresh must send `scope` (Claude Code #89862); `.default` scope; Graph 130,000 per 10 s per app. Certificate credentials are `private_key_jwt`: `client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer`, the assertion signed with **PS256** (not RS256), header `x5t#S256`, claims `aud` = the tenant's token endpoint, `iss` = `sub` = client id, `jti`, `nbf`, `exp` of 5 to 10 minutes; usable anywhere a client secret is (**verified**) | Data: input `tenant` in the issuer template; `scopes.send_on_refresh: true`; `client.auth_method: private_key_jwt` with `alg: PS256`; `rate_limit.per: app`. Core: `Attempt{Kind: admin_consent}`; `Classify` in `oauth2_code` parses `WWW-Authenticate` for `insufficient_scope` and `claims` → `ScopeRequired` | No | Core edits: no `scope` on refresh (`oauth.go:543-548`), three client auth methods (`oauth.go:790`), no step-up |
| 9 | Twilio | REST, no hosted MCP in the doc; HTTP Basic auth with an API key SID and secret (Twilio's preferred way) or the Account SID and auth token (required for account management calls); base `https://api.twilio.com/2010-04-01/Accounts/{AccountSid}/`; request bodies are `application/x-www-form-urlencoded` or multipart; a subaccount is reached with the parent account's credentials (**verified**); `X-Twilio-Signature` on inbound callbacks («WhatsApp and Twilio»). Twilio is also a phone vendor in `internal/phone`; the two uses stay separate | Adapter: scheme `basic` (new, one file; ElevenLabs has it); source `http` with operations defined as data: `POST /2010-04-01/Accounts/{account_sid}/Messages.json`, `body: form`; input `account_sid`; `Verifier twilio_signature` | No | Not expressible: no basic scheme, no HTTP source |
| 10 | API with request signing (AWS SigV4, for example Bedrock or Amazon Connect; PolyAI uses the customer's IAM role with `sts:AssumeRole`) | No token. SigV4 derives a signing key «scoped to a single AWS service, in a single AWS region, on a particular day» and puts the signature in the `Authorization` header or the query; temporary credentials also need the security token; a request must reach AWS within five minutes of its timestamp. `AssumeRole` takes `DurationSeconds` (default 3600, 900 to 43,200, capped by the role's maximum session of 1 to 12 hours; one hour under role chaining) and an `ExternalId`, and returns `AccessKeyId`, `SecretAccessKey`, `SessionToken` and `Expiration` (**verified**) | Adapter: scheme `aws_sigv4` where `Material` = keys or role ARN + external id, `Mint` = AssumeRole when a role is set, cached until `Expiration` minus a margin, `Wrap` = sign the final request and add `X-Amz-Security-Token`. The wrapping order rule makes this safe: the signer sees final headers and body | No, one scheme file | Not expressible: `AuthorizeRequest` only sets headers |
| 11 | Service with mTLS | Client cert + key, sometimes with a CA; no token; sometimes mTLS plus a bearer bound to the cert (ElevenLabs `mtls`; Dialogflow CX tools; RFC 8705) | Adapter: scheme `mtls` where `Wrap` returns a transport with `TLSClientConfig`. For cert plus bearer, the connection has `auth_scheme` and an optional `tls_scheme`, composed by `Bound.Transport`. Each connection owns its transport and connection pool | No, one scheme file | Not expressible: one shared egress client (`egress/public.go`) |
| 12 | REST without MCP (Linq iMessage API, a customer's own CRM) | Static key; HMAC-signed inbound webhooks; the MCP server is stdio, which Router cannot run («Voice agents and Router»); operations must be written by hand or generated from a spec; per-line rate limits | Adapter: source `http` with reviewed, digest-pinned operation definitions (ElevenLabs webhook tool, PolyAI APIs tab, Retell custom function pattern); later source `openapi` from a pinned spec (Eve). Scheme `api_key`. Same resolver, same envelope | No | Not expressible: MCP only (`session/mcp_tools.go`) |

**What the stress test adds to the core, once.** These are the only core features the 12 providers need. After them, the list above is data and adapter files.

1. `inputs` on the connection and `{var}` templates in every manifest endpoint (providers 1, 3, 4, 8, 9).
2. `capture` rules from `callback_query`, `token_response` and `id_token` into `metadata jsonb` (1, 2, 3, 4, 6, 7).
3. `identity` rules with the same three sources and an optional profile request (1, 5, 6, 7).
4. The `Scheme` registry with `Begin`, `Complete`, `Mint`, `Wrap`, `Classify`, `Revoke` (6, 9, 10, 11).
5. `Attempt{Kind}` with `step_up` and `admin_consent` kinds, and `ScopeRequired` as an outcome (7, 8).
6. The `Source` registry and the Dispatcher with one envelope (9, 12).
7. The events endpoint and `Verifier` registry (4, 5, 6, 7, 9).
8. `RefreshPolicy` and `ScopePolicy` fields read by the OAuth scheme (2, 5, 7, 8).
9. `RateLimitRule` and `RetryAfter` in `Classify` (4, 5, 7, 8).
10. A per-connection transport for `mtls` (11).
11. Form-encoded bodies in the `http` source (9).
12. One named hook point, `BeforeComplete` (4).
13. A `verify` flag on a `capture` rule: a value taken from a callback query is untrusted until a request with the minted token confirms it. GitHub says this of `installation_id`; QuickBooks binds the token to its `realmId`, so a wrong value fails on the first call (2, 6).

**A manifest, for scale.** Salesforce, written the way the design reads it. The three hardcoded places on the branch (`catalog.go:111`, `oauth.go:824`, `oauth.go:874`) become the `inputs`, `endpoints` and `identity` fields. Scopes and the hosted endpoint are the branch's own (`connectors.yaml:95-109`).

```yaml
id: salesforce
revision: 2
name: Salesforce
inputs:
  - name: environment
    enum: [production, sandbox]
    default: production
vars:
  login_host:
    production: login.salesforce.com
    sandbox: test.salesforce.com
  mcp_path:
    production: platform/mcp/v1/platform/sobject-all
    sandbox: platform/mcp/v1/sandbox/platform/sobject-all
endpoints:
  authorize: https://{login_host}/services/oauth2/authorize
  token: https://{login_host}/services/oauth2/token
  revoke: https://{login_host}/services/oauth2/revoke
  mcp: https://api.salesforce.com/{mcp_path}
  api_base: "{metadata.instance_url}"
schemes:
  - oauth2_code
  - oauth2_client_credentials
  - oauth2_jwt_bearer
client:
  policy: [operator, customer]
  auth_method: client_secret_post
  env: SALESFORCE
scopes:
  list: [mcp_api, refresh_token]
  separator: " "
  send_on_refresh: false
identity:
  from: token_response
  path: $.id
  format: salesforce:{org}:{user}
capture:
  - name: instance_url
    from: token_response
    path: $.instance_url
refresh:
  rotating: false
  margin: 60s
rate_limit:
  per: org
sources:
  - kind: mcp
    endpoint: mcp
  - kind: http
    base: api_base
hooks: {}
```

## One-way doors and two-way doors

A one-way door is a decision that data, public contracts or security invariants depend on. Changing it later means a migration under the encryption key, a re-consent for every user, or a regeneration of ten SDKs. A two-way door sits behind an interface and can wait.

**One-way doors: decide now.**

| # | Decision | Why it is one-way | Evidence |
| --- | --- | --- | --- |
| 1 | A connection is one row per (app, connector, owner, account). Reconnect, scope upgrade and key rotation update that row. They never create a second grant for the same account | Providers count grants per OAuth client: Google 100 refresh tokens per account, Salesforce 5 approvals per user. A second row silently revokes the first. Changing the owner model later made everyone re-authorize at Dust | «Incidents and provider quirks», «Assistants and IDEs» (Dust). The branch already rejects an account switch on reconnect (`api/connectors.go:829-841`) |
| 2 | Credential material is an opaque, scheme-tagged, versioned blob. Public facts (instance, realm, team id, scopes, expiry) are columns or `metadata jsonb` on the connection, outside the blob | The blob is sealed with AAD bound to tenant, connection and revision (`secrets.go:113-115`). Re-shaping fixed fields later means re-sealing every grant under the KEK. A fixed struct made the branch hardcode Slack, Calendly and Salesforce fields (`secrets.go:14-27`, `oauth.go:95-116`) | Axes 2 and 6 |
| 3 | Connector definitions are rows with revisions. Built-ins are seeded from YAML at startup; a connection pins the revision it was created from | Today the catalog is compiled in (`//go:embed`, `catalog.go:11`), so a provider is a release. `connector_definitions` already exists for custom MCP (`20260929180000:11-24`); extending it to built-ins now avoids a backfill later. `connector-design.md:192` asks for definition revisions | Axes 5, 14 |
| 4 | One `Resolver` door. No source, no API handler and no session code opens the store or calls a token endpoint. Mint runs on a detached context with its own deadline, under the per-connection lock, with the checkpoint kept | This is the boundary a broker plugs into (`connector-design.md:409`). The branch has the lock and checkpoint (`store/connectors.go:237-291`) but runs the refresh on the call's context, so an interruption or the 5 s timeout leaves the connection in `needs_reauthorization` (`runtime.go:97-112`, «Incidents and provider quirks»). A boundary that leaks is not a boundary | Axis 3 |
| 5 | Every outbound transport is wrapped in this order: egress policy, then the scheme, then the base. For every source, not only MCP | A signing scheme must see the final request; the egress check must see the final destination. `mcp.go:186-208` does this for MCP. An `http` source that skips it is the SSRF path Retell blocks and MCP security practices warn about | «Edge cases», SSRF row; CVE-2025-6514 |
| 6 | Tool authority is the internal map (binding, connection, source, tool), never the exposed name. Grants list exact tools with a schema digest; no wildcard | The exposed name is presentation (`connector-design.md:407,415`). The binding schema is a public contract in `openapi.yaml` and ten SDKs (197 files changed on the branch); its shape is additive from here: a `policy` object and a `tool_source` kind can be added, a wildcard cannot be removed once offered | «Keep: strong parts of the design» |
| 7 | Owner identity comes only from the trusted principal (`Spec.Caller`, `CallerKind`). A session with more than one verified participant uses app-owned connections by default | A confused deputy in a shared Slack channel is the first blocking case for Athena. AWS warns that an unverified `ForUserId` is a trap. The branch already refuses anonymous and guest callers (`connector_tools.go:123-127`) | «For Athena», «Enterprise platforms», competitor doc P0 item 1 |
| 8 | Two owners: `app` and `user`. An installation (Slack org, GitHub installation, Vercel installation) is the account on a connection, not a third owner | Adding an owner value later changes the CHECK, the API enum, the ownership checks and every SDK. The enterprise platforms model both with two values plus the account («Enterprise platforms»); ChatGPT's agent-owned is our `app` | Axis 11 |
| 9 | No stdio, public HTTPS only. Channels are never a kind of tool; where their transport lives is decided in the two-way door below | Hosting boundary (`connector-design.md:564`). Sendblue and Linq MCP servers are stdio and stay out. **Decided October 5:** the channel bridge in the Router writes to Stream Chat, and Router's message hook wakes the session («Decisions, 2026-10-05», item 1). One connector serves channel credentials and MCP tools, as at Vercel («eve with Vercel Connect»). A channel is still not a tool: the bridge does not call the LLM as a tool, it starts a conversation and delivers the reply. Thierry has treated the two as one concept since September 17 («connections. Touches both chat and ai»). | «iMessage: a channel, not a connector» |

**Two-way doors: postpone, because an interface holds the place.**

| Decision | Behind | When to decide |
| --- | --- | --- |
| Store backend: sealed Postgres, KMS envelope, or a broker (Nango, Vercel Connect) | `Backend` and `Resolver` | After a month of Slack and Linear in staging, from data («Build our own layer or use a broker») |
| Which providers are in the catalog | Manifests | Per customer need. Not a code decision |
| The `http` and `openapi` sources | `Source` | `http` right after MCP (competitor doc P1 item 3); `openapi` when a customer brings a spec |
| Voice policy fields on the binding or the grant | Additive `policy` object | After the first voice customer; open question in the competitor doc |
| Step-up and interactive approval | `Attempt{Kind}` and session events | Step-up with Microsoft or MCP 2026-07-28 servers; approvals with a browser flow (`connector-design.md:461-480`) |
| Provider-side revocation | `Scheme.Revoke` | Per provider, with a token-egress policy (`connector-design.md:161`) |
| Rate limiting implementation | `Limiter` keyed by the manifest rule | When a provider returns the first 429 in staging |
| Signal verifiers | `Verifier` | Slack first (`tokens_revoked`), then Google RISC |
| Tool search over a large catalog | Dispatcher input | Not before voice latency is measured (`eve-connectors-research.md:49`) |
| `provided_arguments` filled by the backend | Per-tool field on the grant | When a workflow needs a tenant id the model must not set (`eve-connectors-research.md:79-93`) |
| Provider unit for each customer, proxy and token export | Provider app record (Slack app secrets in `store.ConnectorConnection`, owner `app`, read through `core.Resolver`), proxy and token operations, an optional `Export` on `Scheme` | Decided October 5 as the target design: one provider unit for each customer, the proxy by default, token export opt-in («Decisions, 2026-10-05», items 4 and 5) |
| Where a channel's transport lives. **Decided October 5:** the channel bridge in the Router writes to Stream Chat; the message hook wakes the session. Rejected: adapters that call the session directly (no shared history), and a bridge in a separate service (a second token store, a second contact map) | Connection, manifest with a new `channel` block, Resolver, the inbound endpoint and Verifier registry; Router's existing message hook | Decided. The first channel is Slack for Athena. Both Slack tokens, bot and user, live in the Router |

## AI-816: keep, change, add

The branch is a prototype and can be redone. Most of it should not be. What follows is file by file, so the diff is predictable.

**Keep as is.**

| What | Where | Why |
| --- | --- | --- |
| Connection, owner, binding, grant model and the `fixed` vs `session` selection | `store/models.go:359-435`, `connector_tools.go:72-139` | Retell and ElevenLabs reached the same shape; the enterprise platforms model the owner the same way |
| Sealed envelope with AAD and KEK versions, lazy rewrap | `connectors/secrets.go`, `runtime.go:63-72` | Correct and needed by every backend |
| Advisory lock, revision compare-and-swap, checkpoint before the refresh | `store/connectors.go:198-291,323-347` | The race and lost-response tests pass («Appendix: fact check») |
| One-use, expiring, sealed attempts; browser binding by cookie; `iss` check | `store/connectors.go:349-445`, `api/connectors.go:403-488,760-878` | Matches AWS session binding and the Arcade user verifier |
| Grants pinned by `schema_digest`; hidden on change | `mcp.go:172-184,105-111` | ElevenLabs `tool_hash` → `needs_review` |
| Grant recheck at call time against the current config | `connector_tools.go:229-288` | Removing a grant blocks the next call on an open session |
| Account-switch rejection on reconnect | `api/connectors.go:829-841` | One-way door 1 |
| Egress validation of every endpoint | `egress/public.go`, `oauth.go:676-690` | One-way door 5 |
| Official Go MCP SDK transport, paginated `tools/list`, size caps | `mcp.go:186-229,26-28` | Design doc warns against extending a handwritten transport |
| Required vs optional bindings, `connector_unavailable` events | `connector_tools.go:185-226`, `session.go:78-80` | Explicit failure is the right default |
| `ConnectorConnectionReferenced` | `store/connectors.go:25-54` | The seed of dependents and delete protection |

**Change.**

| What | From | To |
| --- | --- | --- |
| Auth type | `CHECK (auth_type IN (...))` in migration `20260929180000:18`; switches in `runtime.go:187-204`, `store/connectors.go:72-83` | Free string `auth_scheme`, validated against the registry at write time; optional `tls_scheme` |
| Credential struct | 11 fixed fields (`secrets.go:14-27`) | `Material{Scheme, Version, Payload}`; public values to `metadata jsonb` |
| Catalog | `//go:embed` YAML, `Endpoint()` with a Salesforce branch (`catalog.go:11,110-120`) | `connector_definitions` rows with `revision`, `inputs`, `endpoints`, `schemes`, `identity`, `capture`, `refresh`, `scopes`, `rate_limit`, `hooks`; YAML seeds the built-ins. Generic `{var}` templating |
| OAuth client code | `oauth.go` mixes the scheme, provider rules (`:265`, `:823-830`, `:832-887`) and the token-response struct (`:95-116`) | `schemes/oauth2code`: the scheme only. Provider rules move to the manifest. Token response parsed by path |
| Resolver | `ResolveCredentials` refreshes on the call's context (`runtime.go:102`) | `Resolver.Resolve` with a fast path cache keyed by (connection, revision); `Mint` on a detached context with its own deadline; `RefreshPolicy` from the manifest; `scope` sent when the policy says |
| Request auth | `AuthorizeRequest` switch (`runtime.go:175-205`) | `Scheme.Wrap` composed in `Bound.Transport` |
| Tool runner | `connectorToolRunner{mcp, next}` (`session/mcp_tools.go`) | Dispatcher over all sources with the authority map and one envelope |
| Validate | Opens MCP and lists tools (`api/connectors.go:669-758`) | `Source.Discover` plus an optional scheme probe from the manifest |
| Connector API enum | `auth_mode: oauth_dcr \| oauth_preconfigured \| oauth_customer_credentials \| none \| bearer \| api_key` (`api/connectors.go:969-1008`) | `schemes[]` and `client_policy` from the manifest. Additive in OpenAPI; old values map to new ones |
| BYO client | `oauth_client_id` and `oauth_client_secret` per authorize body, sealed into each grant (`api/connectors.go:319-353,843-853`) | An OAuth client record per (app, connector); the attempt references it. Rotating a customer secret then touches one row |
| Callback | Reads `state`, `iss`, `error`, `code` only (`api/connectors.go:762-817`) | Passes the full query to `Scheme.Complete` for `capture` and the `BeforeComplete` hook |
| Plugin import | `connectorimport/` and the irreversible migration `20260929210000` | Delete the import. Keep the drop. Staging has 0 plugin rows («Where plugins run today»), and Router has no production |

**Add.**

1. The `Scheme` registry and five schemes beyond the four on the branch: `oauth2_client_credentials` (competitor doc P1 item 6), `oauth2_jwt_bearer`, `basic`, `mtls`, `aws_sigv4`. Plus `github_app` when GitHub is a customer need.
2. `private_key_jwt` as a client auth method inside the OAuth scheme.
3. The `Source` registry with `http` (P1 item 3) and the existing caller bridge behind the same interface. `openapi` later.
4. The events endpoint for each provider app, `POST /v1/connectors/events/{provider_app_id}` (proposal), with the `Verifier` registry. It is a second route into the same handler as the per-connector endpoint `POST /v1/agents/connectors/events/{connector_id}` (decided October 5); that route stays for connectors with no provider app for each customer. Slack `tokens_revoked` first (P1 item 11).
5. `Attempt{Kind}`: `consent`, `reconnect`, `step_up`, `admin_consent`. `ScopeRequired` outcome and the `connector_scope_required` event that exists only in `connector-design.md:389` today (P1 items 7 and 10).
6. `Classify` with `RateLimited` and `RetryAfter`; a `Limiter` keyed as the manifest says (P1 item 8).
7. Dependents and delete protection: `used_by` on the connection, no delete without `force` (P1 item 4; Retell `list-app-usages`, ElevenLabs `used_by`).
8. Invocation log with `latency_ms` and `error_type` split into `customer_auth`, `external_server`, `client_timeout`, `outcome_unknown` (P1 item 5, handover item 7).
9. User lifecycle: delete all connections of a user, hard delete of `owner_id` and `account_id` on request (P1 item 9).
10. `Policy` on the binding: `pre_speech`, `on_interrupt`, `async`, `cancellable`, `read_only` (P1 item 2, P2 item 2).
11. A scope check at connect time: `granted_scopes` against the union of `needs_scopes` of the granted tools (P1 item 7).
12. The CI test that keeps provider names out of `core` (next section).

**Add for channels and direct calls (decided October 5). Items 1, 4, 5, 6 and 7 are the connector layer and stay under AI-816; item 2 is the channel bridge, AI-866; items 3 and 8 are the omni-channel conversation, AI-867 (subtasks doc, Phase 7 and the two sections after it).**

1. The `channel` block in `core.Manifest`, and a `core.Verifier` result that can carry a message, not only a `core.Signal`.
2. The channel bridge: the inbound half (verify, thread link, write to the thread channel, episode card) and the outbound half (the reply through the provider API).
3. Episode records: thread channel, episode card, idle close, summary, facts to memory.
4. A provider app record for each (`app_pk`, connector), created with `apps.manifest.create` for Slack. Its events URL is the events endpoint in item 4 of the list above: one handler for token signals and messages.
5. The proxy operation and the token operation, both server-side only, with an audit table.
6. Raw event destinations, signed with a key of each customer.
7. The `api_key` scheme for Telegram, Linq, Sendblue, Twilio and Telnyx.
8. `stream_app_pk` pins and a message hook in each customer app (tenancy).

**Package layout.** `internal/connectors/core` (model, resolver, dispatcher, attempts, policy, records), `internal/connectors/schemes/<name>`, `internal/connectors/sources/<kind>`, `internal/connectors/backends/<name>`, `internal/connectors/signals/<name>`, `internal/connectors/providers` (YAML manifests and the hook files). `internal/mcp` shrinks to the MCP source. `internal/connectors` on the branch becomes `core`.

## Validation plan before writing code

The plan proves the extension points before the product code exists. Each spike is one to two days and ends with a yes or no.

**Spikes.**

| # | Spike | What it proves | Exit condition |
| --- | --- | --- | --- |
| 1 | Manifests for all 12 stress providers, as YAML only, loaded through the registry in a Go test | The manifest format is enough for the awkward cases | All 12 load. Authorize URLs, endpoint templates, `capture` and `identity` resolve against recorded fixtures. Zero provider names in `core` |
| 2 | Port `oauth2_code` from `oauth.go` to the `Scheme` interface; add `oauth2_client_credentials`, `api_key`, `basic` | A second scheme is one file | The four schemes pass the same contract tests. The diff touches no file in `core` |
| 3 | The `http` source with one operation defined as data (a Twilio-shaped `POST` with a form body, against a fake) | A second source is one file; the wrapping order holds | The operation runs through `Resolver` and the egress policy. A private IP in the base URL is refused |
| 4 | Resolver fast path under load: 50 concurrent sessions, 20 calls each, no refresh due | Whether the advisory lock belongs on the hot path | p50 and p95 of `Resolve` are recorded. Target: under 5 ms p95 with the cache, which is a target to test, not a measurement |
| 5 | `mtls` scheme with a per-connection transport | Transport lifecycle and memory under 1,000 connections | Pools close on disconnect; memory is recorded |
| 6 | Replay the branch's refresh race tests through the new resolver | Nothing regressed | `TestConcurrentCredentialResolutionCommitsOneRotatedRefreshToken` and `TestRefreshOutcomeSurvivesLostResponsesAndCanceledWorkers` pass, run with `go test -tags=integration ./internal/api` as in `connector-handover.md:200-202` |
| 7 | Broker adapter behind `Resolver` for one provider (later, only if the data says so) | A broker is a two-way door | A Nango or Connect `getToken` answers `Resolve` without a change to bindings |

**Contract tests, table-driven, run for every registered adapter.**

- `SchemeContract`: `Begin` then `Complete` round trip on the fake provider; `Mint` under concurrency commits one revision; a lost response leaves `needs_reauthorization` and never replays the refresh token; `Classify` maps `invalid_grant`, a 5xx, a timeout, `insufficient_scope`, a `claims` challenge and a 429 to the six outcomes; `Wrap` puts no secret into a URL, a log line or an error string; `Revoke` is best effort and says so.
- `SourceContract`: `Discover` gives a stable digest for the same schema; an ungranted tool is never dispatched; a changed schema is hidden; a result over 32 KiB is cut with the marker; a cancelled call sends the cancel; a timed-out write returns `outcome_unknown`, not an error the model retries.
- `ManifestContract`: every manifest validates; every `{var}` is a declared input or a captured name; every `capture` and `identity` path resolves on the provider fixture; every hook name exists; the number of hooks is reported and reviewed.
- `PolicyContract`: two users and two accounts never cross; a session with two verified participants gets app-owned connections only; anonymous and guest callers get no personal connection; a guessed connection id gives not-found and no outbound call. The branch has the first and third cases (`connector-design.md:534-553`).

**Fake provider.** One Go HTTP server in `internal/connectors/fakeprovider` with switchable personalities: rotating refresh with a grace window, non-rotating refresh, no refresh token, `invalid_grant`, a lost response, `insufficient_scope` on 403, a `claims` challenge on 401, 429 with `Retry-After`, Slack-style comma scopes and `authed_user`, a QuickBooks-style `realmId` in the callback, a Shopify-style signed callback, a SigV4 verifier, an mTLS listener. The branch's fake-provider integration tests (`connector-design.md:571`) are the start of it. Every contract test runs against it; no vendor account is needed until the live checks in `connector-handover.md` items 2 and 3.

**CI rule that keeps provider code out of the core.** CI runs `go vet ./...` and `go test ./...` (`.github/workflows/ci.yml:78-98`); there is no linter config in the repo, so the rule is a Go test:

1. `TestCoreImportsNoAdapter` loads `internal/connectors/core/...` with `go/packages` and fails if any import path contains `/schemes/`, `/sources/`, `/backends/`, `/signals/` or `/providers/`.
2. `TestCoreNamesNoProvider` parses `core/` with `go/ast` and fails on any string literal equal to a connector id from the seeded catalog, and on any literal in a fixed deny list (`salesforce`, `slack`, `calendly`, `shopify`, `google`, `microsoft`, `github`, `gong`, `linear`, `twilio`). The three places on the branch that would fail it today are `catalog.go:111`, `oauth.go:265` and `oauth.go:824-842`.
3. `TestEveryManifestLoads` and `TestHooksAreRegistered`, from the manifest contract above.
4. The existing OpenAPI freshness test stays: `go run ./cmd/openapi` and the committed spec must match.

## Risks

1. **Over-engineering.** Six schemes, three sources and an events endpoint before there is one customer. The guard: the registries and contract tests are built now, but only the four schemes and the MCP source the branch has today ship first. The second scheme and the second source are the proof, not the product. The stress-test manifests live in tests. If spike 1 shows the manifest needs more than `{var}` templates and JSON paths, stop and add a hook, not a template language.
2. **Latency in voice.** The resolver sits on every call, and refresh runs under a Postgres lock (competitor doc, «Risks of our approach» item 3). Nothing is measured yet (`connector-handover.md` item 8). The design answers with a fast-path cache keyed by (connection, revision) and a detached mint. The cache opens a window in which a disconnect is not yet seen; the design doc wants disconnect to block new dispatches at once (`connector-design.md:435`). Spike 4 decides the window: a status check on every call, or a cache of at most one second. Say which in the API docs.
3. **Manifest creep.** Nango's catalog has 57 entries that need code and 417 that put connection parameters into the URL («Brokers and iPaaS»). The pull toward a DSL is real. The rule: templates are substitution only, paths are JSON paths only, everything else is a named hook, and the hook count is reviewed in CI.
4. **More secret kinds.** Private keys, certificates and AWS keys join OAuth grants under the same KEK. A leak of the database plus the key is worse than today. Same envelope, same AAD; KMS is a backend behind the interface when the time comes. Static egress IPs and a maximum grant age stay on the P2 list.
5. **Provider drift.** MCP removed sessions and deprecated DCR in one revision («What MCP is»). Slack changed its Marketplace rule in September. Manifests have revisions and connections pin them, so a drift is a new revision and a reconnect, not a release.
6. **Scope pressure from channels.** Thierry's interest is Slack, WhatsApp and iMessage as channels. On October 1 he named Linq and Chatbase as the omni-channel-plus-connector products that «grew more than we did», and widened the list to «slack, whatsapp, rcs, texting, imessage, maybe telegram». The design shares the base and keeps the first build narrow. Decided October 5: the channel bridge runs in the Router and writes to Stream Chat, and a new channel is a manifest with a `channel` block, not new Router code. The old cost of the bridge, one Slack bot token in two places, is gone: both Slack tokens live in the Router. Keep the «channel or tool» line from the competitor doc in the API docs.
7. **Staging only.** Router has no production deployment, so migrations are cheap and the plugin import can go. The other side: the KEK keyring and a public HTTPS URL for callbacks must exist in staging before any live OAuth test (`connector-handover.md:139-142`).
8. **Facts from outside the competitor doc.** The eight vendor claims in the stress test were checked against vendor pages and SDK source on October 1, 2026; the list is in «Sources and evidence». Three corrected the first draft: Shopify offline tokens for new public apps now expire and come with a 90-day refresh token; Microsoft signs certificate assertions with PS256, not RS256; QuickBooks moved from a rolling 100-day refresh token to an absolute five-year cap in November 2025. None changed the design; each changed one manifest field. Two sources resisted a direct read: Intuit's docs portal renders client-side, so Intuit's SDK source was used, and Intuit's policy post refused the fetch, so it is quoted from Intuit's own search snippet.
9. **Account identity stays unknown for some providers.** Live Slack and Linear returned no stable id (`connector-handover.md:190-192`). Where the manifest cannot name a source, `account_id` is empty and reconnect cannot prove it is the same account. The design keeps the branch's rule: ask the user to confirm and report identity as unverified (`connector-design.md:190`).

## Sources and evidence

**Competitor doc.** [Voice-agent connectors: competitor analysis](competitor-analysis.md), read in full on October 1, 2026. Sections cited by name in «». Where the text and «Appendix: fact check» disagree, the appendix was used.

**Branch code.** [`codex/connector-support`](https://github.com/GetStream/Vision-Agents/tree/codex/connector-support), local tip `cf62af0d`, read on October 1, 2026. Paths are under `acceleration/`:

- `internal/mcp/catalog.go`, `internal/mcp/connectors.yaml`, `internal/mcp/oauth.go`, `internal/mcp/mcp.go`
- `internal/connectors/runtime.go`, `internal/connectors/secrets.go`
- `internal/store/connectors.go`, `internal/store/models.go:359-435`
- `internal/session/connector_tools.go`, `internal/session/mcp_tools.go`, `internal/session/manager.go:293-370`
- `internal/api/connectors.go`
- `migrations/20260929170000_connectors.sql`, `20260929180000_connector_definitions.sql`, `20260929190000_connector_authorization_kek_version.sql`, `20260929200000_session_connector_selections.sql`, `20260929210000_remove_agent_plugins.sql`
- `docs/connector-design.md`, `docs/connector-handover.md`, `docs/eve-connectors-research.md`
- `.github/workflows/ci.yml:78-98` on `accelerate` for the CI commands

**Standards named in the text.** [MCP 2026-07-28 authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization), [RFC 8705](https://www.rfc-editor.org/rfc/rfc8705) (mTLS client auth), [RFC 7523](https://www.rfc-editor.org/rfc/rfc7523) (JWT bearer), [RFC 9700](https://www.rfc-editor.org/rfc/rfc9700) (OAuth security BCP).

**Vendor pages opened on October 1, 2026 for the stress test.** Shopify: [authorization code grant](https://shopify.dev/docs/apps/build/authentication-authorization/access-tokens/authorization-code-grant), [webhook verification](https://shopify.dev/docs/apps/build/webhooks/subscribe/https), [privacy law compliance](https://shopify.dev/docs/apps/build/compliance/privacy-law-compliance), [webhook topics](https://shopify.dev/docs/api/webhooks), [REST rate limits](https://shopify.dev/docs/api/admin-rest/usage/rate-limits). Intuit: [oauth-jsclient README](https://github.com/intuit/oauth-jsclient/blob/master/README.md) and [OAuthClient.js](https://github.com/intuit/oauth-jsclient/blob/master/src/OAuthClient.js); the docs portal at developer.intuit.com renders client-side and returned no text; the [refresh token policy post](https://blogs.intuit.com/2025/11/12/important-changes-to-refresh-token-policy/) redirects to Medium, which refused the fetch, so it is quoted from Intuit's own search snippet. Zoho: [multi data center OAuth](https://www.zoho.com/accounts/protocol/oauth/multi-dc.html). GitHub: [JWT for a GitHub App](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-json-web-token-jwt-for-a-github-app), [installation access token](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-an-installation-access-token-for-a-github-app), [refreshing user access tokens](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/refreshing-user-access-tokens), [about the setup URL](https://docs.github.com/en/apps/creating-github-apps/registering-a-github-app/about-the-setup-url), [webhook events and payloads](https://docs.github.com/en/webhooks/webhook-events-and-payloads). Google: [OAuth 2.0 for web servers](https://developers.google.com/identity/protocols/oauth2/web-server), [OpenID Connect](https://developers.google.com/identity/openid-connect/openid-connect). Microsoft: [certificate credentials](https://learn.microsoft.com/en-us/entra/identity-platform/certificate-credentials). Twilio: [making requests](https://www.twilio.com/docs/usage/requests-to-twilio). AWS: [Signature Version 4](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv.html), [STS AssumeRole](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html).
