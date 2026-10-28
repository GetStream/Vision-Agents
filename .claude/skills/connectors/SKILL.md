---
name: connectors
description: Walks a developer through connectors end to end - the terms, turning them on in the router, connecting an account, giving an agent a connector's tools (one app account or each user's own), a channel such as the Slack bot, and what to check when something stays silent. Read before you connect a connector to an agent, set one up locally or on staging, or debug one.
---

# Connectors

A connector lets an agent call tools at a provider (GitHub, Linear, Slack) with an account
that someone connected. The router keeps the credential sealed and adds it to each call.
The agent's own code never sees a token.

Connectors replace plugins. Plugins still work but are deprecated: do not start new work on
them (`plugin` skill).

To add a new connector to the catalog, read the `add-connector` skill.

## Terms

One word, one meaning. The full table is in `acceleration/internal/connectors/core/AGENTS.md`.

| Term | What it is | Example |
|---|---|---|
| connector | One provider the router can reach. Built-in, or the app's own (`custom_…`) | `linear`, `slack_bot`, `custom_demo_deepwiki` |
| connection | One account at one connector, owned by the app or by one end user | the app's GitHub token; Ann's Linear login |
| binding | What one agent config may use from one connector: which connection and which tools | `connectors[]` on the agent config |
| `fixed` | The binding always uses one app connection | the support bot reads one GitHub org |
| `session` | The binding uses the connection of the session's verified end user | each user files Linear issues as themselves |
| provider app | The app the customer has at the provider, for a connector that receives messages | a Slack app with its signing secret |

```
connector (linear)
  └── connection (Ann's Linear login, owner: user ann)
        ▲
binding on agent config "support": name linear, connection.type session, tools [list_issues]
```

## 1. Turn connectors on in the router

| Setting | Why |
|---|---|
| `ROUTER_CONNECTORS_ENABLED=true` | Off by default |
| `ROUTER_AUTH_KEK` (or `ROUTER_AUTH_KEK_V<n>` + `ROUTER_AUTH_KEK_VERSION`) | Seals the credentials. The router refuses to start without it when connectors are on. Changing it makes stored credentials unreadable. A hosted deployment sets `ROUTER_AUTH_KMS_KEY` + `ROUTER_AUTH_KEK_V<n>_WRAPPED` in its place: README, «The key encryption key in Cloud KMS» |
| `ROUTER_PUBLIC_URL` | The provider sends the browser back to `<ROUTER_PUBLIC_URL>/v1/agents/connectors/oauth/callback`. The dashboard shows this redirect URI on each OAuth connector |
| `DASHBOARD_BASE_URL` | The consent popup reports back only to this origin, unless the consent was started with a `return_url` (`createAuthorization`'s, or for a login or step-up the chat asks for, the session's own from `createSession`), which then replaces it. Unset, a consent without one reports to the wrong place and the dashboard waits forever |
| `ROUTER_CORS_ORIGINS` | The dashboard origin, when the browser calls the router directly |
| `<ENV>_MCP_CLIENT_ID`, `<ENV>_MCP_CLIENT_SECRET` | Stream's own OAuth client for a connector whose manifest lists `operator` (for example `SLACK_MCP_CLIENT_ID` for `slack`, `SLACK_BOT_MCP_CLIENT_ID` for `slack_bot`). Without it, the app must set its own client |

Full list: `acceleration/README.md`, «Configuration». Local setup: the `dashboard` skill.
Restart the router after a change.

## 2. Connect an account

In the Volt dashboard: **Library › Connectors**. It has two tabs.

1. **Connectors**: the catalog. Open a connector to see its auth, scopes and redirect URI.
   For an OAuth connector that needs the app's own client, set **OAuth client** (client id
   and secret, write-only). **New connector** adds a custom MCP server.
2. **Connections**: **New connection**. Pick the connector, the owner (the app, or an end
   user id) and the auth: OAuth opens a consent popup; a bearer token or API key is typed
   in. **Validate** checks the credential at the provider.

The same with the API (operations in `acceleration/api/openapi.yaml`):

| Step | Operation |
|---|---|
| The catalog | `GET /v1/agents/connectors` |
| The app's own OAuth client | `PUT /v1/agents/connectors/{id}/oauth-client` |
| A custom MCP connector | `POST /v1/agents/connectors` (`id` starts with `custom_`) |
| A connection | `POST /v1/agents/connections` (`connector_id`, `owner`, `auth_scheme`) |
| OAuth consent | `POST /v1/agents/connections/{id}/authorizations` → open `launch_url` in a popup |
| A token or key | `PUT /v1/agents/connections/{id}/credentials` |
| Check it | `POST /v1/agents/connections/{id}/validate`, then `GET /v1/agents/connections/{id}/tools` |

### Direct calls from your backend

Your backend can call the provider's own API with a connection's credential:
`/v1/agents/connections/{id}/proxy/<path>` sends `<path>` under the connector's `api_base`.
Only a connector whose manifest has an `api_base` takes direct calls.

| Built-in | `api_base` | Example path |
|---|---|---|
| `github` (revision 4) | `https://api.github.com` | `repos/octo/hello/issues` |
| `slack_bot` (revision 5) | `https://slack.com/api` | `chat.postMessage` |
| `linear_app` (revision 1) | `https://api.linear.app` | `graphql` (POST a GraphQL body) |

A provider's `Link` header (RFC 8288) that names a URL under `api_base`, such as GitHub's
`rel="next"` page, comes back naming the same path and query under this connection's proxy
URL (`ROUTER_PUBLIC_URL` + `/v1/agents/connections/{id}/proxy`), so a client that follows it,
Octokit's paginate included, stays on the proxy. URLs in a body (`url`, `comments_url`) still
name `https://api.github.com`: never send the router's credential there; call their path
through the proxy. With `ROUTER_PUBLIC_URL` unset the router has no proxy URL to name, so it drops every
`Link` (fail closed): a connection then loses GitHub's pagination until the URL is set.

Every other built-in, `linear` included, is MCP-only: a direct call answers 400 «connector
"linear" has no api_base endpoint». Why Linear has none is in `linear.yaml`.

For Linear's GraphQL API, connect `linear_app` («Linear (own OAuth app)») instead. It signs in
with your own Linear OAuth app, and its token serves both the GraphQL API and Linear's MCP
tools:

1. Read the callback URL: `GET /v1/agents/connectors/linear_app`, field `redirect_uri`.
2. In Linear, create an OAuth app at https://linear.app/settings/api/applications/new with
   that callback URL.
3. Give the router its client id and secret: `PUT /v1/agents/connectors/linear_app/oauth-client`.
4. Create a `linear_app` connection and run its OAuth consent as for any connector.

Gong needs its tech admin first. In Gong: **Admin center › Settings › MCP** (under Ecosystem) **› MCP
servers › New MCP server integration**
([Gong's guide](https://help.gong.io/docs/create-an-integration-to-connect-to-the-mcp-server)).
Set the redirect URI to the connector's `redirect_uri`, and the registration type to
**Automatic** for `dcr` (Manual issues your own client id and secret; the type cannot be changed later). Until then Gong refuses every consent with
`unauthorized_client`, «This MCP client is not authorized for your company». The connection
stays pending, and its last validation says why.

A `linear` connection stays as it is; it does not move to `linear_app`. A connection keeps
the revision it was made on, so a `github` connection made before revision 4 takes direct calls
only after its next consent or token save.

## 3. Give an agent the tools

Dashboard: open the agent › **Tools** › **Connectors** › **Add connector**.

1. Pick the connector.
2. Under **Whose account**, choose **One app connection** (`fixed`) or **Each user’s own** (`session`). The list then shows «App connection <name>» or «Each user’s own account».
3. **Connect** lists the connection's tools. Tick the ones the agent may call. There is no
   wildcard: a tool that is not ticked is never offered.
4. Save.

The agent sees each tool as `<binding name>__<tool>`, for example `github__get_me`.

`github` (revision 5) offers GitHub's Actions tools beside its default ones: `actions_list`,
`actions_get`, `actions_run_trigger` and `get_job_logs`. The manifest asks for them with the
`X-MCP-Toolsets: default,actions` header. A binding's `tools` still narrows them: a tool that is
not granted is never offered. A connection made before revision 5 lists them after its next
consent or token save.

With the API, the binding goes in `connectors[]` of the agent config
(`AgentConnectorBinding`):

```json
{
  "name": "linear",
  "connector_id": "linear",
  "connection": {"type": "session"},
  "tools": [{"name": "list_issues"}]
}
```

A `fixed` binding names `connection.connection_id` and pins each tool's `schema_digest`.
A `session` binding may grant a tool by name alone.

Test it in **Playground**. For a `session` binding the end user may have no connection yet.
The call then waits for them to sign in on its own `ai_tool_call` step, which is
`awaiting_authorization` and carries an `authorization` with the `provider`, `title`, `url`,
`handoff_token` and `expires_at`. The reply also carries a `connector_authorization` card, which
the dashboard renders. **Connect** opens the consent popup, and once they consent the same call
runs in the same reply. A `tool_approval` with only the `tool_call_id` and `allowed: false`
declines it, and a consent that is refused, fails or is not finished in ten minutes ends it.
Either way the model is told, and nothing runs. A native app opens
`url#handoff_token=<handoff_token>` in its web authentication sheet, with its own scheme as the
session's `return_url`. On iOS that is StreamChatAI's `AIToolAuthorizationView`, which
`examples/voice_agents/swift_assistant` uses for Gmail and Linear. A spoken turn on a call has
no reply to show it on, so by voice the binding answers that it is not available.

`require_approval: ["create_issue", "delete_*"]` on a binding holds each call of a tool it names
until the session's user allows it. The call's step is `awaiting_approval`, addressed to them, and
nothing reaches the provider before they answer with `tool_approval` (`tool_call_id`, the reply's
`request_id`, `allowed`), on their socket or their backend's. Declined, or not answered in ten
minutes, the call is not sent and the model is told. A session with nobody to ask (incognito, a
run, a shared chat, an MCP event's) refuses the call at once. An entry must match a tool the
binding grants, and `sandbox: true` is refused beside it (`session/connector_approval.go`).
Beside a sensitive connector (Nessy), a binding's write tool needs `require_approval`: the
config is refused unless each tool it grants only reads (`read_only_tools`) or is held, and a
`tool_patterns` binding needs an entry such as `*` that matches each pattern as written
(`api/configs.go`, `sensitiveBindingsComplaint`). In agent.yaml it is `require_approval:` on
the connector entry. Tools a caller declares at createSession are not checked. The caller runs
them, or a person's device does (`executor: client`). The router sends a client tool's
arguments to the caller on the events socket (`api/sessionws.go`). Every channel member can
also read them in the conversation's tool part, but only when they are 512 bytes or fewer
(`conversation/parts.go`, `maxClientArguments`) (AI-1642). If one of them
sends data on to anyone else, declare it with `approval`. On a voice call the router then asks
the user and sends the call only on yes. In text the router does not ask: the caller's client
must wait for `tool_approval` before it runs the tool.

For a session your backend creates, pass the end user's connection in
`connector_bindings: [{"name": "linear", "connection_id": "…"}]`. Leave it out and the
router uses the user's only connected connection to that connector, if there is exactly one.

`agent.yaml`: the sync API accepts `connectors`, and the Go SDK folder reader reads the block
(`sdks/go/agents/config.go`, `connectorBindings`). An entry whose `connection:` the app does not have
is left out with a warning, and the rest of the agent syncs; `required: true` makes it an error. `sandbox: true` lets code in the session's sandbox call the connector's API with the binding's connection.
The left-out entries go to the router as `left_out_connectors` and are served on the agent config.
Other SDKs: unverified, check `git grep connectors sdks/js/src plugins/stream` before relying on it.

### Sandbox access

A binding with `sandbox: true`, on a config with `sandbox: daytona`, lets `run_code` call the
connector's API. Code reads a placeholder (`dtn_secret_…`) from the env var the manifest's
`sandbox.env` names, such as `GITHUB_TOKEN`, and sends it as `Authorization: Bearer $GITHUB_TOKEN`
over HTTPS to a host in `sandbox.hosts`. Daytona's egress swaps it for the token there
(https://www.daytona.io/docs/en/secrets.md); code never reads the token.

- With the Daytona broker, the router needs `ROUTER_CODE_SANDBOX_SECRET_PREFIX` (`stg` on staging; lowercase letters and
  digits only, so one router's sweep never matches another's); without it a session
  with such a binding is refused. Each session's secret is `va_<prefix>_<session>_<binding>_<8 hex>`.
- A prefix belongs to one deployment (one Redis), and `stg` is never used locally: routers that
  share a prefix delete each other's secrets.
- With the Daytona broker, the grant's hosts join the sandbox's domain allow list; a config with
  `network: true` or a `network_allow_list` keeps its own.
- A connection saved before its connector had a `sandbox` block (GitHub revision 5) gives the
  sandbox nothing, with a WARN «reconnect to use sandbox access»; a `required` binding refuses
  the session.
- Google Drive has a block from revision 5 (`GOOGLE_DRIVE_TOKEN`, host `www.googleapis.com`; AI-1958),
  so its connections of revisions 2-4 give the sandbox nothing until reconnected. Its token
  carries `drive.file`, so code can also create files and change files the app opened. A
  `session` Drive binding of a user with no connection gets a late grant (below).
- Linear has a block on `linear_app` from revision 2 (`LINEAR_ACCESS_TOKEN`, hosts `api.linear.app`
  and `uploads.linear.app`; AI-2082): code sends `Authorization: Bearer $LINEAR_ACCESS_TOKEN`,
  and under the proxy broker a POST to `/graphql` passes only with queries; a mutation gets 403.
  Its connections of revision 1 give the sandbox nothing until reconnected. `linear` (DCR) has
  no block: its token, issued for `mcp.linear.app`, gets 401 from `api.linear.app/graphql`
  (probe 2026-10-27).
- BigQuery has a block from revision 4 (`BIGQUERY_TOKEN`, host `bigquery.googleapis.com`; AI-2114):
  code reads with GETs and with jobs.query (`client.query_and_wait`), and under the proxy broker a
  POST to `/bigquery/v2/projects/*/queries` passes only when BigQuery's dry run of it says
  `SELECT`; a script, DML, DDL, `EXPORT DATA` and jobs.insert (`client.query()`) get 403. Its
  connections of revisions 1-3 give the sandbox nothing until reconnected.
- A `session` binding whose user has not connected yet, in a chat they can log in in, gets a late
  grant: the env var is there from the start, built from the connector's latest `sandbox` block,
  and a request with it gets a 401 (the proxy's «gives no token», or the host's own under Daytona).
  `run_code`'s description tells the model to call `<alias>__list_tools`, which shows the consent
  card and waits. In a chat the run itself shows the card too, holding nothing, one open consent per binding;
  the session reads the connection every 3 s and carries on once it connects (AI-1982). Until the user
  has logged in, every `run_code` result adds a line that says so,
  whatever the code did: if the code needed the binding (a 401 from the grant's hosts), call
  `<alias>__list_tools` (`Result.Unconnected`). Once the user consents, the next run
  binds the grant: at once behind the proxy, within 18 to 24 s as a Daytona secret, whose run that
  binds it says to wait 20 s on a 401 (`Result.Connected`; a run the login finished during says it
  too). A required binding still refuses the session, and a voice
  session gets no grant.
- Before each run a new credential revision updates the secret; a connection that needs a
  reconnect, or was deleted, withdraws it, and so does a binding the agent config dropped, no
  longer marks `sandbox`, or binds to another connector or connection (AI-1686, as the tool
  recheck refuses a call). A withdrawn grant stays withdrawn for the session, even when the binding is put
  back; the next session gets it. The proxy broker also checks the binding on every request, so
  the placeholder stops at once; it costs one agent config read per request. Daytona refuses a request that carries a placeholder it
  does not substitute (another host, the query, another session's): 502 «could not reach upstream
  host».
- Daytona substitutes `Authorization: Basic <placeholder>` and `Basic user:<placeholder>` to a
  granted host when the value is not Base64-encoded (live Basic-auth check, staging 2026-10-21).
  It also substitutes inside a Base64-encoded Basic header, which is how git sends it (live
  staging E2E, 2026-10-22, session `01a16514-923c-7843-9214-40a94453f194`: `git ls-remote` of a public repo with
  `Authorization: Basic b64(x-access-token:$GITHUB_TOKEN)` exited 0, with a fake placeholder it
  got 502). Daytona's docs say it does not («Transformed placeholders»); the live result wins.
- Git over HTTPS under both brokers: `run_code` tells the model `git clone https://x-access-token:$GITHUB_TOKEN@github.com/<owner>/<repo>.git`.
  Shown live under Daytona: `git ls-remote` of a public repo with `x-access-token:<placeholder>` as Basic
  auth (see above). `unverified` under Daytona: `git clone` (upload-pack POST), `git push`
  (receive-pack), a private repo, a user name other than `x-access-token`. The proxy broker's
  tests cover its clone (`brokertest` A15) and refuse its push.
- Read-only under the proxy broker (`code_sandbox.broker: proxy`): code in the sandbox reads a
  granted host and writes through the connector's tools. A request to a granted host that is not a
  GET or a HEAD gets 403 «… refused: code in the sandbox only reads through a connector; write with
  the connector's tools», unless the connector's latest `sandbox.reads` lists it (GitHub revision 8:
  `POST github.com/<owner>/<repo>.git/git-upload-pack`, so clone and fetch work; push is refused).
  A read with `graphql: true` takes a POST whose every GraphQL operation is a query (GitHub
  revision 9: `POST api.github.com/graphql`, so `gh`'s queries work and its mutations get the 403).
  A read with `sql: true` takes a POST whose dry run the host says is one SELECT (BigQuery
  revision 4: jobs.query).
  Connections pinned to an earlier revision get the reads without a reconnect. Allowed domains
  take any method. Under the Daytona broker nothing checks methods: code can write with the token
  (`internal/sandbox/egress/AGENTS.md`, «Read-only»).
  GitHub refuses the token as `Authorization: Bearer` from git (401 to `git-receive-pack`, probe
  2026-10-22T03:53:09Z; Basic `x-access-token:<token>` was taken), so `run_code`'s description
  gives git the URL form. GitHub revision 7 grants github.com (AI-1576); a connection on
  revision 6 keeps its two API hosts until it is reconnected.
- `sandbox: true` lets the code call every API the token's scopes allow, outside the binding's tool
  grants, the connector limiter and the audit. That is intended; under the proxy broker the code
  only reads (above).
- Known limit (A5, accepted for GitHub): Daytona scrubs only the token's literal value. A granted
  host that echoes the token escaped or encoded would hand it to the sandbox. Wave 14's proxy
  broker closes it.
- Known limit (Daytona lag): after the router withdraws a secret (connection deleted or needs a
  reconnect), Daytona keeps substituting it for a while. Daytona's docs give no time for a delete;
  an update takes effect within 15 seconds (https://www.daytona.io/docs/en/secrets.md). On
  staging (v0.6.103, 2026-10-21) a request got 200 at 16 s and 502 at 36 s after the router
  withdrew the secret, so the lag is between 16 and 36 s. Wave 14's proxy broker resolves per
  request and stops substituting at once.
- Known limit: a router crash, or a Close that Daytona refuses, leaves the secret (a live
  token) and maybe the sandbox at Daytona until the sweep (w15-sweep, AI-1577) removes it. A
  Close that times out behind a running run deletes the secret (or the proxy's record) at
  once; the sandbox stays.
- A provider's error text never reaches a log or an error: both brokers log the error's class
  (`sandbox.ErrorClass`), and a required grant fails with `sandbox.ErrNoCredential` or the class.
- `ROUTER_CODE_SANDBOX_BROKER=proxy` uses the router's egress proxy (`internal/sandbox/egress`)
  instead of Daytona Secrets. It needs Redis, `ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL` (the proxy
  as a sandbox reaches it, `http://<public IPv4 address>:port`: no path, query, user, or private,
  loopback or link-local address) and `ROUTER_CODE_SANDBOX_PROXY_CA_CERT`.
  Every router with this broker serves the proxy too, so it also needs
  `ROUTER_CODE_SANDBOX_PROXY_ADDR` and `ROUTER_CODE_SANDBOX_PROXY_CA_KEY`; without them it does
  not start. A host name, an IPv6 address or a non-public address in the public URL also stops the start. It needs no
  secret prefix.
- An `https` `ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL` runs a TLS tunnel in the sandbox (`internal/sandbox/egress/AGENTS.md`): the proxy env vars name `127.0.0.1:3128`, and the image needs `python3`.
- How egress is forced: Daytona's network allow list holds the proxy's address alone
  (`<address>/32`). Daytona gets no domain allow list and no `OutboundProxyUrl`. The sandbox then
  reaches the proxy and, of Daytona's own network, only the netleash gateway (172.20.0.1:18080,
  which answers 403 without an `OutboundProxyUrl`), by TCP or UDP, and TLS passes as it is (live
  probe and CIDR probe, 2026-10-22). Either Daytona setting brings in its netleash. Under a domain allow list, netleash
  ends TLS itself and refuses the proxy's certificate for a granted host (staging E2E,
  2026-10-22: `tls: bad certificate`).
- Behind the proxy, the sandbox gets a placeholder (`va_ph_…`) in the env var. Every command and
  run of code gets the proxy and the CA in its env, since netleash replaces the env vars a
  sandbox is created with, but not those of one command or run (live probes, 2026-10-21):
  - `HTTPS_PROXY`, `HTTP_PROXY`, `https_proxy` and `http_proxy` are
    `http://sbx:<credential>@<proxy address>:<port>`; `NO_PROXY` and `no_proxy` are
    `localhost,127.0.0.1,::1`;
  - `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` and `GIT_SSL_CAINFO` name
    `/tmp/va-egress-ca-bundle.pem`, and `NODE_EXTRA_CA_CERTS` names the proxy's CA certificate,
    `/tmp/va-egress-ca.pem`;
  - `GIT_HTTP_PROXY_AUTHMETHOD=basic`: git sends the proxy credential on its first `CONNECT`
    instead of waiting for a 407 (git-config(1), `http.proxyAuthMethod`:
    https://git-scm.com/docs/git-config#Documentation/git-config.txt-httpproxyAuthMethod).
  Before any code runs, the router writes the bundle: the image's own roots and that CA.
  Daytona's image is Debian 13, whose roots are `/etc/ssl/certs/ca-certificates.crt` (live
  probe, 2026-10-21). An image with no roots file runs no code. The proxy puts the token in place
  of the placeholder in `Authorization` (Bearer, `token`, or Basic), so git works too:
  `git push https://x-access-token:$GITHUB_TOKEN@github.com/<owner>/<repo>.git`. The pushed
  content is never rewritten: a file holding the placeholder arrives with it (A15, `brokertest`).
- DNS behind the proxy: the sandbox has none. The proxy looks up each host. A client that does
  not read the proxy env vars (a raw socket, a DNS lookup) reaches nothing. Plain HTTP through
  the proxy is refused.
- The proxy does one of three things with a `CONNECT`:
  - a host a grant covers: it ends TLS with its own CA and puts the token in;
  - a host of `allowed_domains` only: it sends the bytes on as they are, with no token, and reads
    none of them;
  - any other host, a non-public address or a port other than 443: it refuses.
  A host both cover is a grant's. A config with `network: true` and no `allowed_domains`, or with
  a `network_allow_list`, is refused when a session with a sandbox grant starts: the proxy reaches
  hosts by name only.
- Known limit (accepted, Kanat 2026-10-22): the sandbox's proxy credential is in the env of every
  command and run, so sandbox code can read it and send it out to a granted host or an allowed
  domain. It is a proxy credential, never a token. It is the sandbox's own, it expires with the
  sandbox's record, and it reaches only the sandbox's grants and allowed domains.
- Sandbox code has root (`sudo` with no password, live probe 2026-10-21). No control inside the
  sandbox protects anything: code can change its env, its CA files and its routes. The controls
  are the token, which stays in the router; Daytona's network allow list, which lets the sandbox
  reach the proxy alone (a route change as root: unverified); and the proxy's grant check.
- A connection the proxy cannot get a token for just now (a provider outage) keeps its grant; a
  connection that needs a reconnect gives the sandbox nothing, and refuses a `required` binding.
- Revocation through the proxy is immediate: it resolves the token on every request and keeps no
  copy. A deleted or disconnected connection fails the next request.
- `go test -tags daytona_live -run TestLiveSuite ./internal/sandbox/daytona` runs the attacks
  against Daytona with `DAYTONA_API_KEY` and canary values.

## 4. A channel: the Slack bot

`slack_bot` also receives messages. Someone mentions the bot in Slack, and the agent answers
in the thread. It needs more than tools:

1. **A Slack app.** Create it. In **OAuth & Permissions › Redirect URLs**, add the router's
   redirect URI. Do not set the Request URL yet.
2. **The app's client in the router.** `PUT /v1/agents/connectors/slack_bot/oauth-client`
   with the client id, secret, `provider_app_id` and `signing_secret` (dashboard: Library ›
   Connectors › Slack bot › OAuth client). Do this before the next step: Slack checks the
   Request URL at once, and the router must already hold the signing secret.
   Then, in the Slack app's **Event Subscriptions**, set the Request URL to
   `<ROUTER_PUBLIC_URL>/v1/connectors/events/slack_bot/<slack app id>`.
3. **A connection.** New connection › Slack bot › install the app into the workspace.
4. **One live agent per connection.** Bind it `fixed` on exactly one agent config. With two,
   the bridge does not know which agent answers.
5. **The Stream message hook.** The reply runs when Stream delivers `message.new` for the
   thread to this router. Point the app's hook at it, then check:

   ```bash
   cd acceleration
   go run ./cmd/phone hooks                     # read only: shows every hook on the app
   go run ./cmd/phone hooks -url <ROUTER_PUBLIC_URL> -app <stream app id>
   ```

   `hooks` needs `STREAM_API_KEY` and `STREAM_API_SECRET` in the environment. When the router
   runs per-app tenancy, `-app <stream app id>` is required, and the hook path is
   `/v1/chat/hooks/stream/<stream app id>`. Saving the app's own Slack OAuth client
   (`PUT …/oauth-client`) does not point the hook. Run `hooks`, or set it in the dashboard.

   Hooks are one setting for the whole Stream app, and the command also points the call
   hooks. Look first, and change only your own. In the dashboard they are under
   **Chat › Settings › Webhooks**.
6. **Test with a mention typed by hand.** A message posted through another Slack app (an
   MCP tool, a script) carries `app_id`, and the bot skips it on purpose, so bots never
   answer each other.

## 5. When it does not work

Check the router log first (`docker compose logs -f router` locally).

| What you see | Cause | Fix |
|---|---|---|
| `GET /v1/agents/connections?owner_type=user` lists every user's connections instead of one user's, or the browser console says a header is not allowed | The router got no `X-Stream-User-Id`, which lists every user's. A proxy or gateway in front dropped it | Forward the header for a server-side caller |
| The consent popup finishes, the dashboard keeps waiting | `DASHBOARD_BASE_URL` is unset or another origin | Set it to the dashboard origin |
| «no OAuth client available» on consent | No Stream client (`<ENV>_MCP_CLIENT_ID`) and no app client | Set one of them |
| 409 «Another customer's record already names this provider app» | That Slack app is registered to another customer on this router | Use another Slack app, or remove the old record |
| A 401 on `/v1/connectors/events/…` with the text `api_key is required` | The gateway in front does not allow the route. The router's own refusal is a different body: it checks the provider's signature | Allowlist the events routes on the gateway |
| The events request gets 200, but no reply. The router you expect logs nothing, and the router the hook points at logs «an arriving message's channel names a config nobody in its app holds» | The Stream message hook points at another router. Read it in **Chat › Settings › Webhooks**, or `GET` the app's `event_hooks`. Do not wait for a startup warning: in app tenancy it is skipped | Point the hook (step 4.5). Also check that two configs do not bind the connection, and that the mention did not come from an app (steps 4.4, 4.6) |
| A `session` binding opens with no tools | The user has no connected connection, or the caller is not a verified end user | Connect from Playground; check the user header |
| Validate says failed | The provider refused the credential | Replace the token, or Reconnect for OAuth |
| The consent fails, and the connection's last validation has code `connector_consent_refused` with `unauthorized_client` | The provider does not let this client sign in for the organization. For Gong, its tech admin has not added the router as an MCP integration | An admin at the provider allows the client with the router's redirect URI (for Gong, see step 2), then consent again |

Every tool call is listed under the connection's **Invocations**. Every grant, refresh and
revoke is under its **Audit**.

## Where the code is

| Path | What |
|---|---|
| `acceleration/internal/connectors/` | The connector core, schemes, resolver, built-in manifests. Each package has an `AGENTS.md` |
| `acceleration/internal/api/connectors.go`, `connections.go`, `connector_events.go` | The API |
| `acceleration/internal/channelbridge/` | Inbound messages from a channel to an agent, and the reply |
| `acceleration/internal/session/connector_tools.go` | Bindings turned into tools for one session |
