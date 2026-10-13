# Operator clients for the 2026-10-14 staging connector pass

Do not commit this file. Do not print secret values. GCP project, cluster, and secret names stay in `~/.claude/projects/-Users-kanat-Projects-stream-Vision-Agents/memory/staging-migration-order.md`. This note only names chart files, env vars, and the `rocky secrets set` key shape.

Source of the manifests: `origin/accelerate` (`e70351d46d992fec2407ac0b170a3270527c0195`). Chart: GetStream/chat `origin/thierry/accelerate-gke`.

Staging public URL is already `https://accelerate.gcp.stream-io-api.com` (`router.publicURL` in the staging values file the memory names). `router.connectorsEnabled` is already true. `router.dashboardBaseURL` is already the agent-dashboard Vercel preview. None of those three need a change for this pass.

## Redirect URI and client metadata (every OAuth connector)

One callback for every connector:

`https://accelerate.gcp.stream-io-api.com/v1/agents/connectors/oauth/callback`

Built as `ROUTER_PUBLIC_URL` with a trailing slash stripped, plus `ConnectorCallbackPath`:

- `acceleration/internal/api/authorizations.go:37` — `ConnectorCallbackPath = "/v1/agents/connectors/oauth/callback"`
- `authorizations.go:50-55` — `connectorCallbackURL`: `strings.TrimRight(publicURL, "/") + ConnectorCallbackPath`. Empty when `ROUTER_PUBLIC_URL` is empty, and then no consent can start.
- Chart writes that env from `router.publicURL`: `infra/charts/accelerate/templates/deployment.yaml:76`.

CIMD client-id URL (served, not used by these 15):

`https://accelerate.gcp.stream-io-api.com/.well-known/oauth-client-metadata`

- `authorizations.go:43` — `ConnectorClientMetadataPath`
- `authorizations.go:789-794` — `ConnectorClientMetadataURL` returns that URL only when `ROUTER_PUBLIC_URL` starts with `https://`
- The document's `redirect_uris` is the callback above (`oauth2code.ClientMetadataDocument`, `client.go:336-340`). `client_name` is `Vision Agents` (`client.go:21`).

None of the 15 list `cimd` in `client.registration`. Sentry's authorization server advertises `client_id_metadata_document_supported` (`sentry.yaml` header, fetched 2026-10-07); the manifest still registers with DCR only (`sentry.yaml:35`). DCR posts `redirect_uris: [callback]` and `client_name: "Vision Agents"` (`client.go:224-229`). No vendor console redirect is required for calendly, calcom, sentry, or linear.

The staging gateway already allowlists `GET /v1/agents/connectors/oauth/callback` and `GET /.well-known/oauth-client-metadata` (`projects/stream-accelerate/proxy.go:327-328` on `thierry/accelerate-gke`).

## How a secret reaches the pod

Operator clients are env vars, not chart values. The router reads `<client.env>_MCP_CLIENT_ID` and `_MCP_CLIENT_SECRET` at each consent (`README.md` env table; `oauth2code/client.go:49-60`). A missing id is "no operator client", and the next registration method is tried. Slack lists only `operator`, so a missing id means consent cannot start (handoff F87).

The router container mounts the whole secret with `envFrom` (`deployment.yaml:104`). The ExternalSecret finds Secret Manager objects by prefix and strips that prefix, so the pod sees the env var name (`infra/charts/accelerate/templates/externalsecret.yaml`, `dataFrom.find` + `rewrite`). The chart comment says a new credential is `rocky secrets set` and a redeploy, not a new line in the template. The handoff's "plus a chart change" for F87 is stale against that template.

Set each value the way `infra/docs/accelerate_launch.md` §6 does. `--region` and the Secret Manager prefix are in the memory file. The `--key` argument is `accelerate-staging-<ENV_VAR>`. Use `--value-file=-` so the value stays out of shell history. Do not run `rocky secrets bootstrap`.

```bash
printf '%s' "$SLACK_MCP_CLIENT_ID" | rocky secrets set --region "$CLUSTER" \
  --key accelerate-staging-SLACK_MCP_CLIENT_ID --value-file=-
```

Repeat for every name in the per-vendor sections. ExternalSecret `refreshInterval` is `1h` (`infra/charts/accelerate/values.yaml`, `secrets.externalSecrets.refreshInterval`). A router process only sees env from its start, so after the ExternalSecret status shows the new keys, restart the router the way the memory file's verify section does. Kanat has to approve that restart (handoff §3: chat-repo infra and staging).

Keys to set, nothing else:

| Env | Vars |
|---|---|
| Slack | `SLACK_MCP_CLIENT_ID`, `SLACK_MCP_CLIENT_SECRET`. `SLACK_MCP_SIGNING_SECRET` only if revoke/uninstall events are in this pass (`connector_events.go:43`, `slack.yaml:148-150`). |
| Google (one client, four connectors) | `GOOGLE_MCP_CLIENT_ID`, `GOOGLE_MCP_CLIENT_SECRET` |
| HubSpot | `HUBSPOT_MCP_CLIENT_ID`, `HUBSPOT_MCP_CLIENT_SECRET` |
| Salesforce | `SALESFORCE_MCP_CLIENT_ID`, `SALESFORCE_MCP_CLIENT_SECRET` |
| Shopify | `SHOPIFY_MCP_CLIENT_ID`, `SHOPIFY_MCP_CLIENT_SECRET` — see the Shopify blocker before creating the app |
| GitHub | `GITHUB_MCP_CLIENT_ID`, `GITHUB_MCP_CLIENT_SECRET` |

No operator env: calendly, calcom, sentry, linear (DCR), gong (customer or DCR), instacart (bearer on the connection, not an env var).

## Slack — blocker

`client.registration: [operator]`, `env: SLACK` (`slack.yaml:54-59`). Scopes, comma-separated (`slack.yaml:63-95`):

`search:read.public`, `search:read.private`, `search:read.mpim`, `search:read.im`, `search:read.files`, `search:read.users`, `files:read`, `files:write`, `emoji:read`, `chat:write`, `channels:history`, `groups:history`, `mpim:history`, `im:history`, `channels:write`, `groups:write`, `im:write`, `mpim:write`, `reactions:write`, `canvases:read`, `canvases:write`, `users:read`, `users:read.email`, `channels:read`, `groups:read`, `mpim:read`, `im:read`, `lists:read`, `lists:write`

These are user-token scopes. Authorize URL is `https://slack.com/oauth/v2_user/authorize` (`slack.yaml:38`). Put them under User Token Scopes, not Bot Token Scopes. `mcp:connect` is a bot scope for Slack's own MCP client and is not in this list ([mcp:connect](https://docs.slack.dev/reference/scopes/mcp.connect)).

Vendor settings ([Slack MCP server](https://docs.slack.dev/ai/slack-mcp-server), fetched 2026-10-12):

- Only Marketplace apps and internal apps may call MCP. An unlisted distributed app is refused. Make an internal app in a workspace Kanat can admin.
- No DCR. Fixed `client_id` / `client_secret`.
- Redirect URL, under OAuth & Permissions, must be the callback above, exact.
- Token rotation on (`slack.yaml:123-129`, `refresh.rotating: true`, `access_ttl: 12h`). Slack: OAuth & Permissions, or manifest `settings.token_rotation_enabled: true`. It cannot be turned off later. With it on, access tokens expire in 12 hours ([token rotation](https://docs.slack.dev/authentication/using-token-rotation)).
- If the app has an IP allowlist, MCP calls from the router are rejected unless the router's egress is on it. Staging egress ranges are not in this note.
- Events (optional for a tool-call E2E): Request URL `https://accelerate.gcp.stream-io-api.com/v1/agents/connectors/events/slack` (`connector_events.go:22`). Needs `SLACK_MCP_SIGNING_SECRET`.

## Google — blocker (one app, four connectors)

`registration: [operator, customer]`, `env: GOOGLE` on all four (`google_calendar.yaml:59-66`, `google_drive.yaml:51-53`, `google_docs.yaml:52-54`, `gmail.yaml:51-53`). One web client. The router prefers a customer client when one is stored; for this pass leave the customer record empty so the operator env is used (`client.go:77-80`).

Scopes the router sends (space-separated):

| Connector | List |
|---|---|
| google_calendar | `https://www.googleapis.com/auth/calendar.readonly` (`google_calendar.yaml:77-78`) |
| google_drive | `https://www.googleapis.com/auth/drive.readonly` (`google_drive.yaml:62-63`) |
| google_docs | `documents.readonly` and `drive.readonly` (`google_docs.yaml:63-65`) |
| gmail | `https://www.googleapis.com/auth/gmail.readonly` (`gmail.yaml:61-62`) |

Each consent also sends `access_type=offline` and `prompt=consent` (`google_calendar.yaml:69-71` and the same block on the other three).

Classification, from the scope pages fetched 2026-10-12:

- `drive.readonly` is restricted ([Drive scopes](https://developers.google.com/workspace/drive/api/guides/api-specific-auth)).
- `gmail.readonly` is restricted ([Gmail scopes](https://developers.google.com/workspace/gmail/api/auth/scopes)).
- `documents.readonly` is sensitive; the Docs page also marks `drive.readonly` restricted ([Docs scopes](https://developers.google.com/workspace/docs/api/auth)).
- `calendar.readonly` is listed on the [Calendar scopes](https://developers.google.com/workspace/calendar/api/auth) page. That table does not label it sensitive or restricted. Sensitivity: unverified.

Vendor settings ([Configure the Google Workspace MCP servers](https://developers.google.com/workspace/guides/configure-mcp-servers), page updated 2026-10-09):

- The servers are in the Workspace Developer Preview. Membership is a prerequisite on that page. Whether Stream's project is enrolled is unverified. Enroll at the preview program the gmail manifest names (`gmail.yaml` setup step, `developers.google.com/workspace/preview`).
- Enable, for these four only: `calendar-json.googleapis.com`, `calendarmcp.googleapis.com`, `drive.googleapis.com`, `drivemcp.googleapis.com`, `docs.googleapis.com`, `docsmcp.googleapis.com`, `gmail.googleapis.com`, `gmailmcp.googleapis.com`.
- Consent screen: Google Auth Platform → Branding. Audience Internal if the Cloud project is in a Workspace org (no verification, no test-user list). Otherwise External, leave publishing status Testing, and add every Google account that will click consent as a test user (cap 100). An External app in Testing issues refresh tokens that expire in 7 days ([OAuth 2.0](https://developers.google.com/identity/protocols/oauth2), [app state](https://developers.google.com/identity/protocols/oauth2/production-readiness/overview)). Do not publish. Publishing with these restricted scopes starts verification and a security assessment, which will not finish before the 14th.
- Data Access: add the four manifest scopes above. Google's own MCP guide also lists narrower Calendar scopes plus `drive.file`, `documents`, and `gmail.compose`. The router does not send those. Scopes that are not on the consent screen and are in the authorize URL are rejected.
- Clients → Create → Web application. Authorized redirect URI is the callback above. Copy the one client id and secret into `GOOGLE_MCP_*`.

Four consents, one client: Calendar, Drive, Docs, and Gmail are four connectors and four resources.

## HubSpot — blocker

`registration: [operator, customer]`, `env: HUBSPOT` (`hubspot.yaml:45-47`). No scopes (`hubspot.yaml:48-49`: the vendor says scopes are not set on the connector). PKCE. MCP URL `https://mcp.hubspot.com`.

Vendor ([integrate with the remote MCP server](https://developers.hubspot.com/docs/apps/developer-platform/build-apps/integrate-with-the-remote-hubspot-mcp-server), fetched 2026-10-12): Development → MCP Connectors → Create MCP connector. Name, redirect URL = the callback, copy client id and secret from the details page. OAuth 2.1 with PKCE. The first redirect URL is the default if several are listed. A Jan 2026 changelog calls the same UI "MCP Auth Apps" ([changelog](https://developers.hubspot.com/changelog/public-beta-self-service-mcp-auth-apps-for-the-hubspot-remote-mcp-server)). If the sidebar says MCP Auth Apps, that is the same object. This is not a legacy public app and not the local developer MCP (`hs mcp`).

Account: a free HubSpot account is enough ([account types](https://developers.hubspot.com/docs/getting-started/account-types)). Development → Testing → Test Accounts → Create developer test account. Build the connector in that test account so it does not sit on a production portal. Signup: the "Sign up" link on [developer platform basics](https://developers.hubspot.com/developer-platform-basics).

## Salesforce — blocker

`registration: [operator, customer]`, `env: SALESFORCE` (`salesforce.yaml:81-86`). Scopes `mcp_api`, `refresh_token` (`salesforce.yaml:87-89`). Connection input `environment`: `production` (default) or `sandbox` (`salesforce.yaml:28-32`; login hosts `salesforce.yaml:39-40`, `login.salesforce.com` and `test.salesforce.com`). A Developer Edition org logs in at `login.salesforce.com`, so the connection input stays `production`.

Vendor (official guide [Create an External Client App](https://developer.salesforce.com/docs/platform/hosted-mcp-servers/guide/create-external-client-app.html) returned 403 to a fetch on 2026-10-12; the steps below are the search extract of that page plus [setup overview](https://developer.salesforce.com/docs/platform/hosted-mcp-servers/guide/setup-overview.html) and the [Headless 360 workshop](https://developer.salesforce.com/workshops/headless-360-workshop/explore-salesforce-hosted-mcps-v1/1-enable-and-connect). Treat the checkbox list as unverified against a full page read):

- External Client App, not a Connected App. The guide says Connected Apps are not supported for hosted MCP.
- Setup → External Client App Manager → New. Enable OAuth. Callback URL = the callback above. Scopes: Access MCP servers (`mcp_api`) and refresh token. Enable PKCE. The workshop also says to issue JWT access tokens; without that, OAuth can succeed and MCP calls fail with `INVALID_JWT_FORMAT`. That JWT checkbox is from the workshop extract, unverified against the guide body.
- The guide says the app can take up to 30 minutes before an MCP client can use it.
- Setup → MCP Servers (API Catalog) → Salesforce Servers → activate `sobject-all`. That is the server the manifest calls (`salesforce.yaml` mcp path `platform/mcp/v1/platform/sobject-all`).
- Consumer Key → `SALESFORCE_MCP_CLIENT_ID`. Consumer Secret → `SALESFORCE_MCP_CLIENT_SECRET`.
- Scratch orgs cannot create an External Client App in Setup. Use a Developer Edition org.

Account: [developer.salesforce.com/signup](https://developer.salesforce.com/signup). Real email, username that looks like an email and is not an existing Salesforce username. Activation email, then a password. The org stays while someone logs in at least every six months ([Trailhead help](https://help.salesforce.com/s/articleView?id=005298906&language=en_US&type=1)).

## Shopify — blocker

`registration: [operator, customer]`, `env: SHOPIFY` (`shopify.yaml:56-57`). No scopes. Input `shop` must match `^[a-zA-Z0-9][a-zA-Z0-9\-]*\.myshopify\.com$` (`shopify.yaml:43-44`). A custom domain is refused. MCP URL is `https://{shop}/api/mcp`.

Conflict, checked 2026-10-12: the manifest's 2026-10-07 fetch of `allbirds.myshopify.com` returned an OAuth authorization server for `/api/mcp`, which is why this connector is `oauth2_code`. Shopify's current Storefront MCP page says `https://{shop}.myshopify.com/api/mcp` does not require authentication, and that catalog and cart tools were removed from that path ([storefront server](https://shopify.dev/docs/apps/build/storefront-mcp/servers/storefront), [migration](https://shopify.dev/docs/apps/build/storefront-mcp)). Customer Accounts MCP is a different path (`/customer/api/mcp`) and wants an app, protected-customer-data access, and a custom domain ([customer accounts](https://shopify.dev/docs/apps/build/storefront-mcp/servers/customer-account)).

Do not create a Partner app until someone GETs `https://<dev-store>.myshopify.com/.well-known/oauth-protected-resource/api/mcp` and it still names an authorization server. If it 404s, this connector cannot complete OAuth and an operator client does nothing. That live check was not done in this pass.

If the well-known document is still there: Dev Dashboard → Create app → Start from Dev Dashboard → release a version with the callback as a redirect URL → install on a dev store → Client ID and secret into `SHOPIFY_MCP_*` ([create apps](https://shopify.dev/docs/apps/build/dev-dashboard/create-apps-using-dev-dashboard)). Dev store: a Partner account or a merchant account, then a development store. Fastest path is the Dev Dashboard "select or create the store" step on install. Partner signup URL was not re-fetched; start at [dev.shopify.com/dashboard](https://dev.shopify.com/dashboard), which is the manifest's `setup.url`.

## GitHub — blocker for OAuth, bearer is a same-day fallback

`registration: [operator, customer]`, `env: GITHUB` (`github.yaml:63-69`). Schemes `oauth2_code` and `bearer` (`github.yaml:54`). Scopes, space-separated (`github.yaml:80-91`):

`repo`, `read:org`, `read:user`, `user:email`, `read:packages`, `write:packages`, `read:project`, `project`, `gist`, `notifications`, `offline_access`

`api.githubcopilot.com/mcp/` does not support DCR (`github.yaml:67-69` comment, citing the host-integration doc). A host that uses OAuth must register a GitHub App or an OAuth App ([policies](https://github.com/github/github-mcp-server/blob/main/docs/policies-and-governance.md), [README](https://github.com/github/github-mcp-server/blob/main/README.md)).

Use an OAuth App, not a GitHub App. The manifest sends classic scopes. A GitHub App user token's `scope` is an empty string ([generating a user access token](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-user-access-token-for-a-github-app), cited in `github.yaml:72-73`). What a GitHub App does with `offline_access` is unverified in the manifest.

OAuth App: Settings → Developer settings → OAuth Apps → New. Authorization callback URL = the callback. Enable expiring user tokens, or rely on `offline_access`, which the manifest requests so the token expires and a refresh token is returned (`github.yaml:75-79`). Create it on a personal account. An org OAuth access policy can block a third-party OAuth App ([GitHub's MCP setup note](https://docs.github.com/en/copilot/how-tos/copilot-in-your-ide/customize-copilot/extend-copilot-with-tools-and-context/set-up-the-github-mcp-server)).

Fallback that needs no operator env: a connection on the `bearer` scheme with a PAT. Fine-grained or classic, with access enough for the tool you call. The PAT is a connection credential, not `GITHUB_MCP_*`.

## Calendly, Cal.com, Sentry, Linear — ready

DCR only. No env vars. The router registers `Vision Agents` with the callback at consent time.

| Connector | Scopes | Account |
|---|---|---|
| calendly | `mcp:scheduling:read`, `mcp:scheduling:write` (`calendly.yaml:49`) | Free account at calendly.com. Console OAuth apps are not used (`calendly.yaml:42-45`). |
| calcom | none (`calcom.yaml:32` and the comment under it) | Free account at [cal.com/signup](https://cal.com/signup). Signup page was not re-fetched this pass; the host is `mcp.cal.com`. |
| sentry | none in the manifest (`sentry.yaml:35-36`). The server advertises `org:read`, `project:write`, `team:write`, `event:write`, `alerts:write` and the router does not ask for them. | Free account at sentry.io. |
| linear | `read`, `write` (`linear.yaml:32`) | Free workspace at linear.app. |

Cal.com signup URL: unverified (not fetched). Sentry and Linear signup URLs: unverified as exact paths; both vendors offer a free account and DCR does the rest.

## Gong — blocker

`registration: [customer, dcr]` (`gong.yaml:38`). No operator env. Scopes `mcp:read`, `mcp:write` (`gong.yaml:43`). The about page says the server is read-only, so `mcp:write` is unverified (`gong.yaml:39-42` comment).

[Create an integration](https://help.gong.io/docs/create-an-integration-to-connect-to-the-mcp-server): a Technical Administrator, Admin center → Settings → Ecosystem → MCP → New MCP server integration. Redirect URI = the callback (one per line). Automatic registration is the DCR path (no client id). Manual registration is the customer client (id and secret stored on the connection, not in env). That choice cannot be changed after submit. Personal vs shared access is also permanent. A paid Gong seat is required; collaborators cannot connect. There is no free signup that yields MCP. If Stream has no Gong org with a tech admin and a paid seat, skip Gong for the 14th.

## Instacart — blocker

`schemes: [bearer]` (`instacart.yaml:25`). No OAuth, no scopes, no env var. The connection holds the API key. The manifest endpoint is production `https://mcp.instacart.com/mcp` (`instacart.yaml:21`). A development key is for `https://mcp.dev.instacart.tools/mcp` and is the wrong host for this connector.

[Get an API key](https://docs.instacart.com/developer_platform_api/get_started/api-keys): Developer Dashboard → API Keys → Create → Production → copy once. [Get started](https://docs.instacart.com/developer_platform_api/get_started/overview) says the average time from access request to a production key is 30-40 days. That will not land before 2026-10-14 unless a production key already exists. The enterprise MCP (OAuth, per-partner URL) is a different product and is not this connector.

## Order for Kanat

Agent drives the browser. Kanat only logs in, approves, and pastes a secret into `printf | rocky secrets set --value-file=-`. Secrets never go in chat.

1. Open Salesforce signup and the Instacart access request first, then leave them. Salesforce needs an activation email and the External Client App needs up to 30 minutes. Instacart will not finish in time; the click only matters if a production key already exists.
2. Slack, while Salesforce mail is in flight. Internal app, user scopes from the list above, token rotation on, one redirect. Copy id and secret. Skip the signing secret unless events are in scope.
3. Google, one project. If it can be Internal, pick Internal. Otherwise External + Testing + his Google account as a test user. Enable the eight APIs. Add the four manifest scopes. Web client, one redirect. One id, one secret, into `GOOGLE_MCP_*`.
4. GitHub OAuth App on his personal account, callback URL, expiring tokens on. Copy id and secret. If this slips, a PAT on a bearer connection still exercises the MCP server.
5. HubSpot. Free account if he has none, then a developer test account, then MCP Connectors (or MCP Auth Apps). One redirect. Copy id and secret.
6. Back to Salesforce. Activate the org, External Client App (not a Connected App), PKCE, `mcp_api` + refresh token, activate `sobject-all`, copy consumer key and secret. Connection input `production`.
7. Shopify only after the well-known check above says an authorization server still exists. Otherwise skip the Partner app.
8. Gong only if a paid seat and a tech admin are already available. Automatic integration, redirect = the callback, personal access. No env var.
9. He approves `rocky secrets set` for the vars that exist, waits until the ExternalSecret shows them, and approves the router restart from the memory file.
10. One dashboard sitting on the staging preview, app 1257545, for the consents. DCR first (Linear, Sentry, Calendly, Cal.com) to prove the callback. Then Slack, the four Google consents on the same account, GitHub, HubSpot, Salesforce. Shopify, Gong, and Instacart only if step 7-8 or an existing Instacart production key made them possible.

Free accounts he can create in that sitting if he does not already have them: Calendly, Cal.com, Sentry, Linear, HubSpot test account, Salesforce Developer Edition, a personal GitHub user, a Slack workspace he admins, a Google account. Gong and an Instacart production key cannot be created in that sitting.
