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
| `ROUTER_AUTH_KEK` (or `ROUTER_AUTH_KEK_V<n>` + `ROUTER_AUTH_KEK_VERSION`) | Seals the credentials. The router refuses to start without it when connectors are on. Changing it makes stored credentials unreadable |
| `ROUTER_PUBLIC_URL` | The provider sends the browser back to `<ROUTER_PUBLIC_URL>/v1/agents/connectors/oauth/callback`. The dashboard shows this redirect URI on each OAuth connector |
| `DASHBOARD_BASE_URL` | The consent popup reports back only to this origin. Unset, it reports to the wrong place and the dashboard waits forever |
| `ROUTER_CORS_ORIGINS` | The dashboard origin, when the browser calls the router directly |
| `<ENV>_MCP_CLIENT_ID`, `<ENV>_MCP_CLIENT_SECRET` | Stream's own OAuth client for a connector whose manifest lists `operator` (for example `SLACK_MCP_CLIENT_ID` for `slack`). Without it, the app must set its own client |

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

## 3. Give an agent the tools

Dashboard: open the agent › **Tools** › **Connectors** › **Add connector**.

1. Pick the connector.
2. Choose whose account: **an app connection** (`fixed`) or **each user's own** (`session`).
3. **Connect** lists the connection's tools. Tick the ones the agent may call. There is no
   wildcard: a tool that is not ticked is never offered.
4. Save.

The agent sees each tool as `<binding name>__<tool>`, for example `github__get_me`.

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
The reply then carries a `connector_authorization` card. **Connect** opens the consent
popup; after consent the agent carries on by itself.

For a session your backend creates, pass the end user's connection in
`connector_bindings: [{"name": "linear", "connection_id": "…"}]`. Leave it out and the
router uses the user's only connected connection to that connector, if there is exactly one.

`agent.yaml`: the sync API accepts `connectors`, but the SDK folder readers do not read a
`connectors:` block yet (`git grep connectors sdks/go/agents sdks/js/src` finds none). Use the
dashboard or the API for now.

## 4. A channel: the Slack bot

`slack_bot` also receives messages. Someone mentions the bot in Slack, and the agent answers
in the thread. It needs more than tools:

1. **A Slack app.** In **OAuth & Permissions › Redirect URLs**, add the router's redirect URI.
   In **Event Subscriptions**, set the Request URL to
   `<ROUTER_PUBLIC_URL>/v1/connectors/events/slack_bot/<slack app id>`. Slack checks it
   once; the router must already know the app's signing secret.
2. **The app's client in the router.** `PUT /v1/agents/connectors/slack_bot/oauth-client`
   with the client id, secret, `provider_app_id` and `signing_secret` (dashboard: Library ›
   Connectors › Slack bot › OAuth client).
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
| The browser console says «blocked by CORS policy», a header is not allowed | The proxy or `ROUTER_CORS_ORIGINS` does not allow the dashboard origin or header | Allow the origin and header there |
| `GET /v1/agents/connections?owner_type=user` answers 400 | The router got no `X-Stream-User-Id`. A proxy in front dropped it | Forward the header for a server-side caller |
| The consent popup finishes, the dashboard keeps waiting | `DASHBOARD_BASE_URL` is unset or another origin | Set it to the dashboard origin |
| «no OAuth client available» on consent | No Stream client (`<ENV>_MCP_CLIENT_ID`) and no app client | Set one of them |
| 409 «Another customer's record already names this provider app» | That Slack app is registered to another customer on this router | Use another Slack app, or remove the old record |
| A 401 from the proxy on `/v1/connectors/events/…` | The proxy does not let the provider's request through | Allowlist the events routes: the router checks the provider's signature itself |
| The events request gets 200, but no reply | The Stream message hook points at another router (`go run ./cmd/phone hooks`), or two configs bind the connection, or the mention came from an app | Steps 4.4 to 4.6 |
| A `session` binding opens with no tools | The user has no connected connection, or the caller is not a verified end user | Connect from Playground; check the user header |
| Validate says failed | The provider refused the credential | Replace the token, or Reconnect for OAuth |

Every tool call is listed under the connection's **Invocations**. Every grant, refresh and
revoke is under its **Audit**.

## Where the code is

| Path | What |
|---|---|
| `acceleration/internal/connectors/` | The connector core, schemes, resolver, built-in manifests. Each package has an `AGENTS.md` |
| `acceleration/internal/api/connectors.go`, `connections.go`, `connector_events.go` | The API |
| `acceleration/internal/channelbridge/` | Inbound messages from a channel to an agent, and the reply |
| `acceleration/internal/session/connector_tools.go` | Bindings turned into tools for one session |
