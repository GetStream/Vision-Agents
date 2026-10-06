---
name: plugin
description: How to add an MCP plugin (a hosted MCP server like Sentry, Slack or Google Calendar) that an agent names in agent.yaml, connected once for the app or once per end user. Read before adding a plugin to the catalog or changing how plugins log in.
---

# MCP plugins

A plugin is a hosted MCP server from the router's built-in catalog. An agent names it in
`agent.yaml`; the router logs in with OAuth, stores the token, opens the MCP session when a
conversation starts and offers the server's tools to the model. Nothing in the agent's own
code knows a plugin is there. `examples/text_agents/mcp_plugins` is the working example.

The code is [`internal/plugins`](../../../acceleration/internal/plugins): the catalog
(`plugins.yaml`, `catalog.go`), the OAuth 2.1 + PKCE client (`oauth.go`), the streamable
HTTP MCP client (`mcp.go`) and the per-user tools (`user.go`). Sessions wire it in
[`session/plugin_tools.go`](../../../acceleration/internal/session/plugin_tools.go), the API
in [`api/plugins.go`](../../../acceleration/internal/api/plugins.go).

## Three ways a tool gets its auth

| Shape | Example | How |
| --- | --- | --- |
| Once per app | Sentry: the company's issues | `agent_plugins:` in `agent.yaml`; connected on the dashboard |
| Once per end user | Google Calendar: my own day | `user_plugins:` in `agent.yaml`; connected in the chat |
| None, or local | Blender | not a plugin: the sandbox (`sandbox:`, `sandbox_options:`) plus a skill |

```yaml
agent_plugins:  # the company connects it once, every conversation reads that account
  - sentry
user_plugins:   # each person connects their own, the first time the agent needs it
  - google_calendar
  - name: linear    # a mapping instead of the id says how it is reached
    readonly: true  # the catalog's readonly_url and readonly_scopes
    scopes: [read]  # replaces the scopes asked for at consent
  - name: calcom
    toolsets: [bookings, availability]  # ?toolsets=bookings,availability on the URL
    tools: [get_*]  # offer only these tools, names or path.Match patterns
```

## Where it lives

Slack under `user_plugins` as the example:

- **Set-up page:** Volt's Tools tab (`sections/agent-tools.tsx`) lists the plugins; each
  opens `agents/agents/<config>/tools/apps/<plugin_id>` (`routes/.../tools/apps/$pluginId.tsx`,
  drawn by `sections/agent-app-setup.tsx`): the catalog's `setup_steps`, the redirect URI and
  the client id and secret form, which calls `PUT .../plugins/{plugin_id}/client`.
- **Agent level:** `agent_configs.agent_plugins` and `.user_plugins` (JSONB lists of
  `PluginEntry`) say which plugins the agent has and who logs in. `agent_plugin_clients`, one
  row per `(customer_id, config_id, plugin_id)`, holds the agent's OAuth client: `client_id`
  and `secret_sealed` under the KEK (`kek_version`). Without a row the deployment's
  `<ID>_MCP_CLIENT_ID`/`_SECRET` env is used. The app's own login (`agent_plugins`) is an
  `agent_plugin_connections` row with `user_id = ''`.
- **User level:** `agent_plugin_connections` with `user_id` set, unique on
  `(config_id, plugin_id, user_id)`: `status` (`pending`, `connected`, `failed`), the tokens,
  and while pending the `oauth_state`, `code_verifier`, `client_id` and `token_endpoint` the
  callback finishes with. The button itself lives in the Chat message, not the database.

The API's `PluginEntry` is a `oneOf` of a string and a `PluginWithOptions`, answering as a
bare id when it has no options; the store keeps every entry as a `store.PluginEntry` object.
An entry goes through `session.ConfiguredPlugin`, which every path that opens a server or
starts a login goes through: `attachPlugins`, the user plugin runner, plugin events and
`authorizePlugin`. `session.EntryFor` picks the entry: an app login takes the one in
`agent_plugins`, then `user_plugins`; an app login for a plugin not named yet is the
catalog's, and its callback adds a bare id to `agent_plugins`. `pluginEntriesComplaint`
answers 400 for an id the catalog does not have, one named twice in a list, `readonly` on a
plugin with no `readonly_url`, and a scope missing from the entry's `scopes_supported` (copy
it from the server's `/.well-known/oauth-protected-resource`; left out, any scope goes).
`tools`, there and on `mcp_servers`, becomes `Connection.Tools`: `Open` drops every tool
`Offered` does not match, so it is neither listed nor owned, and `Runtime.Call` refuses it.
The login does not remember the options it was made under, so changing them needs a fresh
login; a user plugin whose server then refuses the old token asks again on its own.

The catalog's `auth` field is read only by `session.Logins`; every catalog plugin is `oauth`. A server
outside the catalog does not go in it: the config names it under `mcp_servers`
(`{name, url, tools, scopes, user}`, https, the name not a catalog id and without `__`).
`attachPlugins` opens those with the catalog logins as `<name>__<tool>`, and
`serverInstructions` adds what each said at initialize to the agent's instructions (capped
at 4000 bytes). The example's TableJourney server is this path, with no login.

The server decides whether it logs in: `Auth.NeedsLogin` says yes for protected-resource
metadata naming an authorization server at either well-known path, or a 401 with a
`WWW-Authenticate` to a tokenless POST. Saving asks each server without `needs_login`
(beside branding, same 5 seconds) and stores the answer as `store.MCPServer.NeedsLogin`,
read-only `needs_login` in the API; nil when unreachable or a 5xx, and then every session
start asks again without storing it. `scopes` or `user` on a server that needs none is a
400. Who logs in mirrors `agent_plugins` and `user_plugins`. Needs none: no token. Needs one
without `user` (`AppLogin()`): the app's login, made with the same authorize, list and
disconnect endpoints as a catalog plugin, keyed by the server's name in
`agent_plugin_connections` (`appPlugin` in `api/plugins.go`); the callback does not add it
to `agent_plugins`. `user: true`: the user plugin runner offers `<name>__list_tools` and
`<name>__call_tool`, and `scopes` left out asks for the `scopes_supported` in the server's
protected-resource metadata. `session.ServerPlugin` turns the server into a `plugins.Plugin`
with `ByURL` set, which is everything the login code needs: `StartAuthorize` never takes a
deployment's `<ID>_MCP_CLIENT_ID` for it (a config may name a server anything), so the server
must offer DCR. Discovery tries the two well-known paths and then the `resource_metadata` in
the `WWW-Authenticate` of an unauthenticated POST. An app-login server with no login yet
gets one stand-in tool, `<name>__list_tools`, that fails with "connect <name> on the
dashboard". The login row's `instance_url` holds the
server's URL, and a login at another URL is never used, so editing the URL never sends a
token to the new host. Its authorization card has no `text` or `thumb_url` and the title
`Connect <name>`; `ValidAuthorization` refuses anything else for a non-catalog id, and
`RequestedAuthorization` takes one only from a server in the session's `Logins`. Saving a config runs
`Auth.CheckLogin` on each server that needs a login: metadata
that answers without OAuth endpoints and a `registration_endpoint` is a 400, a server that
cannot be reached (`*url.Error`) is saved and fails at login. Saving a config asks each
of its MCP servers without branding to describe itself (`plugins.Describe`, an `initialize`
and nothing after), through `egress.NewClient` within 5 seconds, and stores the
`serverInfo` as `branding` (title, or name; description, version, the first https icon,
website). A server that does not answer keeps what it said before at the same URL.

Every `plugins` call given a nil client (`Open`, `Describe`, events, `Auth.HTTP`) goes out
through `egress.NewClient`, because a config names its MCP servers, a login its shop, and
a server's metadata its auth servers: none may reach a private address or the metadata
server. Pass a client only in tests. Router suites pass `PluginHTTP: s.mcpTransport()`,
which reaches only `pluginMCP` (or `pluginHTTP`, for a suite with loopback stand-ins). There are no headers or
secrets on one yet. Something that runs a binary (Blender, a CLI) belongs in
the sandbox with a skill telling the subagent how to drive it, as the example does.

## Once per app

- `POST /v1/agents/configs/{id}/plugins/{plugin_id}/authorize` returns the provider's
  authorize URL; the callback stores the token and adds the plugin to the config.
- `GET /v1/agents/configs/{id}/plugins` lists the config's logins, and every plugin the config
  names without one as `not_connected`. That is the dashboard's reminder to finish setup.
- At session start `attachPlugins` opens every connected server (refreshing a token within a
  minute of expiry), lists its tools and offers them as `<plugin_id>__<tool>`. A server that
  will not start is logged and skipped, so a broken login never refuses the call.

## Once per end user

- The model gets two tools per plugin whose names need no login:
  `<id>__list_tools` and `<id>__call_tool` (`tool`, `arguments`). A session's tools are fixed
  when it opens, before anybody has logged in, so the user's real tools sit behind these.
- Without a login, the tool answers with `AuthorizationResult`: a message for the model and a
  `plugin_authorization` attachment (`plugin_id`, `title`, `authorize_url`) that Chat renders
  as a button. `RequestedAuthorization` only trusts exactly that JSON, from that plugin's own
  tool, with an https URL; never model prose. The plugin must be in `session.Logins`, which
  the manager hands the conversation with `AcceptLogins`: the config's `user_plugins` and
  the `mcp_servers` with `user: true`. Nothing the app logs into, nor a server with no
  login, can put a card in the conversation, whatever its tool is called. See "The authorization attachment" below.
- The callback stores the login under `(customer, config, user, plugin)` and shows a
  "connected, go back to the conversation" page. `Manager.LoginFinished` marks the card
  `connected` and has the session carry on (`Session.FollowUp`): a reply with no user
  message, the model told the plugin is connected. That reply's call opens the MCP session.
- A server refusing the token (`ErrUnauthorized`) drops the login and asks again.
- Only a verified end user is offered user plugins: `api_key` mode with a token naming the
  user, or `proxy` mode. A `noauth` router and an anonymous caller get none, because a login
  made under an unchecked name would be anybody's who used it.

## The authorization attachment

The tool result (`AuthorizationResult` in `plugins/user.go`) the model gets back:

```json
{
  "status": "authorization_required",
  "message": "The user has not connected Linear. They have been shown a button to connect it. Tell them to press it, then ask again once they have.",
  "attachment": {
    "type": "plugin_authorization",
    "plugin_id": "linear",
    "title": "Connect Linear",
    "authorize_url": "https://mcp.linear.app/authorize?client_id=...&state=..."
  }
}
```

The conversation copies `attachment` onto the agent's Chat message, flat, one per plugin
(`conversation/authorizations.go`):

```json
{
  "type": "plugin_authorization",
  "plugin_id": "linear",
  "title": "Connect Linear",
  "authorize_url": "https://mcp.linear.app/authorize?client_id=...&state=...",
  "text": "Search issues and projects, file new issues, and comment on them.",
  "title_link": "https://mcp.linear.app/authorize?client_id=...&state=...",
  "thumb_url": "http://localhost:8080/v1/agents/plugins/linear/logo"
}
```

`title`, `text`, `title_link` and `thumb_url` are Stream's standard attachment fields, so a
Chat client with no `plugin_authorization` renderer still shows a card with a link and a
logo; one that knows the type draws a button from `plugin_id` and `authorize_url`.

| Field | From | Note |
| --- | --- | --- |
| `text` | catalog `description` | what the model reads already |
| `thumb_url` | `Auth.LogoURL` | our own `GET /v1/agents/plugins/{id}/logo`, never hot-linked from the vendor |
| `title_link` | `authorize_url` | the fallback link for clients without a renderer |

Two constraints shape this, and both are easy to trip over:

- **Every field is derived, never taken.** `ValidAuthorization` checks `text` against the
  catalog description, `title_link` against `authorize_url`, and `thumb_url` against
  `LogoPath(plugin_id)`, so an MCP server cannot put an arbitrary image or a tracking pixel
  into somebody's conversation. Adding a field means adding it to `Authorization`,
  `AuthorizationResult`, `ValidAuthorization`, `authorizationAttachments` and
  `authorizationsFromAttachments` together; `RequestedAuthorization` refuses unknown ones.
- **Chat keeps only some fields as an attachment's own.** The set is in
  `conversation_test.go` as `attachmentFields`: `type`, `title`, `title_link`, `text`,
  `fallback`, `image_url`, `thumb_url`, `asset_url`, `og_scrape_url`. Anything else becomes
  custom data. `author_name` is not in it, which is why there is no `author_name` here even
  though Stream's clients render one: it did not survive the round trip, and the title
  already names the plugin.

Still unbuilt, and expected to change as examples need more: `scopes` so the card can say
what the login will be able to do, `status` so the button can turn into a checkmark once the
callback stores the token, and `expires_at` after which the URL needs refreshing.

The Python SDK carries the logo through as `RemoteEvent.image_url`
(`plugins/stream/accelerated.py`), so a non-Chat front end can draw the same button.

## Events: a plugin starting a conversation

A plugin whose MCP server offers [MCP Events](https://developers.openai.com/plugins/build/mcp-events)
(a draft, protocol `2026-07-28`, opened 2026-10-03) can start a conversation instead of
waiting to be asked. The config declares what to watch and what to do:

```yaml
agent_plugins: [sentry]
plugin_events:
  - plugin: sentry          # named under agent_plugins or user_plugins, or the config is refused
    event: issue.created    # as the server's events/list names it
    arguments: {project: web}
    instructions: Say what broke and who should look at it.
```

- [`internal/pluginevents`](../../../acceleration/internal/pluginevents) keeps one
  subscription per declared event and login: the app's for `agent_plugins`, each end user's for
  `user_plugins`. It runs when a config is written, when a login connects or disconnects,
  and once a minute, which refreshes a subscription 10 minutes before its `refreshBefore`
  and retries a refused one after 15.
- The router is the client of the protocol, ChatGPT's role. `plugins/events.go` sends
  `server/discover` (refusing a server without `capabilities.events`), `events/subscribe`
  and `events/unsubscribe`, and signs and verifies Standard Webhooks.
- Each subscription has its own token and `whsec_` secret. The server delivers to
  `public_url` + `/v1/agents/plugins/events/<token>`, so `public_url` must be https and
  public for a real server: a local router needs a tunnel. The row is written before
  `events/subscribe`, because the server checks the callback with a signed challenge before
  it answers.
- The callback answers a verification with its challenge, 401 for a bad signature, 410 for
  a subscription that no longer exists (the server stops delivering), 413 over 256 KiB, 200
  for an `eventId` already taken and 202 for a new one. A new one opens a text session from
  the config, with the declaration's `instructions` added to the config's own and the
  event's `data` as the first message, as JSON the model is told is data, not orders. The
  session is closed once the agent has been quiet for two seconds, or after 15 minutes.
- Only webhook delivery: no polling, streaming, replay cursors, `gap` or `terminated`.
  A disconnected login cannot unsubscribe, so its rows are dropped and the next
  delivery's 410 stops the server.
- Neither Sentry nor Google Calendar offers events yet. `PluginEventsSuite` in
  `internal/api/plugin_events_test.go` stands in a server that does, end to end.

## Adding a plugin to the catalog

1. **Find the server and how it logs in.** Open the vendor's MCP page and note the date. Fetch
   `/.well-known/oauth-protected-resource<path>` (then the origin) and the authorization
   server's `/.well-known/oauth-authorization-server`, falling back to
   `/.well-known/openid-configuration` — GitHub and Salesforce publish only the latter, which
   is why `discoverServer` tries both. If it advertises a `registration_endpoint`, the router
   registers itself (DCR) and nothing needs configuring. If not, the deployment needs a client
   of its own. Of the fourteen entries today, five do DCR (Calendly, Cal.com, Sentry, Linear and
   Shopify) and nine need a client id. A server can advertise registration and still refuse
   us: Gong registers only the redirect URIs it has approved, so it is `client_required`,
   and the router never tries DCR for an entry that is.
2. **Add an entry to `plugins.yaml`**, with the source and date in a comment above it, as the
   Sentry and Google Calendar entries have:

   ```yaml
   # https://vendor.example/docs/mcp, opened <date>. OAuth with dynamic client registration.
   - id: linear              # what agent.yaml names; lowercase, no "__"
     name: Linear
     category: Engineering
     description: Read and file issues.   # the model reads this for user plugins
     url: https://mcp.linear.app/mcp
     auth: oauth
     logo: linear.svg                     # a file in logos/, required
     scopes: [read]                       # optional: asked for at consent
     authorize_params:                    # optional: extra query on the authorize URL
       prompt: consent
   ```

   - **No DCR:** the router reads `<ID>_MCP_CLIENT_ID` and `<ID>_MCP_CLIENT_SECRET` (the id
     upper-cased), sending the secret as `client_secret_post`. Say so in the comment, and the
     redirect URI to register: the router's `public_url` + `/v1/agents/plugins/callback`.
     Set `client_required: true`, `setup_url`, and `setup_steps` (each a `title` and a
     `description`) for the dashboard's Set up page, which shows them beside the redirect
     URI and the client id and secret form. Every entry with `client_required` must have
     them; `catalog_test.go` checks it, and that the steps name the scopes the login asks for.
   - **A refresh token:** some providers only return one when asked. Google needs
     `access_type: offline` and `prompt: consent`; without them the login lasts an hour.
   - **Scopes:** ask for the least the tools need. Google Calendar, Drive and Docs ask
     read-only. Don't guess them: `scopes_supported` in the protected-resource document is
     what the server will accept, and asking for one it does not know is a failed consent.
   - **No single global URL** (Shopify): `url: https://{instance}/...`,
     `instance_required: true` and an `instance_hint`. The app supplies the host when it
     connects, so this only works under `agent_plugins:`; user logins pass no instance.
   - **A read-only server:** `readonly_url` and `readonly_scopes`, for a vendor that runs
     one at its own URL, as Linear does at `/mcp/readonly` (its own resource, accepting only
     `read`). An agent picks it on its entry; the catalog default stays `url`.
   - **Toolsets:** `toolsets`, the names a vendor lets the server be limited to with a
     `toolsets` query parameter, as Cal.com does. An agent picks some on its entry;
     a name not listed is a 400. `resource` at the authorize URL leaves the query off, so
     the login is for the server and survives a change of toolsets. Check that a vendor's
     `scope` does anything before relying on it: Cal.com's MCP authorize ignores it.
   - **One vendor can be several entries.** Google runs Drive and Docs as separate servers
     with separate scope sets, so they are `google_drive` and `google_docs`, not one `google`.
3. **Add a logo** under `plugins/logos/<id>.svg`. They are our own plain 48x48 marks, not the
   vendors' artwork, so a deployment that has licensed the real thing replaces a file and
   changes nothing else. `loadCatalog` reads every one at startup, so a misnamed file is a
   router that will not start rather than a card nobody can see the plugin on.
4. **Update `catalog_test.go`.** `TestTheFourteenPluginsAreListed` pins the ids in order; add a
   test for anything the entry relies on (scopes, authorize params, the endpoint).
5. **No API or client changes.** `plugin_id` is a plain string checked against the catalog,
   and the dashboard lists `GET /v1/agents/plugins`, so the spec does not change.
6. **Try it for real.** Name it in an example's `agent.yaml`, run the router in `proxy` mode
   and connect it: from the dashboard for `agent_plugins:`, from the chat for `user_plugins:`. Unit
   tests use `httptest` servers (`oauth_test.go`, `mcp_test.go`); they cannot tell you the
   vendor's metadata is where you think it is. Short of a real login, a throwaway test in the
   package that calls `discoverResource` and `discoverServer` over every catalog entry against
   the live vendors will catch a wrong URL or a moved document in one run.

## Gotchas

- `public_url` must be reachable by the browser doing the login, and is what the provider
  sees as the redirect URI. A client registered against one URL will not work on another.
- A company login redirects to `DashboardURL/agents/<config>`. An end user's login has no
  editor to return to and shows a plain page instead.
- Plugin tool names are `<id>__<tool>`. `__` is the separator, so it cannot be in an id.
- Running a local agent against Volt: the session belongs to the customer id it was opened
  with, so set `STREAM_ACCELERATION_CUSTOMER_ID` to the Volt app id, not `examples`. Logins
  are per config and per customer too; connecting under `examples` connects nothing in Volt.
- An agent with `sandbox: daytona` refuses to start without `DAYTONA_API_KEY` on the router,
  even for a conversation that would never use the sandbox.
- **A plugin and a channel are different things, and Slack is both.** A plugin is an account
  the agent reads; a channel is somewhere the conversation happens. `slack` in the catalog
  lets an agent search a workspace, and the Slack channel in
  `examples/text_agents/mcp_plugins/channels.py` is a person talking to the agent in Slack.
  Neither implies the other.
- **Microsoft Teams has no hosted MCP server**, as of October 2026. A Teams app hosts its own
  at its own URL, so there is nothing to put a `url:` to and no `teams` entry in the catalog.
  Teams is a channel here, not a plugin.

## Connectors, the successor

[`internal/connectors`](../../../acceleration/internal/connectors) is the manifest-based
rewrite: auth schemes, sources and hooks as adapters, provider manifests in
`connectors/providers/*.yaml` seeded into `connector_definitions` with revisions, and the
`/v1/agents/connectors` endpoints. Sessions do not use it yet; they read `plugins.yaml`. Add a
plugin to `plugins.yaml` today, and read `connectors/core/AGENTS.md` and
`connectors/providers/AGENTS.md` before touching connectors: the core may not name a
provider, and a manifest change needs a new revision or the router will not start.
