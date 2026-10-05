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
| Once per app | Sentry: the company's issues | `plugins:` in `agent.yaml`; connected on the dashboard |
| Once per end user | Google Calendar: my own day | `user_plugins:` in `agent.yaml`; connected in the chat |
| None, or local | Blender | not a plugin: the sandbox (`sandbox:`, `sandbox_options:`) plus a skill |

```yaml
plugins:        # the company connects it once, every conversation reads that account
  - sentry
user_plugins:   # each person connects their own, the first time the agent needs it
  - google_calendar
plugin_options: # optional: how one is reached, and what its login asks for
  - plugin: linear
    readonly: true  # the catalog's readonly_url and readonly_scopes
    scopes: [read]  # replaces the scopes asked for at consent
  - plugin: calcom
    toolsets: [bookings, availability]  # ?toolsets=bookings,availability on the URL
    tools: [get_*]  # offer only these tools, names or path.Match patterns
```

`plugin_options` is read by `session.ConfiguredPlugin`, which every path that opens a
server or starts a login goes through: `attachPlugins`, the user plugin runner, plugin
events and `authorizePlugin`. `readonly` on a plugin with no `readonly_url` is a 400, and so
is a scope missing from the entry's `scopes_supported` (copy it from the server's
`/.well-known/oauth-protected-resource`; left out, any scope goes). `tools`, there and on
`mcp_servers`, becomes `Connection.Tools`: `Open` drops every tool `Offered` does not match,
so it is neither listed nor owned, and `Runtime.Call` refuses it. An
option may name a plugin the config does not name yet, because an app's login only adds the
plugin to `plugins` once it is made. The login does not remember the options it was made
under, so changing them needs a fresh login; a user plugin whose server then refuses the
old token asks again on its own.

The catalog's `auth` field is not read: every catalog plugin logs in with OAuth. A public
MCP server with no login does not go in the catalog: the config names it under
`mcp_servers` (`{name, url, tools}`, https, the name not a catalog id and without `__`).
`attachPlugins` opens those with the catalog logins, with no token, as `<name>__<tool>`, and
`serverInstructions` adds what each said at initialize to the agent's instructions (capped
at 4000 bytes). The example's TableJourney server is this path. There are no headers or
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
  tool, with an https URL; never model prose. See "The authorization attachment" below.
- The callback stores the login under `(customer, config, user, plugin)` and shows a
  "connected, go back to the conversation" page. The next call opens the MCP session.
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
plugins: [sentry]
plugin_events:
  - plugin: sentry          # named under plugins or user_plugins, or the config is refused
    event: issue.created    # as the server's events/list names it
    arguments: {project: web}
    instructions: Say what broke and who should look at it.
```

- [`internal/pluginevents`](../../../acceleration/internal/pluginevents) keeps one
  subscription per declared event and login: the app's for `plugins`, each end user's for
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
   of its own. Of the twelve entries today, five do DCR (Calendly, Cal.com, Sentry, Linear and
   Shopify) and seven need a client id.
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
   - **A refresh token:** some providers only return one when asked. Google needs
     `access_type: offline` and `prompt: consent`; without them the login lasts an hour.
   - **Scopes:** ask for the least the tools need. Google Calendar, Drive and Docs ask
     read-only. Don't guess them: `scopes_supported` in the protected-resource document is
     what the server will accept, and asking for one it does not know is a failed consent.
   - **No single global URL** (Shopify, Salesforce): `url: https://{instance}/...`,
     `instance_required: true` and an `instance_hint`. The app supplies the host when it
     connects, so this only works under `plugins:`; user logins pass no instance.
   - **A read-only server:** `readonly_url` and `readonly_scopes`, for a vendor that runs
     one at its own URL, as Linear does at `/mcp/readonly` (its own resource, accepting only
     `read`). An agent picks it with `plugin_options`; the catalog default stays `url`.
   - **Toolsets:** `toolsets`, the names a vendor lets the server be limited to with a
     `toolsets` query parameter, as Cal.com does. An agent picks some in `plugin_options`;
     a name not listed is a 400. `resource` at the authorize URL leaves the query off, so
     the login is for the server and survives a change of toolsets. Check that a vendor's
     `scope` does anything before relying on it: Cal.com's MCP authorize ignores it.
   - **One vendor can be several entries.** Google runs Drive and Docs as separate servers
     with separate scope sets, so they are `google_drive` and `google_docs`, not one `google`.
3. **Add a logo** under `plugins/logos/<id>.svg`. They are our own plain 48x48 marks, not the
   vendors' artwork, so a deployment that has licensed the real thing replaces a file and
   changes nothing else. `loadCatalog` reads every one at startup, so a misnamed file is a
   router that will not start rather than a card nobody can see the plugin on.
4. **Update `catalog_test.go`.** `TestTheTwelvePluginsAreListed` pins the ids in order; add a
   test for anything the entry relies on (scopes, authorize params, the endpoint).
5. **No API or client changes.** `plugin_id` is a plain string checked against the catalog,
   and the dashboard lists `GET /v1/agents/plugins`, so the spec does not change.
6. **Try it for real.** Name it in an example's `agent.yaml`, run the router in `proxy` mode
   and connect it: from the dashboard for `plugins:`, from the chat for `user_plugins:`. Unit
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
