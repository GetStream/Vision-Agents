---
name: add-connector
description: How to add a connector - a custom MCP server for one app (no code), or a new built-in in the router's catalog (a manifest YAML with sources, a consent test and a PR). Read before you add a provider, change a built-in manifest, or bump its revision.
---

# Add a connector

First decide which kind you need. Connecting an existing connector to an agent is in the
`connectors` skill.

| You want | Kind | Code change |
|---|---|---|
| One app uses its own MCP server | custom connector | none: the API or the dashboard |
| Every app can pick the provider from the catalog | built-in | a manifest in `acceleration/internal/connectors/providers/` |

## A custom connector

Dashboard: **Library › Connectors › New connector**. API: `POST /v1/agents/connectors`
(`CustomConnectorRequest`: `id`, `name`, `endpoint`, `schemes`, optional `scopes`,
`client`, `category`, `description`).

- The id starts with `custom_`, for example `custom_crm`. Built-ins never use that prefix.
- `endpoint` is the MCP server's https URL.
- `schemes` is how a connection signs in: `oauth2_code`, `bearer`, `api_key`.
- A custom connector with no credential at all is not possible yet (AI-1059). For a public
  server, use `api_key` with a dummy header and value, and say so in its description.
- Then add a connection to it and bind it to an agent, as for any connector.
- Delete answers 409 while an agent config binds it. The message names the bindings. Force
  deletes it anyway and leaves those bindings broken.

## A built-in

A built-in is one YAML file, `acceleration/internal/connectors/providers/<id>.yaml`. The router
embeds it and stores it in `connector_definitions` at every start. Read
`acceleration/internal/connectors/providers/AGENTS.md` (rules) and the «Manifest» section of
`acceleration/internal/connectors/core/AGENTS.md` (fields) before you write one.

Start from the closest existing file:

| Your provider | Copy |
|---|---|
| A hosted MCP server with OAuth and dynamic client registration | `linear.yaml` |
| OAuth with an app the operator or customer registers, plus a token option | `github.yaml` |
| Receives messages (a channel) | `slack_bot.yaml`, `whatsapp.yaml` |
| A bearer token only | `linq.yaml` |

### Steps

1. **Write `<id>.yaml`.** The id is the file name and never starts with `custom_`. Set
   `revision: 1`. Fill `name`, `category`, `description`, `endpoints`, `schemes`, `client`
   (`registration`: `operator`, `customer`, `managed`, `dcr`, `cimd`), `scopes`, `refresh`,
   `rate_limit` and `sources`. Add a `channel` block only if the provider sends messages.
2. **Cite every vendor fact.** Put a comment beside each value with the vendor page and the
   date you opened it. A value with no source gets `# unverified`, and the PR lists it.
3. **Stream's own OAuth client** (only if `client.registration` lists `operator`): set
   `client.env: <ENV>`. The deployment reads `<ENV>_MCP_CLIENT_ID` and
   `<ENV>_MCP_CLIENT_SECRET`. Add both to `acceleration/README.md`, «Configuration».
4. **A consent test.** Every OAuth built-in has one in `providers/consent_test.go`. It runs
   `oauth2_code` against `internal/connectors/fakeprovider`, with the manifest's own endpoints
   pointed at the fake. New fake behaviour belongs in `fakeprovider`, not in the test.
5. **Run the checks** from `acceleration/`:

   ```bash
   go test ./internal/store ./internal/connectors/...     # parses every manifest, runs consents
   go test ./internal/connectors/core -run TestCoreNamesNoProvider
   ```

   The store tests need Postgres (see `providers/AGENTS.md`, «Tests»). A test DB of your own:
   the `go-testing` and `parallel-agents` skills.
6. **Try it live.** Start the router locally (`dashboard` skill), connect an account, bind
   one tool and call it in Playground. Then follow the `connectors` skill.
7. **Open the PR** on the `accelerate` branch. Its body lists the sources and every
   `# unverified` value.

### Changing a built-in

- **Any change to what the manifest says needs a new `revision`.** An edit without one stops
  the router from starting. A comment, spacing or key order is not a change.
- **Existing connections keep their pinned revision.** They move to the new one at their next
  consent (OAuth) or token save. So a new field, such as `api_base`, reaches old connections
  only then.
- **To retire a broken revision**, the new revision lists it in `broken_revisions` with a
  reason. Connections on it are asked to sign in again.
- **Never delete a file to remove a connector.** Its revisions stay for the connections that
  pinned them.
- **Never revert by editing.** A revert is a new revision with the old content.

## Before you ask

| Question | Answer |
|---|---|
| Which scheme? | What the provider supports for an agent acting for someone: OAuth with PKCE (`oauth2_code`) when it has it, otherwise a token (`bearer`, `api_key`). Machine-to-machine OAuth is `oauth2_client_credentials` (`salesforce.yaml`) |
| Who registers the OAuth client? | The provider's docs decide. An MCP server with dynamic registration: `dcr`. Otherwise `operator` (Stream's app), `customer` (the app's own), or both |
| Do I need Go code? | No, unless the provider needs a hook. Then talk to the router owners first |
| Where does the redirect URI come from? | `<ROUTER_PUBLIC_URL>/v1/agents/connectors/oauth/callback`. Register it at the provider |
