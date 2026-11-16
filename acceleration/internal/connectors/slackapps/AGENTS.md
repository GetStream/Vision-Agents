# internal/connectors/slackapps

The client the router creates, updates and deletes a customer's own Slack app with (a `managed` provider app, T54, AI-872), and the app manifest it sends. Its one caller is the provider app endpoint, `PUT` and `DELETE /v1/agents/connectors/{id}/provider-app` (`internal/api/connector_provider_apps.go`).

## Why here

A Slack app is Slack's alone, so this is provider code, which lives under `internal/connectors/` beside the schemes and verifiers and never in `core` (`core/AGENTS.md`, «Core imports no adapter», «Core names no provider»). `providers/` holds manifests and no Go (`providers/AGENTS.md`). It imports `core` for the connector manifest it reads scopes and events from, and nothing of `store` or `api`: the caller seals, stores and locks.

## Terms

| Term | What it is | In code | Source |
| --- | --- | --- | --- |
| app configuration token | A token one workspace user generates in Slack's app settings, which the app manifest methods take. «Unique to a user and a workspace, but not an app»; it expires 12 hours after it is generated | `ConfigToken`, `Client.Rotate` | [configuring apps with app manifests](https://docs.slack.dev/app-manifests/configuring-apps-with-app-manifests#config-tokens) |
| refresh token | What `tooling.tokens.rotate` trades for a new configuration token and a new refresh token | `ConfigToken.RefreshToken` | [tooling.tokens.rotate](https://docs.slack.dev/reference/methods/tooling.tokens.rotate) |
| app manifest | The JSON an app is created and updated from | `Manifest`, `ManifestFor`, `Template` | [app manifest reference](https://docs.slack.dev/reference/app-manifest) |
| credentials | What `apps.manifest.create` answers: app id, client id, client secret, signing secret | `Credentials` | [apps.manifest.create](https://docs.slack.dev/reference/methods/apps.manifest.create) |

## Rules

- **Outbound calls go through egress.** `Config.HTTP` is `egress.NewClient(connectorHTTPTimeout, nil)` in `cmd/router`; a test passes `fakeprovider.Server.Client()` with `BaseURL` at the fake's `PathSlackAPI`.
- **Secrets never print.** `Credentials` and `ConfigToken` redact their secrets in `String`, `GoString` and `LogValue`. Check: `go test -run TestSlackAppsSuite/TestCredentialsAndConfigTokensNeverPrintTheirSecrets ./internal/connectors/slackapps`.
- **A spent refresh token is never sent again.** Slack's pages do not say whether the old one keeps working, so the caller saves what `Rotate` returns before anything else, under the provider app lock (`store.WithConnectorProviderAppLock`).
- **The manifest is built from the connector, not written by hand.** Scopes are the connector's `scopes.list`: user scopes for a connector that authorizes at `https://slack.com/oauth/v2_user/authorize` (the Slack MCP server's flow), bot scopes and a bot user otherwise. Events are the connector's `channel.subscriptions`, sent as `bot_events` (or `user_events` for a user-token connector): `providers/slack_bot.yaml` lists `message.channels`, `message.im` and `tokens_revoked`. They are listed rather than derived, because a message rule's `event.type` (`message`) is not a subscription name, and Go names no event.
- **The app is the `slack_bot` connector's.** Its request URL is `api.providerAppEventsPath` + `slack_bot/{app id}`, the route `receiveProviderAppEvent` serves, which verifies only a channel whose secret is `provider_app`. The user-token `slack` connector stays on the operator's app: the Slack MCP server wants clients «backed by a registered Slack app with a fixed app ID» ([Slack MCP server](https://docs.slack.dev/ai/slack-mcp-server/)), and its channel verifies with the operator's secret.
- **Every limit has its page beside it**: name ≤ 35 characters, description ≤ 140, `allowed_ip_address_ranges` ≤ 10, `request_url` https.
- **The events URL names the app id**, which only `apps.manifest.create` returns, so the caller creates the app without event subscriptions, records it, then sets them with `apps.manifest.update`. Slack sends `url_verification` to a request URL when it is configured ([url_verification](https://docs.slack.dev/reference/events/url_verification)), signed with the app's own signing secret, so the record must exist before the update.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/slackapps` (no Postgres), against the fake Slack in `internal/connectors/fakeprovider`. The endpoint's suite, with Postgres: `go test -tags integration -run TestConnectorProviderAppsSuite ./internal/api`.
