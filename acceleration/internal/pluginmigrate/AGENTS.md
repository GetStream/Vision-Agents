# internal/pluginmigrate

`router plugins migrate` (`cmd/router/plugins.go`): the plugin system's rows moved onto connectors, T61 in `acceleration/docs/connectors/subtasks.md` on `connectors/planning` («Plugins move onto connectors» in `architecture.md` there). A person runs it, by hand, once per deployment, before T23 drops the plugin tables. Nothing in the router calls it: not `serve`, not a migration, not a job.

## Flow

```
Run(opts, apply)                       opts.Customer limits it to one app, opts.Plugins to these plugin ids (--plugin); empty is every one
  read     agent_plugin_clients, agent_plugin_connections (live), agent_configs naming plugins
  clients  per (app, plugin): the newest agent_plugin_clients row -> connector_oauth_clients,
           registration customer, secret opened (session.PluginClients) and sealed again
           (api.SealConnectorOAuthClientSecret); the others reported, or "the same client"
  logins   per connected row -> connector_connections MovedConnectionID(row id), connector = the
           plugin's id, owner app when user_id is empty, else user and user_id, scheme oauth2_code
             checks  catalog plugin, connected, config live, plugin endpoint == manifest mcp
                     (one trailing slash apart is the same), oauth2code.MoveGrant: token endpoint
                     discovered == the row's, client == the app's preregistered one, or the
                     public client the plugin registered when it had none set in advance
             write   CreateConnectorConnectionWithID, then pgsealed Update: sealed, connected,
                     then a grant_created audit row, reason plugin_migrate, tokens by
                     fingerprint, and the "connector credential event" line (api.Server.auditGrant)
  bindings plugins entry             -> fixed binding to the app's moved login
           plugins entry, user: true -> session binding
           alias = connector id = plugin id; grants = every tool a validate lists through the
           moved login that the entry's tools allow (plugins.Offered): fixed with the digests
           listed, session by name (each person's connection pins its own on first use, #826)
           the plugin entry is kept beside the binding; the session drops it for the binding
           (session.Spec.withoutBoundPlugins), and the API takes a binding to the plugin's own
           connector under the plugin's name (api.pluginAliasComplaint)
  events   plugin_events reported, not moved (T60)
  -> Report: one Row per row read, Planned (dry run), Written, Exists or Skipped with why
```

## Rules

- **The plugin tables are read, never written.** `EveryPluginConnection`, `EveryPluginClient` and `AgentConfigsNamingPlugins` only select; `agent_configs` gets `connectors` and nothing else of it changes. Check: `go test -tags integration -run TestPluginMigrateSuite/TestThePluginTablesAreByteIdentical ./internal/api`.
- **Dry run is the default.** Without `apply` nothing is written; the command takes `--apply`, and refuses it unless `connectors.enabled` is set, since a binding wins over its plugin entry in a session (`session.Spec.boundProvider`). Check: `go test -tags integration -run TestPluginMigrateSuite/TestADryRun ./internal/api` and `go test -run TestPluginsCommandSuite ./cmd/router`.
- **The command neither migrates nor seeds.** It opens the database with `store.Open`, not `openStore`, and refuses one with a migration this binary carries not applied or one migrated by a newer build (`store.MigrationDrift`, read only), so a newer binary run against an older database writes nothing; a built-in not seeded yet is a skipped row. Check: `go test -tags integration -run 'TestPluginsMigrateSuite/(TestARunNeither|TestADatabaseBehind)' ./cmd/router`.
- **A grant with a refresh token moves by default only to a connector whose manifest says `refresh.rotating: false`**; a rotating one, or one whose manifest does not say (`rotating` nil: sentry, hubspot, shopify, calcom, gong, salesforce), moves only with `IncludeRotating` (`--include-rotating`). The plugin row keeps its copy of the refresh token, and with rotation whichever side renews first retires the other's. Check: `go test -tags integration -run 'TestPluginMigrateSuite/(TestARotating|TestAGrantWhoseRotation)' ./internal/api`.
- **A binding is written to `connectors` alone**, under the config's row lock (`store.AddConnectorBinding`, `appconfig.Store.AddConnectorBinding`), so an edit of any other column made meanwhile is kept. Check: `go test -tags integration -run 'TestStoreSuite/(TestAnEditCommittedWhileABindingWaitsIsKept|TestABindingIsAddedOnceAndOnlyToALiveConnection)' ./internal/store`.
- **The plugin entry stays, and the config stays editable.** Removing it would stop the plugin's events (`pluginevents.reaches`), fail a config naming them (`pluginEventsComplaint`), and lose the entry's options a rollback needs; deleting the binding is the rollback. Check: `go test -tags integration -run 'TestPluginMigrateSuite/TestAMovedConfigStays|TestConfigsSuite/TestABindingToThePluginsOwnConnector' ./internal/api`.
- **A moved grant is audited.** One `grant_created` row, reason `plugin_migrate`, with the moved tokens' fingerprints, and one log line; a dry run or a second run adds none. Check: `go test -tags integration -run TestPluginMigrateSuite/TestAMovedGrantIsAudited ./internal/api`.
- **A session binding grants by name.** A provider may describe a tool per person (Slack), so one person's digests would leave the others' tools unavailable. Check: `go test -tags integration -run TestPluginMigrateSuite/TestASessionBindingGrantsByName ./internal/api`.
- **`--plugin` moves only the named plugins' rows**: their clients, logins, entries and events; an id the catalog lacks is an error, and a flag set with no id left (`--plugin ""`) is an error, never every plugin. Check: `go test -tags integration -run 'TestPluginMigrateSuite/(TestOnlyTheNamed|TestAPluginTheCatalog)' ./internal/api` and `go test -tags integration -run 'Test(PluginsMigrateSuite/TestPluginNames|PluginsCommandSuite/TestAPluginFlag)' ./cmd/router`.
- **A second run changes nothing.** The connection id is derived from the plugin row's id, a client record or a binding already there is left as it is, and a connection made but never given its grant is finished, not made again. Check: `go test -tags integration -run 'TestPluginMigrateSuite/(TestASecondRun|TestARunStopped)' ./internal/api`.
- **A grant moves only where it still renews.** Same server, same token endpoint, same client; anything else is a Skipped row and the person logs in again. Check: `go test -run 'TestOAuth2CodeSuite/(TestAMoved|TestAGrant|TestAClient)' ./internal/connectors/schemes/oauth2code`.
- **Tokens are sealed.** They reach the row only through `pgsealed`, bound to the customer, the connection and its revision. No row of the report names a token or a secret. Check: `go test -tags integration -run TestPluginMigrateSuite/TestAnAppLoginIsAppOwned ./internal/api`.

## Tests

From `acceleration/`: `go test ./internal/pluginmigrate` (catalog against the built-in manifests, no database), and with Postgres and Redis `go test -tags integration -run TestPluginMigrateSuite ./internal/api` (the moves, then the session path of #775) and `go test -tags integration -run TestPluginsMigrateSuite ./cmd/router` (the command's wiring).
