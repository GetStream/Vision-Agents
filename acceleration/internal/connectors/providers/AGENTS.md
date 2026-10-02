# internal/connectors/providers

The built-in connector manifests: one `<id>.yaml` per connector, in the schema `core.ParseManifest` reads (`internal/connectors/core/manifest.go`). `providers.go` embeds them; at every start the router's `openStore` calls `store.SeedConnectorDefinitions` (`internal/store/connectors.go`), which writes each one to `connector_definitions` under the built-in customer.

## Rules

- **One YAML per built-in, and the id is the file name.** `slack.yaml` holds `id: slack`. The seeder refuses a file whose id is not its name.
- **No built-in id starts with `custom_`.** That prefix belongs to custom definitions, so a customer's definition can never shadow a built-in. The seeder and a CHECK on the table both refuse it.
- **Do not bump `revision` here.** The database numbers revisions: the seeder stores a changed manifest as the latest revision plus one and writes that number into the stored manifest. The file says `revision: 1` only because `ParseManifest` requires a revision.
- **"Changed" means the parsed manifest changed, not the file.** The seeder compares the JSON `core.Manifest` marshals to, with the revision left out. Comments, spacing, key order and quoting never add a revision. Any other edit adds one at the next start of every router, and connections stay pinned to the revision they were created from.
- **Removing a file removes nothing.** Its revisions stay in the table for the connections that pinned them.
- **An invalid manifest stops the router from starting**, with the file and the field named. The unit tests in `internal/store` parse every file here, so `go test ./internal/store` catches one before a deploy does.
- **Every vendor fact carries its source** in a comment beside it: a vendor page (with the date it was opened), a line of the prototype `internal/mcp/connectors.yaml` on `codex/connector-support` at `cf62af0d`, a row of the architecture doc (`acceleration/docs/connectors/architecture.md` on `connectors/planning`), or a fixture in `internal/connectors/core/testdata/manifests/`. A value with no source is marked `# unverified` and listed in the PR.
- **Every id here is a provider name the core may not spell.** `TestCoreNamesNoProvider` (`internal/connectors/core/guard_test.go`) reads the ids from this directory into its deny list.
- **No Go here beyond the embed** for now. A provider's hook (`hooks:` in a manifest) is the only Go a provider may need, and it is added with the hook registry, not before.

## Tests

From `acceleration/`: `go test ./internal/store ./internal/connectors/...` without Postgres, and with it:

```bash
ROUTER_POSTGRES_DSN='postgres://postgres:postgres@localhost:55432/model_router_test?sslmode=disable' \
  go test -tags integration -run 'TestStoreSuite|TestOpenStoreSuite' ./internal/store ./cmd/router
```
