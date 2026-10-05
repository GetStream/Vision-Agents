# internal/connectors/providers

The built-in connector manifests: one `<id>.yaml` per connector, in the schema `core.ParseManifest` reads (`internal/connectors/core/manifest.go`). `providers.go` embeds them; at every start the router's `openStore` calls `store.SeedConnectorDefinitions` (`internal/store/connectors.go`), which writes each one to `connector_definitions` under the built-in customer.

## Rules

- **One YAML per built-in, and the id is the file name.** `slack.yaml` holds `id: slack`. The seeder refuses a file whose id is not its name.
- **No built-in id starts with `custom_`.** That prefix belongs to custom definitions, so a customer's definition can never shadow a built-in. The seeder and a CHECK on the table both refuse it.
- **Any change to what a manifest says gets a new `revision`**, as a migration gets a new number. The seeder stores each file at the revision it names, because builds with different files share a database: old and new pods in a rolling deploy, a rollback, a branch build beside an accelerate build on staging. A build that finds its own revision stored changes nothing, and an older build never makes its revision latest again.
- **An edit without a new revision stops the router from starting**, with the file and revision named, so a connection pinned to a revision never reads two manifests. "Changed" means the parsed manifest changed, not the file: the seeder compares the JSON `core.Manifest` marshals to, so comments, spacing, key order and quoting need no new revision.
- **Reverting a manifest is a new revision** whose content is the old one. Rolling the router back does not make an older manifest latest.
- **Removing a file removes nothing.** Its revisions stay in the table for the connections that pinned them.
- **An invalid manifest stops the router from starting**, with the file and the field named. The unit tests in `internal/store` parse every file here, so `go test ./internal/store` catches one before a deploy does.
- **Every vendor fact carries its source** in a comment beside it: a vendor page (with the date it was opened), a line of the prototype `internal/mcp/connectors.yaml` on `codex/connector-support` at `cf62af0d`, a row of the architecture doc (`acceleration/docs/connectors/architecture.md` on `connectors/planning`), or a fixture in `internal/connectors/core/testdata/manifests/`. A value with no source is marked `# unverified` and listed in the PR.
- **Every id here is a provider name the core may not spell.** `TestCoreNamesNoProvider` (`internal/connectors/core/guard_test.go`) reads the ids from this directory into its deny list.
- **No Go here beyond the embed and its tests** for now. A provider's hook (`hooks:` in a manifest) is the only Go a provider may need, and it is added with the hook registry, not before.
- **Every OAuth built-in has a consent test**, Slack's and Linear's in `internal/connectors/schemes/oauth2code/oauth2code_test.go` and the rest in `consent_test.go` here: Begin, the browser and Complete of `oauth2_code` against `internal/connectors/fakeprovider`, with each endpoint role the manifest writes pointed at the fake and none added, so the test takes the manifest's own discovery path, `client.registration`, scopes and capture rules. A behaviour no fake personality has is asserted as the failure it causes (Salesforce's identity URL), not faked here: new fake behaviour belongs in `fakeprovider`.

## Tests

From `acceleration/`: `go test ./internal/store ./internal/connectors/...` without Postgres (the store's unit tests parse every file here; `TestConsentSuite` here runs each consent), and with it:

```bash
ROUTER_POSTGRES_DSN='postgres://postgres:postgres@localhost:55432/model_router_test?sslmode=disable' \
  go test -tags integration -run 'TestStoreSuite|TestOpenStoreSuite' ./internal/store ./cmd/router
```
