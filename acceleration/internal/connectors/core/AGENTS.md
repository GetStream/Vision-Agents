# internal/connectors/core

The contracts every connector adapter implements: `Scheme`, `Source`, `Runtime`, `Backend`, `Resolver`, `Verifier`, `Hook`, the `Registry`, and the value types they pass. No implementation lives here. `doc.go` states the rule the rest of this file expands on: adapters import `core`, `core` imports no adapter.

## Rules

- **Core imports no adapter.** Schemes, sources, backends, signals and providers live in their own packages under `internal/connectors/` and import `core`. `TestCoreImportsNoAdapter` fails on any import under `internal/connectors/` outside `core`.
- **Core names no provider.** No string literal in a non-test file may be a provider name from `deniedNames` in `guard_test.go`, or a connector id seeded from `internal/connectors/providers/*.yaml`. Provider specifics go in a manifest or an adapter. `TestCoreNamesNoProvider` enforces it. When you add a name to `deniedNames`, add where it comes from to the comment above the list.
- **Keep the imports small.** `core` imports neither `store` nor `harness`, so `store` can import `core`; tools are `llm.Tool`. Check with `go list -deps ./internal/connectors/core | grep -E 'internal/(store|harness|llmrouter)'`, which must print nothing.
- **Secrets never print.** `Credential` keeps its secret unexported and redacts it in `String` and `GoString`. `Material` redacts `Payload` in `String`, `GoString` and `LogValue`. A new type that carries a secret does the same, with a test in `scheme_test.go` covering `%v`, `%+v`, `%#v`, `%s` and both slog handlers.
- **`Scheme.Wrap` sits on top of egress.** The `base` a scheme wraps is the transport from `egress.NewClient(timeout, scheme.Wrap)`, so the scheme sees the final request and the egress address check runs under it, at dial. Build connector clients no other way.
- **Every hardcoded value says where it comes from**, in a comment next to it, as `deniedNames` does.

## Tests

From `acceleration/`: `go test ./internal/connectors/...`. Testify suites, no mocks (`.claude/skills/go-testing/SKILL.md`). The guard tests parse this package's source with `go/parser`, so a provider name fails before anything uses it.
