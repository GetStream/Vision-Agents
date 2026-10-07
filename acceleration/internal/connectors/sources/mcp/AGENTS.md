# internal/connectors/sources/mcp

The `core.ToolSource` for a connector's MCP server, registered as `mcp` (`Kind`), the name a manifest's `sources[].kind` gives it. Ported from the prototype's `internal/mcp/mcp.go` on `codex/connector-support` at `cf62af0d`, on the official Go SDK (`github.com/modelcontextprotocol/go-sdk`). Callers today: `POST /v1/agents/connections/{id}/validate` (`Discover`, `internal/api/connection_tools.go`). The session's dispatcher (T21) will call `Open`.

## Flow

```
Discover(ctx, binding)                   within startupTimeout (10 s)
  endpoint     manifest sources[kind=mcp].endpoint -> Endpoints[role]
  client       binding.HTTP (core.Transports), each response capped at 4 MiB
  connect      server/discover (2026-07-28), else initialize (2025-11-25)
  tools/list   every page; a repeated cursor is an error
  ToolSpec     name, description, input schema,
               SchemaDigest = sha256(name, description, input schema),
               NeedsScopes from the manifest's sources[].tools[]

Open(ctx, binding, grants)               same connect and list
  alias        binding.Name, never holding "__"
  keep         granted by name AND digest unchanged AND schema compiles
  offer        alias + "__" + name

Toolset.Call(call)
  unknown name             -> error, nothing sent
  arguments vs schema      -> error, nothing sent
  tools/call               binding.Timeout bounds it
  text parts, else structured JSON, else "not text" -> cut at 32 KiB + marker
  isError                  -> *core.ToolError, even with no text
```

## Rules

- **Only through `ResolvedBinding.HTTP`.** The source never builds a client. It sends with a copy of the binding's client whose transport caps the response and then calls the client's own transport, so the credential, the 401 renewal (`core/AGENTS.md`, «Transport») and the egress checks all run. The redirect policy stays egress's. Check: `go test -run 'TestSourceSuite/(TestEveryRequest|TestARefused|TestACredentialNothing|TestAResolverRefusal)' ./internal/connectors/sources/mcp`.
- **A tool is offered only when its grant names it and pins the digest it has now.** A changed name, description or input schema is a new digest, so the tool is hidden until it is granted again. Check: `go test -run 'TestMCPSourceContract/(TestOnlyTheGranted|TestAnUngranted|TestAToolWhoseSchema)' ./internal/connectors/sources/mcp`.
- **Arguments are checked before anything is sent.** Against the tool's own JSON Schema, with every external `$ref` refused, so a schema cannot make the router fetch a URL. Check: `go test -run 'TestMCPSourceContract/TestArguments|TestSourceSuite/TestASchemaReferring' ./internal/connectors/sources/mcp`.
- **Names never collide.** An alias holding `__` is refused, so a name splits at its first `__` into one alias and one tool. Check: `go test -run 'TestSourceSuite/(TestABindingName|TestToolsOfTwoBindings)' ./internal/connectors/sources/mcp`.
- **Bounded.** `startupTimeout` bounds `Discover` and `Open`, even when the SDK would wait longer; one HTTP response is at most 4 MiB; one result at most `core.MaxResultBytes` (32 KiB), cut at a UTF-8 boundary and ending with `core.TruncatedMarker`. Check: `go test -run 'TestSourceSuite/(TestTheStartupTimeout|TestAResponseOverTheCap)|TestMCPSourceContract/TestAResultOverTheCap' ./internal/connectors/sources/mcp`.
- **`isError` is an error.** A result the server marks `isError` is a `*core.ToolError` carrying its text, or a fixed sentence when it has none (an image only). Check: `go test -run 'TestMCPSourceContract/TestAnErrorWithNoText|TestSourceSuite/TestAToolError' ./internal/connectors/sources/mcp`.
- **Every hardcoded value says where it comes from**, beside it: `defaultStartupTimeout`, `maxResponseBytes`, `Separator`, `implementation`.

## Tests

From `acceleration/`: `go test -race ./internal/connectors/sources/mcp`. `TestMCPSourceContract` runs `core/contracttest.SourceContract`; `SourceSuite` the rest. The provider is an MCP server of the official SDK on a loopback TLS server that takes a bearer token, one tool per `tools/list` page, reached through `core.Transports` with the `bearer` scheme and a resolver kept in memory. `TestTheFakeProvidersServerIsDiscovered` runs against `fakeprovider` (T5), whose MCP endpoint is written by hand. No mocks (`.claude/skills/go-testing/SKILL.md`).
