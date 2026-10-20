# G1 brief: code in the sandbox reaches a connector's API without seeing its credential

Read this before any PR of waves 13 and 14. Context doc (Kanat, 2026-10-19/20):
https://claude.ai/code/artifact/4087645f-e972-4b5b-9f05-e3f884e89689

## Problem (G1)

`run_code` (`internal/sandbox/sandbox.go:17`) runs model-written Python in a Daytona sandbox
(`internal/sandbox/daytona/daytona.go`, created lazily by `session.Manager.box`,
`internal/session/manager.go:1363-1369`). Connector credentials reach only the router: MCP
tools via `mcp.Toolset` and `core.Resolver` (`internal/connectors/core/resolver.go:17`).
Code in the sandbox gets nothing: no env var, no placeholder, no proxy
(`git grep -n connector origin/accelerate -- acceleration/internal/sandbox` → nothing).
So a subagent that must `git push` or call `api.github.com` fails with 401/403.

## Target design (Tommaso, DM 2026-10-19 11:40 EDT; industry pattern)

The sandbox holds a placeholder, never the token. An egress proxy outside the sandbox swaps
the placeholder for the real credential, only in request headers, only for the connector's
hosts. Sources: Claude Code cloud (https://code.claude.com/docs/en/security «never enter the
session VM»), Vercel Sandbox credentials brokering (https://vercel.com/docs/sandbox/concepts/firewall),
v0 → Snowflake (https://vercel.com/blog/how-v0-authenticates-to-snowflake-without-exposing-the-users-oauth-token:
inject only into the auth field; never blind string replacement).

## Mechanism: Daytona Secrets (P1)

Daytona already implements that proxy. SDK we pin, `github.com/daytona/clients/sdk-go v0.211.2`:
- `types.SandboxBaseParams.Secrets map[envVar]secretName` (`pkg/types/types.go:109-114`): the
  env var «holding the secret's opaque placeholder, which is resolved to the real value only
  when the sandbox connects to one of the secret's allowed hosts».
- `Client.Secret` Create/Update/Delete/List (`pkg/daytona/secret.go`), `CreateSecretParams{Name,
  Value, Hosts}` (`types.go:202-213`); name `^[a-zA-Z_][a-zA-Z0-9_-]*$`, unique per org (409).
- Docs https://www.daytona.io/docs/en/secrets.md (opened 2026-10-20): substitution only in
  HTTPS request **headers** (not bodies, query, plain HTTP); the header must carry the
  placeholder verbatim (Base64 Basic auth does not match); placeholder `dtn_secret_<random>`;
  response echoes rewritten back to the placeholder; Update «takes effect in all sandboxes that
  use the secret within 15 seconds»; a sandbox created with no secrets must restart before newly
  attached ones work.
- Our account: `GET $DAYTONA_API_URL/secret/paginated?limit=1` with the local `.env` key →
  200 `{"items":[],"total":0}` (2026-10-20 21:56Z). Staging's key/org: unverified (wave 13 entry).

## Rules for every PR

- Evidence-based: every claim in code comments and PR bodies cites file:line, a vendor page, or
  a command + result. Hardcoded values get their source in a comment.
- Opt-in only: no existing session, config or connector behaves differently unless a binding
  says `sandbox: true`. That keeps every PR additive (no switch, no behaviour change).
- No migration: bindings are JSON in `agent_configs.connectors` (`store.ConnectorBinding`,
  `internal/store/models.go:425`). An older router drops an unknown JSON key, so a rollback
  reads the binding without `sandbox` = off (fail closed).
- The token never lands in a log, an error, a span, a tool result or the model's context.
  Only the Daytona secret name and the placeholder may be logged.
- SDK change: Go first; append the «other SDKs need updating» note to `.claude/skills/sdk/SKILL.md`.
- Tests: testify suites (go-testing skill); no mocks of our own code; Daytona calls behind the
  existing `HTTPClient` option so a test serves Daytona's API from an `httptest.Server`.
- Never commit `testing.yaml`; each PR its own DB `model_router_test_w13_<t>_test`.
