## Kanat's rules (apply to every line you write)

1. **First production caller.** A PR ships with its first production caller: an endpoint, the session or a command. A test is not a caller. At the end, grep your new exported symbols outside `_test.go` and outside their package, and report the caller.
2. **No behaviour change.** Our changes must not change how `accelerate` works today. With no connectors, connections, provider apps, event destinations, contact rows or bindings configured, and no new opt-in flag set, every existing path must behave exactly as before. That includes sessions, calls, plugins, message hooks, conversations, channels, incognito and router startup. Anything that would touch an existing path must be off by default. Prove it with a test where nothing is configured and the behaviour matches base. Staging runs with connectors OFF (`ROUTER_CONNECTORS_ENABLED` unset).
3. **Never switch anything on.** Never enable connectors. Never run `router plugins migrate` or any data move. Never remove the plugin system. The plugin → connector migration and the switch are done later, by hand, with Kanat.
4. **Independent review.** An independent reviewer runs `/pr-review` before merge. Every Blocker and Should fix must be fixed before merge; only Nits may wait. So build it right the first time: tests for every acceptance item, a mutation check for each new rule (break it, see a test fail, restore it), and race-safety for anything two routers can do at once.

## Common technical rules

- **Hardcoded values.** Every hardcoded value has its source written beside it: an RFC, a vendor doc you opened, or the design doc. Anything without a source is marked `unverified` and listed in the PR.
- **Operations.**
  - Huma with `doc:` tags.
  - Errors via the `APIError` helpers in `internal/api/apierror.go` (`invalidRequest`, `notFound`, …, or a shared `errX` value). Never `huma.ErrorXXX`.
  - Created errors use `stack.Wrap`.
  - Server-side only unless the brief says otherwise.
  - List endpoints: read `.claude/skills/pagination/SKILL.md`.
- **After operation changes.**
  - `go run ./cmd/openapi`
  - `cd sdks/go && go generate .`
  - `cd sdks/js && npm install && npm run types && npm test` (do not commit an unrelated `package-lock.json`)
  - The Python client, per `acceleration/README.md` «Regenerate the HTTP layer», only if a Python-facing field changed.
  - Add a note for the other SDKs at the bottom of `.claude/skills/sdk/SKILL.md`.
- **Migration.**
  - Use ONLY the version slot assigned to you in the brief (e.g. `20261007160000_<name>.sql`).
  - If you need two, use slot+100, slot+200, and so on.
  - Do not pick another version. Before pushing, check `origin/accelerate`. If its newest migration is at or after your slot, stop and report instead of renaming.
  - goose Up/Down; check on a scratch `_test` DB: accelerate's migrations, then yours, then Down, then Up.
  - It must apply to existing rows without a table rewrite.
- **New packages** get an `AGENTS.md` and a `CLAUDE.md` containing only `@AGENTS.md`.
- **Tests.** Testify suites (`.claude/skills/go-testing/SKILL.md`, `RouterSuite` for API integration). Never mock. Use the fake provider, local HTTP servers and the Stream Chat test server. Test behaviour.
- **Secrets.** Never print secrets.
- **Known failures on base**, not yours:
  - `internal/agent` `TestAgentSuite` (timing);
  - `TestHarnessSuite/…Images`;
  - `TestDataMoveSuite`;
  - `internal/conversation` `TestDisplaySuite/TestAFinishedLoginMarksTheReplyThatAskedForIt`;
  - `TestSessionVerbsSuite/TestChangingASessionsModelsLeavesEveryOtherSessionOnItsOwn` (order-dependent);
  - `TestSessionUpdateSuite/TestAUserRenamesASessionThatEnded`;
  - `TestSIPTrunksSuite` (from #752);
  - the `-race` race in `session.Manager.Create`;
  - live-provider suites without keys.

## Test database isolation (six agents run integration tests at once)

- **In YOUR worktree only:**
  - Set the DSN database in `acceleration/internal/config/testing.yaml` to `model_router_test_<slug>_test`.
  - Export `ROUTER_POSTGRES_DSN` pointing at db `model_router_test_<slug>` (no `_test`), and `ROUTER_REDIS_ADDR=localhost:56379`. Never print passwords.
- **Skips.** Without the env vars, the `internal/api` suites silently skip. Report skipped counts.
- **Never commit `testing.yaml`.** Restore it before you finish.
- **Other databases.** Do not touch docker or any other database.
- **Fresh DB.** A first run on a fresh DB may fail; re-run once.

## Branch, PR, report

1. `git fetch origin accelerate connectors/planning`, then `git switch -c <branch> origin/accelerate`.
2. Read and follow root `AGENTS.md` (it changed recently: `APIError`, local dev), the `go-testing`, `commit` and `pr` skills, and `.github/pull_request_template.md`.
3. Checks:
   - gofmt;
   - `go vet` with and without `-tags integration`;
   - `go test -race` on the touched packages;
   - full `go test ./...` once;
   - integration runs of every suite you touch, plus the suites named in your brief;
   - the goose check;
   - mutation checks.
4. Commits: conventional, by layer, staged by name. Each message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never use `--no-verify`.
5. Before pushing, `git fetch origin accelerate` and rebase.
6. Push, then open a **draft** PR with `gh pr create --draft --base accelerate`.
   - Title ≤ 50 chars, linking the Linear issue.
   - Body per the pr skill: a chart first, short tables (input → result → test), «First production caller», «No behaviour change without config», «Where the values come from», `unverified`, «Integration tests» (CI does not run `-tags integration`; list your manual runs and skipped counts).
   - PUBLIC repo: no infra names, no credentials.
   - End with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
7. Run `gh pr checks <N> --watch`. Fix the failures this PR caused. The Python `test_modalities.py` job may fail on base; ignore it.
8. Do not edit Linear. Do not merge.

**Report back (concise):**
- PR URL, branch, commits, files, migration name;
- the exact test commands and results, including mutation checks and skipped counts;
- the caller grep;
- the no-behaviour-change proof;
- decisions, each with one concrete example;
- `unverified` values;
- anything open.

## Token rules (Kanat, 2026-10-07; they override the steps above)

- **Tests.**
  - Each fix round, run only the suites the fix touches plus a mutation check of each fix.
  - Run the full integration set once, just before the final review, and attach its results to your report.
- **Nits.**
  - Do NOT fix Nits unless the message explicitly lists them as "do in this push".
  - List the remaining Nits in the PR body under "Left for tickets".
- **SDK notes.** Append your note to `.claude/skills/sdk/SKILL.md` as before. A `merge=union` rule removes the tail conflict once it lands.
- **Report.**
  - Keep the report short: head SHA, what changed, a mutation table, and test totals.
  - No restating of the brief.
