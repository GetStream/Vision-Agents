Wave 3b: connector dispatch hardening (step-up, provider rate limits, binding policy), episode close and the tenancy hook.
Repo: /Users/kanat/Projects/stream/Vision-Agents (GetStream/Vision-Agents, Go router in acceleration/)   Base: accelerate at 005d8312

You are one agent in a wave of PRs. An orchestrator routes your report; it reads only your final report, never your transcript. Your role and inputs follow after this brief.

## Hard rules

- No behaviour change when nothing is configured (no connectors, connections, bindings, provider apps, destinations, contact rows, episode_cards, new opt-in flag). Staging runs with connectors OFF. Prove it with a control test that matches base.
- Never run or enable: `router plugins migrate`, any data move, ROUTER_CONNECTORS_ENABLED on any shared env, anything against staging or prod.
- Migrations are expand-only (add, never drop, rename or tighten), in your assigned slot only (slot+100 for a second file). If origin/accelerate's newest migration is at or after your slot, stop and report.
- Never merge, post review comments to GitHub, or edit Linear. Authors may push their branch and open a draft PR.
- Work only in your worktree, your test database and your scratchpad folder.
- Never commit secrets, `.env` files, tokens or `acceleration/internal/config/testing.yaml`.
- The repo is public: no infra names (GCP projects, clusters, secret prefixes, private repos) in code, commits or PR bodies.

## Technical rules

- Read first: root AGENTS.md, `.claude/skills/go-testing/SKILL.md`, `.claude/skills/parallel-agents/SKILL.md`, and the full rules in /private/tmp/claude-504/-Users-kanat-Projects-stream-Vision-Agents/a0abf0a3-74bd-4641-9e95-9b4aed318e30/scratchpad/wave3-common.md (first production caller, Huma + APIError + doc tags, hardcoded values need a source, PR body format).
- Test DB: the embedded `acceleration/internal/config/testing.yaml` WINS over env. Point its DSN at your `<db>_test` (worktree edit or `-overlay`, restore after), export ROUTER_POSTGRES_DSN on `<db>` and ROUTER_REDIS_ADDR=localhost:56379, create `<db>` first. Run integration with `-tags integration -p 1`. Without the env vars internal/api suites silently skip: report skipped counts.
- Generated files are regenerated, never hand-merged: `cd acceleration && go run ./cmd/openapi`; `cd sdks/go && go generate .`; `cd sdks/js && npm install && npm run types && npm test`. Check with `.agents/skills/pr-review/scripts/generated.sh . origin/accelerate`. Add an "other SDKs" note at the end of `.claude/skills/sdk/SKILL.md` when the API changes.
- Must pass: `cd acceleration && gofmt -l . && go build ./... && go vet ./... && go vet -tags integration ./...`.
- Before every `git rebase --continue` or merge commit: `git diff --check` and grep for conflict markers.

## Known base failures

These fail on the base sha; do not fix or report them.
- internal/api TestSIPTrunksSuite — 2 tests (from #752)
- internal/agent TestTurnRecordingSuite — expects agent id "agent-1", setup sets "agent-test-<nanos>" (ticket AI-938)
- internal/agent TestAgentSuite — timing, flaky
- internal/api TestConnectionEventsSuite/TestADeliverySignedWithAnotherSubscriptionsSecretIsRefused — flaky
- internal/imagerouter TestImageRouterIntegrationSuite — fails if the env DB does not exist yet
- flaky or order-dependent: TestDataMoveSuite, TestAppConfigSuite (fresh DB), TestDisplaySuite/TestAFinishedLoginMarksTheReplyThatAskedForIt, TestSessionUpdateSuite/TestAUserRenamesASessionThatEnded, TestSessionVerbsSuite/…OnItsOwn, SetupSuite duplicate-key race on a fresh DB, TestKnowledgeUrlsSuite, TestLLMSuite/TestStreamGeneratesAResponseIDWhenTheCallerHasNone, the -race in session.Manager.Create, Python test_modalities.py

## Evidence rules

- Cite `file:line`, a command or a commit for every claim. Mark anything not checked `unverified`.
- Every new rule gets a mutation check: break it, see a test fail, restore it. A test that survives its mutation is vacuous.
- Build and test on the PR merged onto the current base, not only its own head.
- Run touched suites only; the full integration run happens once, before the final review.

## Findings format

`[Blocker|Should fix|Nit|Question] file:line — problem — fix`

Blocker and Should fix block the merge. Nit and Question become tickets; never fix them in the same PR.

## Context limit

If your context nears ~300k tokens, write a handoff (done, left, open questions) to your scratchpad folder and stop with `HANDOFF: <path>` as your report. A fresh agent continues.

## Report

- Use your role's report format exactly; the orchestrator parses it.
- Stay within your role's line budget. Put logs, test output and probes in `<scratchpad>/<pr>/<role>-<round>.md` and give the path.

---
