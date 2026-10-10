# Review: volt-dashboard PR #994, delta round 2, 2026-10-09
Head 394f2f551e, previous head 5354652b39, base origin/connectors/agents-ui c3e9244b7e. Fresh worktree $S/volt/rv-a2, removed afterwards.

VERDICT: GO

## R1.1 type errors: FIXED
- `npx tsc --noEmit -p tsconfig.app.json` gives 1121 errors on head and 1121 on base.
- Every file the PR touches has the same error count on head and base. src/gen and routeTree are left out of that check, because they are generated.
- With line numbers stripped, the head and base error sets are identical (`diff`: no output).
- The delta adds `?? []` / `?.` guards, `variant="form"` on both EmptyStates, drops `size` from the Show button, accepts `| null` in argumentShapes, and adds `rotated: false` to the test fixture.

## R1.2 tests: FIXED
Each mutation was re-applied to head. Tests run: connector-consent, connections-page and agent-user.

| Mutation | Result | Failing test |
|---|---|---|
| M5b (always `?force=true`) | killed (2 failing) | "deletes a connection no agent binds without forcing it" |
| M8 (drop the X-Stream-User-Id override) | killed | agent-user "is still the actor when a request is for another end user" |
| M12 (credentials without owner) | killed | "validates, replaces and deletes a user's connection as its owner" |
| M13 (delete without owner) | killed | same test |
| M14 (validate without owner) | killed | same test |

## Gates (Node 24.21.0)
- `bun lint`: 0. `bun run knip:check`: 0. `bun run build`: 0.
- `bun run vitest run --project unit`: 2615/2615 pass.
- `.env.local` was copied in. It is gitignored, and `git status` was clean.

## New in the delta (none block the merge)
- [Nit] tests/unit/agents/connections-page.test.tsx: the owner test uses `volt-42`, which is also the signed-in user. The header cannot tell an owner from the fallback. The test still asserts the userId option, so M12-M14 are killed. A distinct id such as `customer-7` would be clearer.
- [Nit] connections-page.tsx:273: the Show button is now default size next to an `InputGroup size="sm"`. The visual mismatch is not checked in a browser.

## Unverified
- A browser render, as in round 1.
