# Review: volt-dashboard PR #994 (phase A), round 1, 2026-10-09

Head 5354652b39 (origin/connectors/agents-ui-a), base c3e9244b7e (origin/connectors/agents-ui). Worktree $S/volt/rv-a, removed after.

VERDICT: NO-GO (2 Should fix)

## Checks run (Node v24.21.0, bun)
- `cmp` of `git show c000cedc:acceleration/api/openapi.yaml` (Vision-Agents) with api/agents/openapi.yaml: identical.
- `bun run gen:agents`, then `git status --short`: clean. src/gen/agents was regenerated, not edited by hand.
- `bun lint`: exit 0. `bun run knip:check`: exit 0 (exports 316). `bun run build`: exit 0.
- `bun run vitest run --project unit`: 2612/2612 pass. The worktree needs the main checkout's `.env.local` copied in; without it, 157 tests fail with "VITE_AMPERE_API environment variable is not defined". tests/unit/agents: 63 files, 666 tests pass.
- `npx tsc --noEmit -p tsconfig.app.json`: head has 1133 errors, base 1121. The 4 PR files add 13 error lines (12 errors plus 1 continuation line): $S/volt/rv-a-newtsc.txt. In base, src/components/dashboard/agents has 4 errors. The repo's own `bun lint` runs `tsc --noEmit` on the root tsconfig, which is `"files": []` plus references and has no `-b`, so it type-checks nothing.

## Findings
1. [Should fix] The new files have type errors that lint cannot see:
   - connection-delete-dialog.tsx:29 and connection-detail-page.tsx:394,395,487 use `used_by`, which is `ConnectionUse[] | null`.
   - connection-detail-page.tsx:368-369 uses `granted_scopes`, which is `string[] | null`.
   - connection-detail-page.tsx:429 uses `tools.tools`, which is `| null`.
   - connection-detail-page.tsx:515: `row.arguments` is `| null`, but argumentShapes takes `| undefined`.
   - connection-detail-page.tsx:579,677: `<EmptyState>` has no `variant`, which is required (the base uses `variant="form"`).
   - connections-page.tsx:273: `<Button size="sm" variant="secondary">` is invalid, because `size` is only for icon variants.
   - tests/unit/agents/connections-page.test.tsx:252: the credential fixture lacks the required `rotated`.
   - Today's router never sends null for these: connections.go:614-640 builds `make(...,0)` and `append([]string{}, ...)`, and connection_tools.go:431 uses `make` (both at c000cedc). The nullability still comes from the spec.
   - Fix: add `?? []` guards, `variant="form"`, drop `size`, add `rotated: false`. Re-run `tsc -p tsconfig.app.json` and check that no errors remain in these files.
2. [Should fix] Tests miss rules this PR adds. agentRequest is mocked, so the header mapping is never checked. Surviving mutations:
   - M8: removing `if (actingFor) headers['X-Stream-User-Id'] = actingFor` (src/api/agents.ts) still passes. Fix: add a test of agentRequest/agentHeaders that stubs fetch and checks two things: X-Stream-User-Id is the override, and X-Stream-Actor-Id is still the signed-in user.
   - M12, M13, M14: `userId: () => undefined` in the credentials dialog:43, the delete dialog:33 and the detail-page validate:115 still passes. Fix: add a test for a user-owned connection that runs replace, delete and validate, and checks that each sends the owner's id.
   - M5b: always sending `?force=true` still passes. Fix: assert that deleting an unbound connection sends no `force`.
3. [Nit] connector-consent.ts:163 finishConsentLanding runs on any path with `connection_id` and `status`. Any same-origin page or link (e.g. `/?connection_id=X&status=connected`) can broadcast a fake "connected" to a waiting tab. The impact is a wrong toast and a refetch, and the waiting consent stops listening, so its real result is then ignored. Under the HTML "script-closable" rule, a fresh tab opened from such a link also closes itself (unverified in a browser). This is acceptable for phase A, because the state comes from the refetch. Optional hardening: a per-attempt nonce in sessionStorage (window.open copies it into the popup), checked on landing.
4. [Nit] connector-consent.ts:131: when DASHBOARD_BASE_URL's origin is not the current dashboard origin (for example the author's :3011 dev server), the launch page's ready message is dropped. The person then sees "The consent page did not open. Allow pop-ups", which names the wrong cause.
5. [Nit] use-connector-consent.ts: a popup the person closes is never detected. About 10 minutes later the expiry timer toasts "The consent expired". Also, `authorize.data` keeps handoff_token in the mutation cache until gcTime. It is in memory only: not in the DOM, not logged and not in storage.
6. [Nit] connections-page.tsx:72: draftUser is not re-synced when owner switches from app to user. The input stays empty while the list shows volt-<id>.
7. [Nit] connection-detail-page.tsx:94: the `validation` alert stays after a replace or reconnect and can be stale.
8. [Nit] connections.ts:83: connectionQueryOptions duplicates agentQueryOptions with one more key part. A `userId` option on agentQueryOptions would keep one builder.
9. [Question] Spec drift: Vision-Agents origin/accelerate is now 9f629685. It renames config fields (agent_plugins/user_plugins become plugins[].user, progressive_tools becomes tools.progressive, thinking_llm becomes subagent, speed is dropped). No connection endpoint changed, so phase A is fine. Phase C has to re-copy the spec.
10. [Question] ROUTER_PUBLIC_URL is ngrok. If the tunnel shows ngrok's browser-warning interstitial, the 15 s READY_TIMEOUT_MS can expire before the launch page loads. Phase D should check this.

## Security notes (verified OK)
- The ready message is accepted only when `event.source === popup` and `event.origin === launchOrigin(launch_url)`. The handoff goes to that origin only, once (the handedOff flag, and the listener is removed). launch_url must be https, or http on localhost, 127.0.0.1 or [::1].
- The launch page (authorizations.go:453-473) posts ready to `window.opener` with targetOrigin = DashboardOrigin. It accepts the handoff only from window.opener at DashboardOrigin and nulls window.opener before it goes to the provider. Both sides agree.
- The redirect back (authorizations.go:643-655) is DASHBOARD_BASE_URL plus connection_id and status. Its status set matches CONSENT_STATUSES (authorizations.go:81-84).
- The handoff token is never in a URL, never logged and never in storage. A blocked popup gives a toast and makes no request. A failed authorization request closes the popup.
- The dashboard has no COOP header (none in vercel.json, vite.config or index.html), so window.opener survives on the launch page.
- userId: agentHeaders overrides only X-Stream-User-Id, and the actor stays the signed-in user. It is not dev-only: on the hosted router it rides a `server: true` JWT that the browser signs with the app secret (agents-credentials.ts:79-91). Any holder of that token can already act for any user (router actingUser is server-side only), so there is no new trust. The dev proxy passes the header through unchanged (scripts/agents-dev-proxy.ts). A non-dev build without VITE_AGENTS_ROUTER_URL throws in agentEndpoint.
- PUT credentials sends `{expected_revision, values}` per the ConnectionCredentials schema. Its values are write-only, the form unmounts on success, and the response is a Connection with no secret.
- force=true is sent only when used_by is non-empty. That matches the spec: "refused with a 409 unless force is set" for a fixed binding, and used_by lists exactly those (connections.go:56,185,472).
- argumentShapes follows InvocationArgument: `(undeclared)` carries the count in length. credentialLines shows fingerprints only.

## Mutations (each on head; tests run were connector-consent + connections-page)
| # | Mutation | Result |
|---|---|---|
| M1 | drop `event.source !== popup` | killed, 1 failing |
| M2 | drop `event.origin !== origin` | killed, 1 failing |
| M3 | targetOrigin → '*' | killed, 1 failing |
| M4 | drop `data.connection_id !== connectionId` | killed, 1 failing |
| M5 | force never | killed |
| M5b | force always | SURVIVED |
| M6 | expected_revision + 1 | killed, 2 failing |
| M7 | consent authorize without the owner | killed |
| M8 | drop the X-Stream-User-Id override in agentHeaders | SURVIVED |
| M9 | drop the https check | killed, 2 failing |
| M10 | drop view.close() | killed |
| M11 | list without userId | killed, 2 failing |
| M12/13/14 | credentials, delete or validate without the owner | SURVIVED |
| M15 | drop replaceState | killed |

## Unverified
- A signed-in browser render of either page, and EmptyState rendering without `variant`.
- The real popup flow end to end: the BroadcastChannel delivery, the popup self-closing after COOP swaps at providers, and the ngrok interstitial.
