# volt PR #994 (phase A) findings
## Round 1 (reviewer a805cd1c915077e06, head 5354652b39) — NO-GO
R1.1 [Should fix] 12 new type errors in the PR's files that `bun lint` misses (root tsconfig checks nothing). `npx tsc -p tsconfig.app.json` shows them (list in volt/rv-a-newtsc.txt): connection-delete-dialog.tsx:29, connection-detail-page.tsx:394,395,487 (`used_by` is `ConnectionUse[] | null`); connection-detail-page.tsx:368-369 (`granted_scopes` nullable); :429 (`tools.tools` nullable); :515 (`row.arguments` can be null, `argumentShapes` takes only undefined); :579,677 (`<EmptyState>` needs `variant`); connections-page.tsx:273 (`<Button size="sm" variant="secondary">` invalid; size only for icon variants); connections-page.test.tsx:252 (credential fixture lacks `rotated`). Fix: `?? []` guards, `variant="form"`, drop `size`, add `rotated: false`, re-run `tsc -p tsconfig.app.json`.
R1.2 [Should fix] Tests mock `agentRequest`, so new header/owner rules are unchecked. Surviving mutations: M8 (removing X-Stream-User-Id override in src/api/agents.ts passes) — add an agentRequest test that stubs fetch and asserts user-id header overridden while actor-id stays the signed-in user; M12/M13/M14 (dropping owner from replace connection-credentials-dialog.tsx:43, delete connection-delete-dialog.tsx:33, validate connection-detail-page.tsx:115 passes) — one test on a user-owned connection asserting all three send the owner; M5b (always sending ?force=true passes) — assert an unbound delete sends no force.
## Nits / Questions (tickets or later phases, not this fix)
- [Nit] connector-consent.ts:163 landing runs on any path; same-origin crafted link fakes "connected" toast. Optional per-attempt nonce.
- [Nit] connector-consent.ts:131 wrong toast (blocked pop-ups) when DASHBOARD_BASE_URL origin differs.
- [Nit] use-connector-consent.ts closed popup not detected; "expired" toast after ~10 min.
- [Nit] connections-page.tsx:72 user ID input not re-synced on owner switch to End user.
- [Nit] connection-detail-page.tsx:94 validation alert stays after replace/reconnect.
- [Nit] connections.ts:83 connectionQueryOptions duplicates agentQueryOptions.
- [Question] origin/accelerate moved to 9f629685 (agent config renames); phase C must re-copy the spec.
- [Question] ngrok warning page vs 15 s ready timeout — check in phase D.
