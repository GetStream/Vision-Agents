# volt PR #996 (phase B) findings
## Round 1 (head 0096df3f42) — NO-GO; full text volt/reviewer-b-1.md, mutations volt/rvb-mut.py
R1.1 [Should fix] 12 new rules untested (12 surviving mutations): OAuth client form rules lib/connectors.ts:89-113; Remove only when a client is stored connector-oauth-client-section.tsx:63; Edit only on custom connector-detail-page.tsx:74; custom id pattern + https endpoint lib/connectors.ts:150,161; newest tick wins in Connector filter connections-page.tsx:133; removing End user filter resets owner connections-page.tsx:142; View connections passes connector_id connector-detail-page.tsx:70; edit prefills stored registration lib/connectors.ts:193. Fix: one test per rule asserting the request or the screen.
R1.2 [Should fix] lib/connectors.ts:198-217 + connector-dialog.tsx:53: Save revision drops stored client.auth_method and client.alg (router stores both, connectors.go:296-297). Fix: carry both through on edit + test.
R1.3 [Should fix] Wrong copy for provider-app-only connectors (linq, telnyx, whatsapp): empty state connector-oauth-client-section.tsx:156 says "use Stream's client… or register one at each consent"; dialog connector-oauth-client-dialog.tsx:88 says "Leave empty only for a public client". Fix: build empty-state copy from the connector's registration; say provider app only when no OAuth scheme.
## Nits / Questions (not in the fix)
- [Nit] lib/connectors.ts:85,87 secrets trimmed; router allows spaces.
- [Nit] Connections tab lost "End user = me" one-click — resolved by #995's "Signed-in user" boolean filter when B merges #995.
- [Nit] connectors/index.tsx:22 prefetch ignores search word.
- [Nit] AgentsConnectorsSelectors vs AgentsLibrarySelectors (kept as directed).
- [Question] redirect_uri — router ticket AI-1047 (wave 7).
