# Fixer A round 1
Mutations (each applied by sed, vitest run agent-user + connections-page, restored):
- M8 drop actingFor override (src/api/agents.ts): killed by agent-user.test.ts "is still the actor when a request is for another end user"
- M12 replace userId -> undefined (connection-credentials-dialog.tsx:43), M13 delete (connection-delete-dialog.tsx:33), M14 validate (connection-detail-page.tsx:115): killed by connections-page.test.tsx "validates, replaces and deletes a user's connection as its owner"
- M5b always '?force=true': killed by "deletes a connection no agent binds without forcing it" (and the owner test)
Checks: bun lint ok, knip exit 0, build ok, unit 231 files/2615 tests pass, tsc app: only pre-existing usage-summary-utils.test.ts error remains in touched/adjacent files.
