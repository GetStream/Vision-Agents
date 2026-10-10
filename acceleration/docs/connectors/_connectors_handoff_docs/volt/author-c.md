# Phase C author log (2026-10-09)
PR: https://github.com/GetStream/volt-dashboard/pull/997 (draft) head connectors/agents-ui-c 9f137f5ca1, base connectors/agents-ui 744c52f775. Worktree $S/volt/wt-c. PR body: $S/volt/pr-c/body.md.

## Commits
- 649f7018ed spec at Vision-Agents origin/accelerate 26051062 + rename adaptations (greeting {text,mode}, thinking_llm->subagent, lcm->decision_model incl. guardrail type and usage modality).
- 9f137f5ca1 bindings editor (lib/connector-bindings.ts, dialogs/connector-binding-dialog.tsx, sections/agent-connectors.tsx, agent-tools.tsx app note, draft.connectors, agent-tabs connectors->tools) + playground card (session.ts zod branch + isToolAttachment, use-conversation connectorLogins, conversation-chat button, use-connector-consent handOff).

## Evidence
- Local router 26051062: /v1/lcm/providers 400 (validation_failed, enum), /v1/decision_model/providers 200.
- bun lint 0; knip:check 0 (exports 317); build 0; unit 234 files / 2633 tests pass ($S/volt/pr-c/unit.log).
- tsc -p tsconfig.app.json: identical to base modulo one pre-existing findLast error's printed union ($S/volt/pr-c/tsc-base.log vs tsc-head.log).
- Mutations ($S/volt/pr-c/mut.py, results mut.log): M1-M24 all KILLED. M2 and M21 first survived; added the bindingFromFields unit test and tests/unit/agents/agent-safety.test.tsx, then killed.

## Browser
- Dev server :3012 started, but every page redirected to /login (no session for local.getstream.io:3012; :3000 tab was on /forgot-password). No signed-in render, no screenshots (the MCP refused writing under $S/volt/shots: not a workspace root). Dev server stopped, certs and .env.local removed.

## Questions
- Events on a binding are kept but not editable here; a draft test copy (POST config) carries bindings incl. events as before (kept spread) - subscribe twice? Router side.
- Session binding tool names are read from the app's or the signed-in user's own connected connection; with neither, names are typed. OK?
