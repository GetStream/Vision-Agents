# Reviewer I round 1 (head 07a2c3dd8e vs base 437fb4ce92)
VERDICT: NO-GO
Should fix:
1. src/components/dashboard/agents/lib/agent-reads.ts:100-101 toolsReads still prefetches GET /v1/agents/plugins and /configs/{id}/plugins; used by tools/index.tsx:18 and tools/apps/$pluginId.tsx:18. Browser :3014 Tools tab: reqid 2867 /v1/agents/plugins [200], 2868 /configs/<id>/plugins. Mutation (drop both from toolsReads): 820 pass -> survives.
2. widgets/conversation/config-used.tsx:293 `connections` useQuery still runs while the row is hidden (GET /configs/<id>/plugins, reqid 5786 on session Config tab). Mutation (replace query with stub): 820 pass -> survives. Fix: enabled: SHOW_PLUGINS or delete query inside the gate.
3. Route tools/apps/$pluginId.tsx still renders AgentAppSetup (plugin login UI) by URL; link removed only by the section gate. Gate the route (redirect to tools/) or accept as known.
Nit: remaining user text "connected apps": simulations-editor.tsx:219 ("tools and connected apps"), main.tsx:17 comment. Copy "Connections" in dialogs/playground is right meaning (logins not copied), sentence case ok; but playground note appears only when draft.appsLeftOut (plugins), so the plugin-specific note now says "Connections" which is also true of connector bindings? unverified.
Checks: lint 0, knip 0, build 0, unit agents 68 files/820 pass, tsc -p tsconfig.app.json 2753 lines identical to base.
Data safety: agentRequestFromDraft spreads ...kept; test at agent-draft.test.ts:48 uses fixture with string + object entries (readonly/user); mutation `plugins: undefined` fails 2 tests. OK.
Mutations: SHOW_PLUGINS=true killed (2 tests); drop plugins killed (2); toolsReads drop survived; config-used query stub survived. 4 checked, 2 survived.
Browser :3014: Tools tab shows Abilities/Skills/Connectors only, no Connected apps; session Config tab shows Knowledge base now, no Connected apps now row. Save-in-browser not done (would mutate a real config).
UNVERIFIED: save via UI keeps plugins on API (unit only).
