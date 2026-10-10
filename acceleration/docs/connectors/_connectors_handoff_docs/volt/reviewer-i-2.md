VERDICT: GO (round 2, head 3ad8232c4b)
- Items 1-3 and the Nit fixed: toolsReads gated, connections query enabled: SHOW_PLUGINS, beforeLoad redirect, simulations-editor copy and main.tsx comment reworded.
- Mutations, all killed: drop toolsReads gate (plugins-hidden.test.ts "reads no plugin when the Tools tab loads"); connections enabled=true (config-used.test.tsx "sends no plugin request while plugins are hidden"); disable redirect (plugins-hidden.test.ts "sends the app setup route back to the Tools tab").
- Full unit suite with .env.local: 237 files, 2769 tests pass. Lint exit 0.
- Browser :3014: /tools/apps/slack redirects to /tools/; Tools tab and session Config tab sent no /plugins request (performance entries empty), no Connected apps text.
- UNVERIFIED: UI save keeping plugins on the API (unit only). Not re-run: knip, build, tsc (src delta is small; not part of the delta checks asked).
