Hides the plugins ("Connected apps") section on the agent Tools tab; Connectors stays. Kept in code behind `SHOW_PLUGINS = false` until the plugins to connectors migration, then deleted.

- `agent-tools.tsx`: one constant gates `<ConnectedApps />`; plugin queries no longer fire.
- Saving keeps `plugins` byte-for-byte (`agentRequestFromDraft` spreads the saved config); new test in `agent-draft.test.ts`.
- Copy: save-as-new and duplicate dialogs and the playground note say "Connections", not "Connected apps".
- Tests: plugin-section tests replaced by one asserting the section is absent; the "Sessions use the X connector" assertion dropped from the migrated-app test (that tag lived in the hidden row).

Mutations: `SHOW_PLUGINS = true` fails "shows connectors and hides the connected apps section"; dropping `plugins` from the request fails both agent-draft plugin tests.
Checked in browser on :3011: Tools tab headings are Abilities, Skills, Connectors.

Design standards: AGENTS.md sentence-case copy (dialogs, playground); no new components, so tier, icon and test-id rules are not applicable; knip passes without deleting code.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
