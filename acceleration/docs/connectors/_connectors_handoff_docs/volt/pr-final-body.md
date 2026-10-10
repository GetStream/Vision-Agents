## What

This PR adds the agents connectors UI on top of `ai-team/agent-dashboard`. It was built as eight phase PRs into `connectors/agents-ui`. Every phase PR was reviewed by an independent reviewer with mutation checks, merged only on GO, and checked in a real browser.

```
Library
└── Connectors                       ← one nav entry, two tabs
    ├── Connectors   catalog · OAuth client (secrets write-only, redirect URI) · New connector (custom) · delete (409 → force)
    └── Connections  list · filters (Signed-in user / End user ID / Connector) · New connection · validate · replace token · reconnect · delete
Agent › Tools
└── Connectors                       ← "Connectors this agent can use"
    Add connector → pick connector → Connect to choose its tools → checkboxes → fixed or each user's own
    call settings under Advanced, voice agents only · broken bindings marked
Playground
└── connector_authorization card → Connect → consent popup (postMessage handoff) → session carries on
```

| Phase | PR | Scope |
|---|---|---|
| A | #994 | Connections page, consent handoff utility, agent user header |
| A2 | #995 | Design-audit fixes: split widgets, TableFilters, typed-name delete, Signed-in user filter |
| B | #996 | Library › Connectors (catalog, OAuth client, custom connectors); Connections moved under it |
| C | #997 | Bindings editor on the Tools tab, playground consent card, spec refresh to the router's config renames |
| E | #998 | New connection (OAuth, bearer, api key); multi-tool names; nits from the first E2E |
| F | #999 | Delete custom connectors (409 then force); redirect URI with copy |
| G | #1000 | Failed validation shown; broken bindings marked |
| H | #1001 | Connect from the Add dialog; voice-only call settings; naming (New connector / Add connector); searches; connector links; scope tags |
| I | #1003 | Hide «Connected apps» (plugins) behind `SHOW_PLUGINS = false`: no plugin requests, plugin route redirects, `plugins` kept on save |

«Connected apps» (plugins) is hidden behind `SHOW_PLUGINS = false` (`lib/show-plugins.ts`). Its code stays until the plugins→connectors migration removes it. Saving a config keeps `plugins` unchanged.

## Router

The UI targets the Vision-Agents router at `accelerate` 599298c6. `api/agents/openapi.yaml` is a copy of the router's spec, and `bun run gen:agents` leaves no diff.

## Verified

Full local E2E ran on https://local.getstream.io:3000 against the local router, with every connection created through the UI:
- GitHub PAT app connection, validated and used from Playground;
- Linear consent per user;
- a Slack user post;
- a slack_bot reply in thread;
- custom connector create, edit and delete (409 → force);
- the Connections tab filters, validate, replace token and reconnect.

The router logged 0 `level=ERROR`. Evidence for each run is in the phase PRs.

## Design standards

Each phase PR body has its own checklist:
- `src/components/dashboard/agents/` tier only, with no `custom/` imports;
- design-system primitives (`Tag`, `EntityDeleteDialog`, `TableFilters`, `PaneSearchInput`, …);
- one component per file;
- sentence case;
- BEM test ids in `src/test-ids.ts`;
- nav words matching headings (AGENTS.md).

## Known follow-ups (router side)

- AI-1052: store the last validation result. «Last check» is cache-only until then.
- AI-1053: connector `channel` flag.
- AI-1049: a second config bound to a channel silences the agent.
- AI-1051: bindings left after a forced connector delete.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
