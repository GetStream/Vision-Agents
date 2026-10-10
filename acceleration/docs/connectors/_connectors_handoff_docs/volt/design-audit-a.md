# Design-standards audit, volt phase A (744c52f775, PR #994)
Read via worktree at 744c52f775 (removed). Rules: AGENTS.md, .claude/rules/code-style.md, docs/vision-agents-design-review.md.

## Findings
1. [Should fix] src/constants/sidebars.tsx:867-879 (+ routes under agents/resources/connections/, test-ids AgentsResourcesSelectors)
   New one-entry group "Resources". AGENTS.md:94 "The current set - Access, Agent, Call, Channel, Configuration, Explorers, Feed, Library, Logs, Queues, Settings, Telephony - is the reference" (no Resources). AGENTS.md:102 "Product Sidebars: One Shape": <Domain> group = "Library, Telephony, Queues". Existing Library group (sidebars.tsx:~836) holds Knowledge and Voices, the other app-level resources. Fix: put Connections in Library -> /agents/library/connections/ (routes, test-id prefix AgentsLibrarySelectors, tests).
2. [Should fix] src/components/dashboard/agents/widgets/connection-detail-page.tsx:266,293,418,469,597
   6 components in one 694-line file. .claude/rules/code-style.md "One component per file using function declarations". Fix: split ConnectionOverview, ConnectionTools, ConnectionInvocations, ConnectionAudit (+ ValidationResult) into own files in widgets/. Caveat: 14 existing widgets also break it (agent-draft-pickers, playground-page, ...), so a reviewer may accept it as precedent.
3. [Nit] connections-page.tsx:243-285 hand-rolled filter row (Select, form + Input + "Show" button, Select) above the Card. Sibling explorer sessions-table.tsx:~385 puts filters in the Volt `TableFilters` inside `DataTable.Header` (design-review.md "Volt explorer tables"). The aria-label-only Selects match usage-page.tsx:90 / simulations-runs-section.tsx:82, so only the user-ID form is off-pattern. Fix: TableFilters with owner/connector filters, or keep the Selects and make "Show" apply on Enter/blur.
4. [Nit] connection-delete-dialog.tsx:8 imports `listNames` from lib/voices (another feature's module). Fix: move it to lib/format.ts. Unverified as a written rule; pattern only.
5. [Nit] connection-delete-dialog.tsx:35-55 no `confirmation` typed name; voices-delete-dialog.tsx:44 passes `confirmation={voice.name}` for an irreversible delete. Deleting a connection drops credentials irrecoverably. Fix: confirmation={name}.

## COMPLIANT
- No import from components/custom/, no Remixicon/lucide: grep clean across the 6 PR component files (AGENTS.md "Icons: design system Icon in dashboard/"). EmptyState icons "link", "clock-user" are design-system names.
- Tier folders: widgets/, dialogs/, lib/ under dashboard/agents; shared widgets reused: TableSection, DataTablePagination, RowActionsMenu, ActionsMenu, TableErrorState, PrivilegeTooltipGuard, EntityDeleteDialog. ViewerTable is unused by any agents list (sessions-table, audit-table hand-roll DataTable), so connections matches the pattern.
- Sidebar: group "Resources" one word (AGENTS.md:~76 rule met), label "Connections" no double wording, heading "Connections" = nav word.
- Sentence case in all labels, headers, titles, toasts (OAuth, API, ID acronyms).
- Test ids BEM: agents-resources__connections-table, __connections-user-input, __connection-reconnect; in src/test-ids.ts.
- Volt primitives: Pane, DataTable, Card, DescriptionList, FormDialog + Form.PasswordField (labels above fields), Alert, Status, Tag, Breadcrumbs, EmptyState variants as in sessions-table.
- Empty/error states via EmptyState + TableErrorState; pagination via shared DataTablePagination; write actions guarded by PrivilegeTooltipGuard like voices pages.
- No useMemo/useCallback, function declarations, props interfaces, kebab-case file names, route loaders prefetch + PageLoading like siblings.
Not run: bun lint, knip (read-only audit).
