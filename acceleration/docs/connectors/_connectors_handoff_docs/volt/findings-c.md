# volt PR #997 (phase C) findings
## Round 1 (head 9f137f5ca1) — NO-GO; full text volt/reviewer-c-1.md, mutations volt/pr-c/rv-mut.py
R1.1 [Should fix] src/components/dashboard/agents/lib/connector-bindings.ts:154-158 bindingFromFields keeps the old binding's schema_digest on a session binding: edit a fixed binding, switch to "Each user's own" → still pins the app connection's digests (probe output {"type":"session"}, tools:[{"name":"post","schema_digest":"aaaa…"}]). pluginmigrate/AGENTS.md: "A session binding grants by name". Fix: never write a digest on a session binding (or keep only if the existing binding was already session); test the fixed→session switch.
R1.2 [Should fix] Rules with no failing test: R1 connector-bindings.ts:104 alias may not be an MCP server's name; R2 :106 fixed binding needs a connection; R3 :115 timeout_ms ≤ 30000; R5 agent-tabs.ts:82 connectors → Tools tab; R7 conversation-chat.tsx:81 no button without handoff_token; R9 aliasFor unique suffix :57. One test each.
## Nits (not in the fix)
- conversation-chat.tsx:95 connect button md vs plugin links sm.
- agent-tools.tsx:358 "Sessions use the linear connector" tag on a catalog app with no plugin entry.
- sections/agent-connectors.tsx:89-107 Edit/Remove disabled with no reason.
- editor does not cap tools at 128 (spec maxItems) → router 400.
## Questions → router tickets
- Test copy / duplicate of an agent subscribes connection events twice (mcpevents.go:210-227, store/connection_events.go:76).
- channelbridge drops inbound when len(configs) != 1 (bridge.go:425-433, gate.go:64): a copy/duplicate with a fixed binding to a channel app's connection silences the live agent.
