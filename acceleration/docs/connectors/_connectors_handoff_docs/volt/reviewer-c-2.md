VERDICT: GO (PR #997, delta 9f137f5ca1..9d3100100a)
R1.1 fixed: bindingFromFields drops existing digest unless fixed; new test fixed->session saves {name} only; R10 (revert) KILLED.
R1.2 fixed: tests for R1,R2,R3,R5,R7,R9 added; all KILLED (R5 by agent-tabs.test.ts "puts the connectors on the Tools tab").
Merge: origin/connectors/agents-ui d6c5204f9a merged into 9d3100100a locally, clean, no conflict, not pushed.
Checks on merged tree: lint 0, knip:check ok, build ok, unit suite 234 files/2649 tests pass, tsc app 1121 errors = base 1121, none in touched files.
UNVERIFIED: none. Note: sharing main's node_modules lacks user-event; used own bun install --frozen-lockfile.
