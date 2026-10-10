# PR #1001 phase H, round 2 delta review (2026-10-09)
VERDICT: NO-GO (1 Should fix, a test gap in the finding-1 fix)
Head 46d07dbbb0 vs 1655836c9b: connectors-catalog.tsx (+8/-2), two test files.
- Finding 1: fixed in code. Nav link drops q -> box clears (test kills R20). Deep link kept (R18 killed). Debounce into URL OK; the guard (q !== wanted) cannot drop the last keystroke (probe below passes on real code at sleeps 300/302/305/310 ms).
- Finding 2: R2 R4 R5 R7 R8 R9 R14 R18 all killed (each by the new named test).
- Open (Should fix): reset-to-q mutation (replace "if ((search.q ?? '') !== wanted) setQuery(...)" with unconditional setQuery) SURVIVES the PR tests (exit 0). With it, a keystroke typed right after the debounce fires is wiped when the URL catches up (probe fails at 302/305/310 ms, passes at 300). Fix: add a test like /private/tmp/claude-504/-Users-kanat-Projects-stream-Vision-Agents/a0abf0a3-74bd-4641-9e95-9b4aed318e30/scratchpad/volt/rh2-probe-connectors-page.test.tsx (PROBE2: type a, wait 300 ms, type ab without waiting, expect URL q=ab and box value ab); make it deterministic (fake timers or a controlled navigate), not a sleep.
- Checks on 46d07dbbb0 as is: lint 0, knip 0, build 0, unit 236 files / 2769 tests pass (logs rh2-*.log). R19 pattern gone (guard rewritten), n/a.
UNVERIFIED: browser not re-run; tsc not re-compared (diff touches one component file, no type changes).
Worktree removed; :3000 untouched.
