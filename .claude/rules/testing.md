---
paths:
  - "**/*_test.go"
---

# Testing

## Go Tests

**Rules for new and modified tests** (not lint-enforced; older tests predate them):
- No `TestMain`, no global variables — each test gets isolated dependencies
- Package name: `package_test` (external test package)
- Generate identifiers per test — never hardcode them
- Always `require` over `assert`
- Every `t.Skip(...)` cites a concrete trigger that would unblock it (a specific setup change, a planned fixture, a ticket). "Pending fixture" / "out of scope" / "requires X in setup" without a named action is a smell — placeholders rot. If you cannot name what would make the test run, delete it. Before adding a skip, root-cause why the test won't run; the cause is often a 5-minute fix (schema mismatch, missing helper) rather than a missing infrastructure piece.
- Golden / generated-SQL tests compare the entire formatted query via `require.Equal` against a complete expected literal (both run through the same SQL formatter). Don't assert SQL with `require.Contains` / `require.NotContains` / `strings.Fields` / `strings.Count` — substring checks silently miss column reordering, wrong join keys, duplicated clauses, and extra predicates, which is exactly what golden tests must catch.
- Hand-write the expected literal in a clean, readable style (one indent level per nesting, each clause on its own line); never paste raw formatter "river" output. Both sides are re-formatted at runtime, so the literal's whitespace is cosmetic — when the test fails, fix the literal's tokens to match, never reformat it into river style.
