# Reviewer round 2 (delta) — volt PR #995, c5ac2eb27a -> 6a70436d8b
Date 2026-10-09. Worktree $S/volt/rv-a2 at 6a70436d8b (detached, removed after). Base still 744c52f775.
Delta: connections-page.tsx (commit empty -> remove(id), .slice(0,256), onAddFilter 'user' + agentUser() -> update({owner:'user'})), +6 tests.

VERDICT: NO-GO

## Checks (logs $S/volt/rv-a2-r2.*.log, summary rv2a-r2-checks.out)
lint 0; knip 0; unit 231 files / 2621 tests pass; tsc -p tsconfig.app.json 1121 errors, same set as base 744c52f775 (rv-a2-base.tsc.log, line numbers stripped). Tree clean after.

## Mutations (python3 $S/volt/rv2a-mut.py from the worktree; log rv2a-mut-r2.log; M2-M4 retargeted to the new code, M10-M12 added)
KILLED: M1 confirmation, M2 user_id, M3 owner switch, M5 remove keeps owner, M6 connector ignored, M7 newest tick, M8 clear, M9 remove connector, M10 no prefill.
SURVIVED: M4 `else remove(id)` -> `else update({ user_id: undefined })`; M11 -> `else update({})`; M12 drop `.slice(0, 256)`.

## R1.1 (tests) — mostly fixed; M4/M11 survive
Fixer's claim "an empty apply removes the chip" is right for "": design-system core/index.js TableFilters `U(e,t)`: `if (F(null), bN(t)) { s(e); return; } o(e,t)` and `bN` is `e === ""` for strings. But "   " is not empty to bN, reaches commit, trims to "" and hits `else remove(id)`. So the branch is reachable and untested (the "applied empty" test clears to "", which the DS handles). Under M4 a whitespace apply would keep owner=user and show the signed-in user.
Fix: in "returns to the app's own when the end user ID is applied empty", type '   {Enter}' (or add a case).

## R1.2 (signed-in default) — NOT fixed
Adding "End user ID" sets ?owner=user and the chip shows the signed-in id, but the editor opens with an empty draft. RN keeps `useState(initialDraftValue)`, `NN(def)` = null for text, and only the chip click (`h(t.value)`) loads the value. Any close then drops the filter: dismiss -> `ee(id, null)` -> bN(null) -> onRemoveFilter; Apply -> `U(id, null)` -> onRemoveFilter. Our remove() sets owner undefined, so the page goes back to the app's list.
Evidence:
- jsdom probe (scratch test, deleted): add filter -> editor value "" -> Escape -> search {}; Apply -> search {}.
- Browser, my tab on :3014, signed in: add -> URL ?owner=user, chip "End user ID: volt-1115938", textbox "User ID" empty; Escape -> URL back to /connections/ and the Filters button only.
So the coordinator's "editor opens empty" is a finding, not a quirk: the signed-in list stays only while the editor is open.
The new test passes because it never closes the editor.
Fix (suggested): a `kind: 'boolean'` filter, e.g. "Signed-in user". The DS commits a boolean filter on add without an editor (`H`: `o(e, pN(n))`). commit -> update({owner:'user', user_id: undefined}); remove -> app. Keep "End user ID" for typed ids. Test: add it, then assert the request carries header volt-42 and the URL keeps owner=user after Escape.

## Nit (256 cap) — accepted
`.slice(0, 256)` keeps the URL inside connectionsSearchSchema max(256), so it no longer falls back to the signed-in user. The chip shows the truncated id that is queried, so the result is visible. M12 survives; it is a nit-level guard, not blocking. Ticket if wanted: block > 256 with a message instead.

## UNVERIFIED
- Click-outside dismissal not driven separately; same onOpenChange(false) -> ee path as Escape per core/index.js RN.
