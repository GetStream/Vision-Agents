# Reviewer round 3 (delta) — volt PR #995, 6a70436d8b -> 8c74979e8a
Date 2026-10-09. Worktree $S/volt/rv-a2 at 8c74979e8a (removed after). Base still 744c52f775.
Delta: connections-page.tsx adds a `kind: 'boolean'` filter 'me' ("Signed-in user", only when agentUser()); applied 'me' when owner=user && !search.user_id && userId; 'user' chip only with search.user_id; commit 'me' -> update({owner:'user', user_id: undefined}); remove 'me'|'user' -> app; prefill hack gone. Tests: +2, empty-apply test types '   {Enter}'.

VERDICT: NO-GO (one test missing; code behaves correctly)

## Checks (logs $S/volt/rv-a2-r3.*.log, rv2a-r3-checks.out)
lint 0; knip 0; unit 231 files / 2622 tests pass; tsc -p tsconfig.app.json 1121 errors, same set as base 744c52f775. Tree clean.

## Mutations ($S/volt/rv2a-mut.py, retargeted: M4/M11/M5 now use the multi-line user-branch strings; M10 retired; M13-M19 added; run: `python3 ../rv2a-mut.py M1 M2 M3 M4 M5 M6 M7 M8 M9 M11 M12 M13 M14 M15 M16 M17 M18 M19`; log rv2a-mut-r3.log)
KILLED: M1-M9, M11, M13 me sets user_id=agentUser, M14 me no-op, M15 me chip not applied, M16 me offered only when signed out, M17 remove me keeps owner.
SURVIVED:
- M12 drop .slice(0,256): accepted nit from round 2, ticket.
- M18 `update({ owner: 'user', user_id: undefined })` -> `update({ owner: 'user' })`: picking "Signed-in user" while an End user ID is applied keeps the typed user_id, so the typed user stays listed and the signed-in choice silently does nothing.
- M19 applied 'me' guard drops `!search.user_id`: both chips show for a typed user.
Both are new rules in this round's code -> [Should fix].

## Findings
R2.1 (M4/M11): fixed. Whitespace test kills both (my retargeted M4/M11 KILLED).
R1.2 (signed-in default): fixed. DS commits a boolean filter on add with no editor (core/index.js `H`: `o(e, pN(n))`), so nothing to dismiss.
Browser (my tab on :3016, signed in; tab closed, server stopped): from `?owner=user&user_id=ui-test-r3`, Add filter menu = ["Signed-in user", "End user ID (disabled)", "Connector"]; picked Signed-in user -> `?owner=user`, only chip "Signed-in user"; Escape -> still `?owner=user`, chip kept.
[Should fix] tests/unit/agents/connections-page.test.tsx — M18/M19 survive — add (verified to kill both, scratch probe; passes unmutated):
```tsx
it('switches from a typed end user to the signed-in one', async () => {
  const user = userEvent.setup()
  const router = setup(listUrl + '?owner=user&user_id=customer-7')
  await user.click(await screen.findByRole('button', { name: /add filter/i }))
  await user.click(await screen.findByRole('menuitem', { name: 'Signed-in user' }))
  await waitFor(() => expect(router.state.location.search).toEqual({ owner: 'user' }))
  expect(screen.queryByRole('button', { name: /End user ID/ })).not.toBeInTheDocument()
})
```

## UNVERIFIED
- Fixer's a2-mut3.py not run as-is; equivalent mutations run in my script.
