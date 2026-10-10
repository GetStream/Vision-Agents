# Slack connector e2e, step 2: shape of Slack's token response (2026-10-08)

## The failure

Router log, 14:27:13Z (reported by the coordinator):
`a consent did not complete connection=9c3d34a08dcdcceb682ace9b9cc09a77 error="oauth2code: capture user_id: token_response has no $.authed_user.id"`

- The error is built at `acceleration/internal/connectors/core/manifest.go:711`.
- The rule is in `internal/connectors/providers/slack.yaml`: `user_id` from `token_response`, path `$.authed_user.id`.

## Debug build (temporary, never committed)

- Worktree `$S/e2e/dbg`: detached at `0bc2fe89`, with uncommitted changes only. `$R` is untouched.
- Change: `acceleration/internal/connectors/schemes/oauth2code/exchange.go`, in `exchange()` right after `decodeObject(raw)`:
  `slog.Info("E2E DEBUG token response shape", "status", <http status>, "shape", "<path>:<type> …")`.
  - The `e2eShape` helper replaces every value with its JSON type: `string`, `number`, `bool`, `null`, `object` or `array(n)`.
  - Nested keys are joined with `.`. Array elements are written as `path[]`.
  - It logs before the error checks, so an error body is logged too.
- Format check with a fake body (throwaway test, now deleted):
  `access_token:string authed_user:object authed_user.id:string authed_user.scope:string expires_in:number … ok:bool team:object team.id:string`.
  No values appear in the output.
- Build and swap. `$S/e2e/compose.dbg.yaml` only overrides `router.build.context` to `$S/e2e/dbg/acceleration`. The env still comes from `$S/compose.e2e.yaml`.
  ```
  cd $R && docker compose -p vision-agents -f compose.yaml -f $S/compose.e2e.yaml -f $S/e2e/compose.dbg.yaml up -d --build --no-deps router
  ```
  - New image: `sha256:cba6d8e9ad2a…`. The binary contains the debug string (`grep -c` gave 1).
  - The env is unchanged: `DASHBOARD_BASE_URL=http://localhost:3091`, `ROUTER_AUTH_MODE=api_key`.
- Original image kept as tag `vision-agents-router:orig-0bc2fe89` (`sha256:42a689724d25…`).

## Fresh consent (READY)

- I called `curl 'http://localhost:3091/?fresh=1'`: HTTP 200 in 2.3 s.
  - Consent `3c26b59d20c5d1bfcd70abc8acd22f2f`, connection `9c3d34a08dcdcceb682ace9b9cc09a77`.
  - It expires at 2026-10-08T14:39:56Z.
  - The page has the Connect Slack button.
- The session reopened on the new container (log 14:29:54.811Z `session joined session=01a11bde-…`).
- Kanat opens **http://localhost:3091/**. If the consent has expired, the page asks the agent again.

## Logged key structure

Kanat clicked. One line was logged at 2026-10-08T14:32:28.412Z, HTTP 200 (`$S/e2e/shape.log`):
```
access_token:string app_id:string enterprise:null expires_in:number is_enterprise_install:bool ok:bool
refresh_token:string scope:string team:object team.id:string team.name:string token_type:string user_id:string
```
- There is no `authed_user`. The user id is at the top level as `user_id`.
- `refresh_token` and `expires_in` are present, so rotation is on.
- The same consent then failed with `capture user_id: token_response has no $.authed_user.id` (14:32:28.413Z).
- The fix is PR #822 (`89966e26`): `slack.yaml` rev 5 captures `$.user_id`.

## Slack docs

Opened 2026-10-08 at about 14:30Z.

1. [oauth.v2.user.access](https://docs.slack.dev/reference/methods/oauth.v2.user.access).
   The example success response has `authed_user: {id, scope}` and `team: {id}` next to a top-level
   `access_token` and `token_type: "user"`.
   This example is where `slack.yaml` took `$.authed_user.id`.
   It has no `refresh_token` and no `expires_in`, so it is not a token-rotation example.
2. [Using token rotation](https://docs.slack.dev/authentication/using-token-rotation), from `curl` of the page.
   It says: "If you make use of a user token, expect those two new fields along with your access token in the response:"
   ```
   "id": "U1234", "scope": "chat:write", "access_token": "xoxe.xoxp-1-1234-...",
   "expires_in": 43200, "refresh_token": "xoxe-1-..." "token_type": "user"
   ```
   The fragment is for `oauth.v2.access`, and it does not show the object that holds `id`.
   It could be the contents of `authed_user`, or the top level of the response.
   The page does not show a full `oauth.v2.user.access` response with rotation on.
   So whether `authed_user` is absent with rotation is `unverified` from the docs. The logged shape will settle it.
   Hypothesis to check against the log: the user id is at the top level as `id`, and there is no `authed_user`.

## Restore the original router

```
cd $R && docker compose -p vision-agents -f compose.yaml -f $S/compose.e2e.yaml up -d --build --no-deps router
```
- This builds from `$R/acceleration` with the same env.
- Faster alternative: `docker tag vision-agents-router:orig-0bc2fe89 vision-agents-router`, then the same command without `--build`.
- Then remove the worktree: `git -C /Users/kanat/Projects/stream/Vision-Agents worktree remove --force $S/e2e/dbg`.

## Restored, on the fix (step 3)

- `$R` is at `89966e26` (AI-816, #822). `git merge-base --is-ancestor 89966e26 HEAD` passes.
- Rebuilt with `cd $R && docker compose -p vision-agents -f compose.yaml -f $S/compose.e2e.yaml up -d --build --no-deps router`.
  - New image `sha256:c2341761fe07…`, started 15:16:13Z.
  - `E2E DEBUG` appears 0 times in the log and 0 times in the binary.
- Removed the `$S/e2e/dbg` worktree, `compose.dbg.yaml` and the `orig-0bc2fe89` tag.
- `GET /v1/agents/connectors` shows `{"id":"slack","revision":5}`.
- `DELETE /v1/agents/connections/9c3d34a08dcdcceb682ace9b9cc09a77` (rev 4, pending) returned HTTP 204.
  A GET then returned 404 `connection_not_found`. The list for `owner_type=user&connector_id=slack` is `[]`.
- A fresh consent via `http://localhost:3091/?fresh=1` (HTTP 200):
  - consent `f8ebe7ca7590421bc9bcd48a26c8929b`, expires 15:26:33Z;
  - new connection `0d476f229bd23f0bf45079afa5be0444`, `definition_revision: 5`, status pending, owner user `e2e-user-1`.
