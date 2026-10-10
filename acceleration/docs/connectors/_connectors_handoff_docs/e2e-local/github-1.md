# GitHub static-token connector e2e, run 1 (2026-10-09, local router)

- Router: `vision-agents-router-1`, image created 2026-10-09T15:26:00Z. The code it runs is `$S/accelerate-run` at `1d34dc26` (AI-990, #850).
- Auth: api_key mode, with `lib.sh` (`va`). `ROUTER_CONNECTORS_ENABLED` is set in the container (`docker exec … env`, names only).
- Secrets:
  - The PAT is read from `.env` `E2E_GITHUB_PAT` into an env var only (`gh/ghlib.sh`). Shape: `len=93 prefix=github_pat_`.
  - Request bodies that carry it are built by `jq … env.PAT` and sent on stdin (`va_in`, `--data-binary @-`). The PAT is never on argv or stdout.
- Files: `$S/e2e/gh/` holds `NN-*.out` (API responses), `turn-*.out` (session items; tool_result text replaced by its length), and `router-gh.log` (router log since 15:42:40Z).

## Ids

| What | Value |
|---|---|
| custom connector | `custom_github` rev 1, schemes `[bearer, api_key]`, endpoint `https://api.githubcopilot.com/mcp/` |
| connection (good PAT) | `249a40aa2b11502c1b2ea33421ae945d`, owner user `e2e-gh-user`, scheme `bearer` |
| connection (wrong token) | `d70c105ba018c0decf6ac65c0dde1c8f`, scheme `bearer` |
| connection (api_key probe) | `68b45589be8b4b8ecc360d1f36cef30a`, scheme `api_key`, still pending |
| config `e2e-github` | `a0d658355741e9966ead771256fa74f1` |
| session (good) | `01a12155-9d5c-7855-83fd-fffc4552115c` |
| session (wrong token) | `01a12158-3271-721d-b240-ce90190999b1` |

## 1. Connector definition: PASS

**Built-in.** A `github` connector exists, but it is OAuth only:
- `GET /v1/agents/connectors/github` at 15:42:45Z → 200, `revision 2`, `schemes ["oauth2_code"]`, `client.registration [operator, customer]` (`01-builtin-github.out`).
- The manifest is `providers/github.yaml:46` (`schemes: [oauth2_code]`).
- No built-in takes a PAT for GitHub. Only `linq`, `telnyx` and `whatsapp` list `bearer` (catalog listing, same call).

**How GitHub wants the PAT.** The README of github/github-mcp-server, `main` @ `eb47a99d` (fetched 15:41:50Z), lines 69-79, gives a PAT config for `https://api.githubcopilot.com/mcp/` with `"Authorization": "Bearer ${input:github_mcp_pat}"`:
https://github.com/github/github-mcp-server/blob/eb47a99ddb866ca2b8a162920e6bda9521f33ebb/README.md

**Scheme choice: `bearer`, not `api_key`.**
- `bearer` sends exactly `Authorization: Bearer <token>` (`schemes/bearer/scheme.go:35-37`, `:177-180`).
- `api_key` refuses the `Authorization` header (`schemes/apikey/scheme.go:209-216`). Live check at 15:46:24Z: `PUT …/68b45589…/credentials` with `{api_key:"Bearer <PAT>", header:"Authorization"}` → 400 `the credentials were refused: apikey: the header is one the router sets itself or must not send to a provider` (`17-apikey-put.out`).

**Custom definition.** `POST /v1/agents/connectors` (registered at `internal/api/connectors.go:218-231`, handler `:291-308`, `customManifest` `:312-367`).
At 15:42:53Z it returned 200: `custom_github` rev 1, `custom:true`, schemes `[bearer, api_key]`, scopes `[]`, client `{}` (`02-custom-create.out`).

## 2. Connection, credential, validate, tools: PASS

| Time (Z) | Call | Result |
|---|---|---|
| 15:43:04 | `POST /v1/agents/connections` `{connector_id:custom_github, owner:{type:user,user_id:e2e-gh-user}, auth_scheme:bearer}` as `e2e-gh-user` (`connections.go:187-202`) | 201, `status pending`, `revision 1` |
| 15:43:13 | `PUT /v1/agents/connections/249a40aa…/credentials` `{expected_revision:1, values:{token:<PAT>}}` (`connection_tools.go:147-163`, handler `:197-271`) | 200, `status connected`, `revision 2`, `granted_scopes []`, no `expires_at` |
| 15:43:19 | `POST …/validate` `{}` (`connection_tools.go:164-179`, handler `:274+`) | 200 in 1205 ms, `status connected`, `tools_digest f54e3703…`, `checked_at 15:43:21.084542Z` |
| 15:43:21 | `GET …/tools` | 200, **49 tools**, the same digest |

The 49 tools are the GitHub server's default toolset. Read tools include `get_me`, `search_repositories`, `list_issues`, `list_pull_requests`, `issue_read` and `pull_request_read`. Write tools include `issue_write`, `create_pull_request`, `merge_pull_request`, `push_files` and `delete_file`. The full list is in `06-tools.out`.

The client was a mistake here, not the router: the first create at 15:42:59Z passed the user id in the wrong `va` argument and got 400 `owner.user_id must be the user this backend acts for`.

## 3. Agent config, session, answers: PASS

**Binding type: `session`.**
- A `fixed` binding must name an **app-owned** connection (`configs.go:524-537`, and the enum doc at `:1714-1717`). The PAT connection is the user's own, so the binding has to be `session`.
- The session picks the connection with `connector_bindings` at create (`schemas.go:260`, `:306-309`).
- Tools are granted **by name only**, read tools only: `get_me`, `search_repositories`, `list_issues`, `list_pull_requests`, `issue_read`, `pull_request_read`. No write tool is offered to the model.

Steps:
- `POST /v1/agents/configs` (`gh/config-github.json`, `gemini/gemini-3.8-flash`, text) at 15:43:37Z → 201 `a0d65835…`.
- `POST /v1/agents/sessions` `{config_id, text:true, connector_bindings:[{name:github, connection_id:249a40aa…}]}` at 15:43:41Z → 201 `01a12155…`, `state live`. The log has no `opening the session without a connector` line for it.

Turns (`turn-a…d.out`). The model was offered `github__<tool>` directly: no `list_tools` or login.

| Turn | Tool calls (args) | Answer |
|---|---|---|
| a 15:43:53 | `get_me` → `search_repositories {query:"user:kanat"}` → `list_issues` + `list_pull_requests {owner:kanat, repo:rxandroidble_sample, state:open, perPage:3}` | user **kanat**; repos rxandroidble_sample, pion-android, fat-aar-sample, pokedex; no open issues, no open PRs |
| b 15:44:20 | `search_repositories {query:"is:private"}` | «No private repositories were found.» |
| c 15:44:43 | `search_repositories {query:"user:kanat fork:true"}`, then issues and PRs of rxandroidble_sample | 9 repos (matches `get_me` `public_repos: 9`), each with `0` open issues; none open |
| d 15:45:13 | `list_issues` + `list_pull_requests` on GetStream/Vision-Agents | issues #657, #650, #638; PRs #795, #779, #749 |

**Cross-check of turn d** with `gh issue list` / `gh pr list -R GetStream/Vision-Agents --state open --limit 3` at 15:45:31Z. The same 6 numbers, titles and created_at came back:
- issues: #657 «wrt our discussion in …/issues/646», #650 «telnyx examples: temp Call Control App leaks when the process is killed; no --log-level flag», #638 «Proposal: EvalPort adapter for testing/_judge.py's Judge/LLMJudge (open interchange format for eval results)»
- PRs: #795 «chore(sdk): keep port notes out of the skill», #779 «chore(deps): update plugin dependency version caps», #749 «Voice turn-taking: faster replies without talking over the caller»

**Which repo the PAT was scoped to: `unverified`.**
- `is:private` returned 0 repos.
- Every repo found is public, and a fine-grained PAT reads all public repos. So nothing the MCP tools return tells the selected repo apart (F39).

## 4. Records, audit, logs, leaks: PASS, with gaps (F36)

**Invocations.** `GET /v1/agents/connections/249a40aa…/invocations` at 15:45:38Z → 10 rows (`09-invocations.out`).
- All 10 have binding `github`, session `01a12155`, config `a0d65835` and `error_type` null.
- Tools: `get_me` ×1, `search_repositories` ×3, `list_issues` ×3, `list_pull_requests` ×3. Latency is 295-889 ms.

**Pins.** `connector_tool_pins` for the connection: 6 rows, `pinned_at` 15:43:42.73Z, at session open.
- Each digest equals the one validate listed: `get_me 2749ec586d78`, `search_repositories dc82cad66e84`, `list_issues 0154a1871760`, `list_pull_requests d840cef6f5b1`, `issue_read 37efe05ecbb3`, `pull_request_read e9dba12c6845`.
- The connection row: `bearer`, `revision 2`, `credentials_sealed` 175 bytes, `kek_version 1`, `expires_at` NULL, `tools_checked_at` 15:43:21.084542Z.

**Audit.** `GET /v1/agents/connector-audit?connection_id=249a40aa…` (`connection_records.go:168-180`) → 1 row: `grant_created`, `reason credentials`, `revision 2`, `request_id e117f73e…` (the PUT), 15:43:13.81937Z.
- **There is no `credential` field.**
- This is by code: only `oauth2_code` implements `core.Fingerprinter` (`schemes/oauth2code/scheme.go:128`). For any other scheme, `FingerprintsOf` returns zero (`core/fingerprint.go:75-79`), and `auditGrant` leaves `Credential` unset (`internal/api/connections.go:464-479`).

**Log.** One `connector credential event` line per change (`router-gh.log` line 7, 15:43:13.819Z):
`event=grant_created connection=249a40aa… connector=custom_github revision=2 reason=credentials previous_access_fingerprint="" access_fingerprint="" … rotated=false access_expires_at=0001-01-01T00:00:00.000Z refresh_expires_at=0001-01-01T00:00:00.000Z`
- The fingerprints are empty and the expiry is Go's zero time.
- The log has **no ERROR line** (`grep level=ERROR` = 0 over 140 lines, 15:42:40-15:47:03Z).
- The only WARNs are `search-fast` (no EXA/PERPLEXITY/TAVILY keys), plus the two in §5.

**Leak scan.** The check ran in-process (`leakcheck` in `gh/ghlib.sh`) for the full PAT, its first 20 characters, and the wrong token. Each result was not found:
- `router-gh.log` (15:47:03Z), all 18 `*.out` files and `turn-*.out`.
- A `pg_dump --data-only` of `model_router`, 705 421 bytes, checked at 15:46:04Z.
- Redis was not scanned (`unverified`).

## 5. Wrong token: PASS (clear status; wording is OAuth's, F37)

The wrong token is the PAT with its last character changed, built in-process. Check: same length, 1 position differs.

| Time (Z) | Call | Result |
|---|---|---|
| 15:46:15 | create `d70c105b…` (bearer) | 201 pending |
| 15:46:15 | PUT credentials `{token:<wrong>}` | 200 `connected`, rev 2 (nothing is checked at PUT) |
| 15:46:15 | POST validate | 200 in 282 ms: `{"status":"needs_reauthorization","error":"The provider rejected the grant; reconnect the account"}`, no `code` |
| 15:46:15 | GET connection | `status needs_reauthorization` |
| — | audit | `grant_created`/`credentials` 15:46:15.428Z, then `grant_revoked`/`invalid_grant` 15:46:15.739771Z, with no `credential` field |
| — | log | two `connector credential event` lines (`grant_created`, then `grant_revoked reason=invalid_grant`), with empty fingerprints |

- The message is `resolver.go:54` (`rejectedGrant`).
- GitHub's 401 becomes `invalid_grant` through `oauth2code.ClassifyStatic` (`classify.go:121-127`), which the bearer scheme uses (`bearer/scheme.go:131-133`).

**A session on the broken connection.** `01a12158…`, opened at 15:46:30Z → 201 live.
- Log 15:46:31.078Z: `WARN opening the session without a connector connector=github reason=needs_reauthorization`.
- On «Which GitHub user am I?», the model called `github__list_tools`. The router tried to begin a consent and refused it.
- Log 15:46:38.522Z: `WARN could not begin a consent in the conversation connector=github error="auth_scheme \"bearer\" needs no consent"` (`session/connector_login.go:304`, from `api/authorizations.go:347-348`).
- The tool returned `{"status":"unavailable","message":"github cannot be connected here right now, … do not offer to set it up."}`, and the agent answered «GitHub is not available right now…».
- Nothing leaked (§4 scan).

## Left in place

- The three connections, the config, the two live sessions and `custom_github` rev 1 are all local only.
- The good connection still holds the PAT, sealed. Delete it with `va DELETE /v1/agents/connections/249a40aa2b11502c1b2ea33421ae945d '' e2e-gh-user`.
- Bearer `Revoke` does not revoke the PAT at GitHub (`bearer/scheme.go:135-141`).
