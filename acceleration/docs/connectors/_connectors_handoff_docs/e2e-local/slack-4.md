# Slack connector e2e, run 4: by-name grants and a fresh user (2026-10-08)

Router: `$R` at `daa71764` (#822, #823, #826), rebuilt in compose at 16:53:23Z (`build-4.log`).
Config `1b21facf…` (`e2e-slack`, `gemini/gemini-3.8-flash`).

| Who | Session | Connection |
|---|---|---|
| e2e-user-1 (old) | `01a11bde-0c6f-75dc-a86c-470bc7c3a820` | `0d476f229bd23f0bf45079afa5be0444`, rev 5 |
| e2e-user-2 (fresh) | `01a11c6f-ea36-7ca9-8c5d-3b2f5fd49eeb` | `6d92545ffbb5cd68402a9a35e1f681b1`, rev 6 |

Slack user, team and message ts values are redacted (`U<r>`, `T<r>`, `p<ts>`).
Commands: `lib.sh` `va`; `turn.sh` and `slack.sh` with `E2E_STATE=state-2.json` for e2e-user-2.
`PS` is `docker exec vision-agents-postgres-1 psql -U postgres -d model_router -Atc`.

## 0. Harness accident (F14)

- What happened: the helper restart killed the pid in `consent_helper.pid`, which was 82604, a bash wrapper. The python helper (82606, on `state.json`) kept listening on :3091.
- At 16:54:52Z a page load hit that old helper. It found no live consent for e2e-user-1, so `ask_again` re-sent the old prompt to session `01a11bde`.
- Gemini then called `slack__slack_send_message {"channel_id":"C0C8MKNUNBA","message":"hello from accelerate connectors e2e"}` (16:54:54.9Z). The message was posted to #kanat-test as Kanat. Nobody asked for that post.
- The old helper was killed by its python pid. `pgrep -fl consent_helper.py` at 16:59:38Z showed only 63364, the helper on `state-2.json`.

## 1. Router and catalog

- `/health` → 200. `goose_db_version` on `model_router`: `20261011150000` (16:53:24Z).
- `GET /v1/agents/connectors` → slack `revision: 6`.
- Neither that list nor `GET /v1/agents/connectors/slack` has `broken_revisions` (F13).
- The marks are seeded: `PS "select * from connector_broken_revisions where connector_id='slack'"` → revisions 1, 2, 3, 4, each with the reason from slack.yaml:30.
- The old connection `0d476f22…`: `definition_revision: 5`, `definition_status: outdated`, `status: connected`.
- `GET …/tools` → 200, 26 tools, `checked_at 15:20:41Z` (`tools-old-4.out`).
- The binding: `PATCH /v1/agents/configs/1b21facf…` at 16:54:08Z → 200. It now grants `[{"name":"slack_send_message"},{"name":"slack_read_channel"}]` with no `schema_digest` (`config-patch-4.out`).
- `PS "select count(*) from connector_tool_pins"` → 0 before the run.

## 2. Fresh user asks, gets a consent

`items-4.out`:
```
16:54:38.98Z said        Post "hello again from accelerate connectors e2e (fresh user)" to #kanat-test (channel C0C8MKNUNBA)
16:54:41.31Z tool_result slack__list_tools {"status":"authorization_required", …}
16:54:42.77Z answer      Please connect your Slack account using the button shown, …
16:59:11.91Z said        (same prompt again, re-sent by the helper's ?fresh=1 load at 16:59:14 local log)
16:59:13.38Z tool_result slack__list_tools {"status":"authorization_required", …}
```
The conversation has two consents:
- `d4cb02f4…`: `status: connected`.
- `3d21ac5f…`: `status: null`, expires 17:09:13Z. It was made by the re-sent prompt and stays pending against the now-connected connection.

The helper's log shows the popup's return at 12:59:20 local: `?connection_id=6d92545f…&status=connected`.

## 3. (a)+(b) Connected, and the automatic carry-on turn posts (F4 fixed)

`GET /v1/agents/connections/6d92545f…` as e2e-user-2 (`conn-4.out`, 16:59:38Z):
- `status: connected`, `definition_revision: 6`, `definition_status: current`, 29 scopes.
- `expires_at 2026-10-09T02:27:12.492522Z`, about the same as the old connection's `02:27:11.811955Z`. This supports F11: Slack hands the same token back to the same Slack user.

No validate, grant or session restart came before this turn:
```
16:59:20.82Z said        Slack is connected now. Carry on with what I asked for before you needed it.
16:59:22.54Z tool_call   slack__list_tools
16:59:22.54Z tool_result {"tools":[{"name":"slack_send_message","description":"Sends a message … the current logged in user's user_id is U<r>. …
16:59:24.38Z tool_call   slack__call_tool {"tool":"slack_send_message","arguments":{"message":"hello again from accelerate connectors e2e (fresh user)","channel_id":"C0C8MKNUNBA"}}
16:59:24.56Z tool_result {"message_link":"https://getstream.slack.com/archives/C0C8MKNUNBA/p<ts>","message_context":{"message_ts":"<ts>","channel_id":"C0C8MKNUNBA"}}
16:59:25.90Z answer      The message has been posted: https://getstream.slack.com/archives/C0C8MKNUNBA/p<ts>
```
Audit row: `GET /connections/6d92545f…/invocations` → `slack_send_message`, started 16:59:24.383Z, 179 ms, no `error_type`, session `01a11c6f`.

Slack's own confirmation comes from the read in §5. It shows «hello again from accelerate connectors e2e (fresh user)» at 12:59:24 EDT, from Kanat, «Sent using Accelerate connectors test».

## 4. (c) Pins (F8)

`PS "select connection_id, tool_name, schema_digest, connected_at, pinned_at from connector_tool_pins"` at 17:00:03Z:

| connection | tool | digest | connected_at | pinned_at |
|---|---|---|---|---|
| 0d476f22 | slack_read_channel | e44200743aed… | 15:20:02.188Z | 16:54:52.458Z |
| 0d476f22 | slack_send_message | e1e931415f4c… | 15:20:02.188Z | 16:54:52.458Z |
| 6d92545f | slack_read_channel | e44200743aed… | 16:59:19.895Z | 16:59:20.435Z |
| 6d92545f | slack_send_message | e1e931415f4c… | 16:59:19.895Z | 16:59:20.435Z |

- Each connection has its own rows, pinned when it was first opened after its connection (`internal/session/connector_tools.go:320-378`).
- **The digests do not differ.** A join on `tool_name` gives `same = t` for both tools.
- The reason: both connections are Kanat's one Slack account. Comparing `account_id` and `metadata->>'user_id'` in SQL gives `t|t`, and the id embedded in the description is the same.
- What is verified: per-connection pins, and a second router user who gets the tools with no digest grant. What is `unverified`: two **different** Slack users, which needs a second Slack account.
- F9 still holds. `GET /connections/6d92545f…/tools` → `{"tools":[]}`, `checked_at: null`, although the session listed and pinned from the provider.

## 5. (d) e2e-user-1 still works (rev 5, outdated)

`turn.sh` on session `01a11bde` at 17:00:18Z (`turn-4d.out`):
```
17:00:21.99Z tool_call   slack__slack_read_channel {"channel_id":"C0C8MKNUNBA","limit":2}
17:00:22.07Z tool_result Channel: #kanat-test … 12:59:24 EDT … hello again from accelerate connectors e2e (fresh user) … 12:54:55 EDT … hello from accelerate connectors e2e
17:00:23.79Z answer      The latest 2 messages in #kanat-test are: 1. "hello again … (fresh user)" 2. "hello from accelerate connectors e2e"
```
Audit rows on `0d476f22`:
- `slack_send_message` at 16:54:54.927Z (the §0 accident), with no error.
- `slack_read_channel` at 17:00:21.989Z, 82 ms, with no error.

An outdated rev-5 connection keeps working with by-name grants, and it has its own pins (§4).

## Verdict

- **F4 is fixed.** The carry-on turn had the granted tools and posted (§3).
- **F8 is half verified.** Pins are per connection, and two router users each got the tools (§4). Different Slack users with different digests stay `unverified`: the same Slack account was used for both.
- New findings: F13 (`broken_revisions` is not on the connectors API) and F14 (the harness helper restart).
