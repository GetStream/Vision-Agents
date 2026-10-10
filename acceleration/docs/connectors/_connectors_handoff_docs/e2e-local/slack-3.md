# Slack connector e2e, step 3: connected, tools, post and read (2026-10-08)

Router: built from `$R` at `89966e26` (#822). `slack` is at revision 5.
Ids: config `1b21facf50e34e8b6a5180cea1406ebd`, session `01a11bde-0c6f-75dc-a86c-470bc7c3a820`,
user `e2e-user-1`, connection `0d476f229bd23f0bf45079afa5be0444`.
Slack user, message and team ids are redacted; message ts values are shown as `<ts>`.
Commands: `slack.sh`, and `turn.sh "<text>"`, which sends one turn and prints its items redacted.

## 1. Connected

`slack.sh connection` at about 15:20Z:
`status: connected`, `definition_revision: 5`, `granted_scopes`: 29.
- Metadata keys are `team_id` and `user_id`. `account_id` is present.
- `expires_at: 2026-10-09T02:27:11.811955Z`.
- The response has no `connected_at` field.

## 2. F4: the automatic "connected, carry on" turn

```
15:20:02Z said        Slack is connected now. Carry on with what I asked for before you needed it.
15:20:06Z answer      One moment, I'm checking the available Slack actions now that Slack is connected.
15:20:06Z tool_call   slack__list_tools
15:20:06Z tool_result {"tools":[]}
15:20:10Z answer      Slack is connected, but no posting action is currently available. I couldn't send the message to **#general**.
```
The hand-back works: the login opened on the new connection. But the binding grants `tools: []`, so the turn had no Slack tools.

## 3. Tools

- `GET /connections/{id}/tools` first returned `{"tools":[],"checked_at":null}` (F9).
- `POST /v1/agents/connections/0d47…/validate {}` → 200, `status: connected`, `checked_at 15:20:41Z`, `tools_digest ff041752…`.
- Then 26 tools:

`slack_send_message, slack_schedule_message, slack_add_reaction, slack_create_conversation,
slack_create_list, slack_update_list, slack_add_list_record, slack_update_list_record,
slack_create_canvas, slack_update_canvas, slack_search_public, slack_search_public_and_private,
slack_search_channels, slack_search_users, slack_read_channel, slack_read_thread,
slack_read_canvas, slack_read_user_profile, slack_list_channel_members, slack_read_file,
slack_read_list, slack_list_user_channels, slack_send_message_draft, slack_search_emojis,
slack_get_file_upload_url, slack_complete_file_upload`

No tool's `needs_scopes` is set.

Self-DM: `slack_send_message` covers it. Its description says «To DM a user, use their user_id as channel_id. If the user wants to send a message to themselves, the current logged in user's user_id is U<redacted>.»
That sentence puts the user's id into the tool's schema digest (F8).

## 4. Grants

- `slack.sh grant slack_send_message` → PATCH 200. The grant is `slack_send_message`, digest `e1e931415f4c…`.
- After that, `slack.sh stop` returned 204. The next turn reopened the session on the updated config, with the connection already chosen.
- Later, `slack.sh grant slack_send_message slack_read_channel` → `["slack_send_message","slack_read_channel"]`, then stop again.

## 5. Posting

The coordinator moved the target from Kanat's DM to `#kanat-test` (`C0C8MKNUNBA`) during the step.

| Time (Z) | LLM | Target | Tool args (as sent) | Result |
|---|---|---|---|---|
| 15:21:03 | openai/gpt-5.6-sol | own DM (`U<redacted>`) | `channel_id, draft_id:"", message, reply_broadcast:false, thread_ts:"", unfurl_app_links:false` | `execution_failed: invalid_thread_ts` |
| 15:21:49 | openai/gpt-5.6-sol | own DM, told to leave out thread_ts | same, again with `thread_ts:""` | `invalid_thread_ts` |
| 15:22:51 | openai/gpt-5.6-sol | #kanat-test | same, with `thread_ts:""` | `invalid_thread_ts` |
| 15:22:59 | anthropic/claude-sonnet-5-5 | #kanat-test | no call | `401 Invalid Anthropic API Key` (F12) |
| **15:23:48** | **gemini/gemini-3.8-flash** | **#kanat-test** | `{"message":"hello from accelerate connectors e2e","channel_id":"C0C8MKNUNBA"}` | **posted**: `{"message_link":"https://getstream.slack.com/archives/C0C8MKNUNBA/p<ts>","message_context":{"message_ts":"<ts>","channel_id":"C0C8MKNUNBA"}}` |

The DM was never sent. The config LLM is now `gemini/gemini-3.8-flash`. It was changed with `PATCH /v1/agents/configs/{id} {"llm":…}` plus `slack.sh stop`, to keep OpenAI's empty optional fields out of the test (F7).

## 6. Read

Prompt: «Read the latest message in the Slack channel #kanat-test (channel id C0C8MKNUNBA) and tell me what it says.»
```
15:24:01Z tool_call   slack__slack_read_channel  {"channel_id":"C0C8MKNUNBA","limit":1}
15:24:01Z tool_result Channel: #kanat-test (C0C8MKNUNBA) === Message from Kanat Kiialbaev <…> (U<redacted>) at 2026-10-08 11:23:48 EDT ===
                      Message TS: <ts>  hello from accelerate connectors e2e  *Sent using* Accelerate connectors test
15:24:02Z answer      The latest message in #kanat-test is from Kanat Kiialbaev and says: "hello from accelerate connectors e2e"
```
This is Slack's own confirmation of the post. The message appears as Kanat's, at 11:23:48 EDT (15:23:48Z), with the app attribution «Sent using Accelerate connectors test».

## 7. Audit rows

`GET /v1/agents/connections/0d47…/invocations` (`invocations.out`) returned 5 rows. Every row has `connector_id slack`, `binding slack`, this config and this session.

| started_at (Z) | tool | latency_ms | error_type |
|---|---|---|---|
| 15:21:03.277 | slack_send_message | 101 | external_server |
| 15:21:49.410 | slack_send_message | 111 | external_server |
| 15:22:51.368 | slack_send_message | 167 | external_server |
| 15:23:48.772 | slack_send_message | 160 | — |
| 15:24:01.147 | slack_read_channel | 107 | — |

These match the turns one for one. `invalid_thread_ts` is recorded as `external_server` (F10).

## 8. Token expiry and refresh

- `expires_at = 2026-10-09T02:27:11.811955Z`. The token value was not read.
- That is 40030 s after this consent (15:20:02Z). It is exactly 43200 s, Slack's 12 h, after 14:27:11.8Z, when the first rev-4 consent failed (F11).
- When it is renewed: `oauth2code` renews a token once less than the margin is left. The default margin is 1 minute (`internal/connectors/schemes/oauth2code/credential.go:19`, `:60-70`); `slack.yaml` rev 5 sets `refresh: {rotating: true, access_ttl: 12h}` and no margin.
- So the first call after **2026-10-09T02:26:11Z** refreshes the token. Renewal is lazy: nothing refreshes without a call.
- To test rotation then: `./turn.sh 'Read the latest message in #kanat-test (C0C8MKNUNBA)'` after 02:26Z, then `slack.sh connection`. Expect a new `expires_at` about 12 h later.

## New findings

F7, F8, F9, F10, F11 and F12 are in `$S/e2e/findings.md`. F4 is now verified.
