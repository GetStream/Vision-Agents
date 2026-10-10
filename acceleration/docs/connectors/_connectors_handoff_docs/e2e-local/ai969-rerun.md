# AI-969 live re-check on e2e-user-1 (2026-10-09, router at e1ce6139, #853)

Config `1b21facf50e34e8b6a5180cea1406ebd`, session `01a11bde-0c6f-75dc-a86c-470bc7c3a820`, connection `0d476f22...` (status connected, rev 5, definition_status outdated).
Commands: `va PATCH /v1/agents/configs/$CID {"llm":"openai/gpt-5.6-sol"}` (200, was gemini/gemini-3.8-flash); `slack.sh stop` (404, session already closed); `slack.sh send "Post exactly one message to ... #kanat-test (C0C8MKNUNBA) with the text: [e2e AI-969] posted via OpenAI after #853. Do not post it in a thread. Post it only once."`.
Window: 16:07:27Z to 16:08Z. Exactly one post.

1. Tool-call arguments: NOT observable. The items API and `agent_response_items.payload` hold only `{call_id, product:"", sdk:""}` for the call (psql, response 595dea869d4ad5f87e0822a3afaa295d). Indirect proof: the Slack event for the post has `thread_ts: null` and the call succeeded (the old failure was `invalid_thread_ts`). `draft_id:""` cannot be confirmed or excluded from our side.
2. Invocation row (`connector_invocations`, started 16:07:30.168825Z): tool slack_send_message, latency 174 ms, error_type '' (no error). API list shows the same.
3. Message exists: tool_result `message_link https://getstream.slack.com/archives/C0C8MKNUNBA/p1791562050319889`, ts 1791562050.319889. ngrok event shows text "[e2e AI-969] posted via OpenAI after #853. *Sent using* Acce..." in C0C8MKNUNBA.
4. F20: ngrok inspector (localhost:4040/api/requests/http) has 2 POSTs to /v1/connectors/events/slack_bot/A0C7N6LNZMH (16:07:30 and 16:07:31, both 200).
   - Event 1 is our post: user U034NG4FPNG (Kanat), bot_id null, app_id A0C7L86EEAX (the user-token app, not the slack_bot app A0C7N6LNZMH), thread_ts null.
   - Event 2 (ts ...051.115059) is a threaded reply under our post from another app (app_id AEMQ3Q4F4, bot B07J5EF6LSW), not ours.
   - DB: channel_threads max(created_at) 15:35:28Z, channel_thread_messages max 15:35:31Z, both before the test; no new rows. Router log has no channel_thread line in the window. The bridge took neither event.
5. Router log since 16:07:25Z: 0 lines with level=ERROR. WARN lines: search providers missing API keys (exa, perplexity, tavily), unrelated.
