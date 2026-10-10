# Incident cleanup 2026-10-09 (ids only)
Stream (app from .env), hard-deleted: channels agent:thread-911ee886-b4a3-4d49-83ca-1f3fe862028d, agent:thread-7b6aceb1-bd6b-4ec9-ad67-248fa8fb7108
Stream cards hard-deleted: episode-c9c55ab05ea653b3693de7015abfafdc (in agent:omni-9693009c-218d-41ea-88c1-0dbf02ff4fa0), episode-b9a0dd4d9fa744ec05ee51fdd76bd2a0 (in agent:omni-16cfde16-e22e-4388-ac86-8b7870d6df63)
Postgres model_router, one tx: channel_threads 2, channel_thread_messages 6, episodes 2, episode_activity 2, agent_sessions 2 (01a120e7-45a6-77c7-94c2-4c845d091e1f, 01a120e7-c54b-7482-928c-bf2ad6c9314f), calls 2, turns 2 (65,66), requests 14 (186-197,227,228), agent_logs 20, agent_responses 2, agent_response_items 4, agent_session_tools 2
Left in place: channel agent:omni-9693009c-... and contact_map row e121a725309693229d4671f84cc368b7 (other person's contact, created 13:43:09 by the leak)
Verify: channel query 0; card GET not found x2; full-table regex scan 0 matches; channel_threads D0B0GM3H7K6% = 0 (9 other threads kept)

## Follow-up (coordinator-approved)
Hard-deleted Stream channel agent:omni-9693009c-218d-41ea-88c1-0dbf02ff4fa0 (created 13:43:10.075Z, 0 messages, 0 members) and contact_map row e121a725309693229d4671f84cc368b7 (created 13:43:09.226Z). No other rows referenced them.
Verify: channel query 0; public-table scan for omni-9693009c / contact id / U09J46YEC7R = 0.
