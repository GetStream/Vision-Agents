# E2E fixes (AI-989 High)
status: done (v0.6.29 deployed 15:07Z)   base: accelerate d65f5e17   brief: <scratchpad>/waves/3d-brief.md
Kanat 2026-10-09: «запусти 2 агентов чтобы пофиксить High проблемы» (= OK for these behaviour fixes; connectors are off on shared envs).

| pr | finding | branch | worktree | db | slot | head | verdict | state |
|---|---|---|---|---|---|---|---|---|
| f20 | F20 agent posts via user token wake slack_bot + F26 mention gating (Kanat 2026-10-09) | connectors/slack-bot-skip-app-posts | <S>/wt-f20 | model_router_test_e2e_f20 | none | 396d7f6e (#849) | | review |
| f23 | F23 episode card dispatched as a message | connectors/episode-card-no-dispatch | <S>/wt-f23 | model_router_test_e2e_f23 | none | 138c4b9d (#848) | | review |

Side check: Slack refresh-token rotation (agent, local DB only) -> e2e/rotation/
## Agent runs
| f20 | author | opus | 1 | a08c2e8017274ee59 | running |
| f23 | author | opus | 1 | aeff3f31e26cd334f | running |
| rotation | check | opus | - | a2aa0e598a9e55ef1 | running |
| fp | author | opus | 1 | a9d37bc79650adb1c | running |
rotation: done 13:48Z — Slack rotates refresh token every refresh; router stores and uses it (e2e/rotation/rotation-1.md); 119k tok, 6m.

## Incident 2026-10-09 13:43Z
slack_bot app (SLACK_BOT_E2E_APP_ID) has a user-token install by Kanat; message.im events for his DM D0B0GM3H7K6 with U09J46YEC7R reached the local bridge; threads thread-911ee886…, thread-7b6aceb1… have reply rows (reply possibly posted, unverified). ngrok stopped ~13:52Z. Guard (authorizations must include bot install) added to f20 scope. Pending Kanat: check DM, remove user scopes/user events from app, OK to delete the two threads + cards.
| f23 | author | opus | 1 | aeff3f31e26cd334f | done 151k 11.7m; PR #848 |
| f23 | reviewer | opus | 1 | a6b0a5300cf62c795 | running |
| f20 | author | opus | 1 | a08c2e8017274ee59 | done 228k 27.7m; PR #849 |
| f20 | reviewer | opus | 1 | a1d18847480549cb1 | running |
| fp | author | opus | 1 | a9d37bc79650adb1c | done 244k 26.6m; PR #850 d45aff28, slot 20261011180000 |
| fp | reviewer | opus | 1 | ae5dfe0da06705a75 | running |
| f20 | reviewer | opus | 1 | a1d18847480549cb1 | NO-GO 162k 16.7m (R1.1, R1.2 Should fix) |
| f23 | reviewer | opus | 1 | a6b0a5300cf62c795 | GO 142k 21.5m; update-branch -> 6bc0236d trivial; CI go: TestDisplaySuite fail (also #850), local 30/30 pass; rerun |
| f23 | merge | - | - | - | #848 MERGED f6c1783a (head 6bc0236d; CI go rerun x2 passed; base moved by benchmark-only commits) |
| f20 | fixer | sonnet | 1 | ad737d3c076a8dd2f | done 103k 7.6m; head b7eafe77 (tests only + clean base merge) |
| f20 | reviewer | opus | 2 | a1d18847480549cb1 | delta running |
| fp | reviewer | opus | 1 | ae5dfe0da06705a75 | NO-GO 150k 21m (R1.1-R1.5 Should fix, missing tests) |
| fp | fixer | sonnet | 1 | a3e0034034f96af1d | running |
| f20 | reviewer | opus | 2 | a1d18847480549cb1 | GO 174k 5.3m |
| f20 | merge | - | - | - | #849 MERGED 921ae7e9 (head b7eafe77, CI green). AI-989 Done. |
| fp | fixer | sonnet | 1 | a3e0034034f96af1d | done 109k 6.9m; head 810fd7aa |
| fp | reviewer | opus | 2 | ae5dfe0da06705a75 | delta running |
| fp | reviewer | opus | 2 | ae5dfe0da06705a75 | GO 161k 4.7m |
| fp | merge | - | - | - | #850 MERGED 1d34dc26 (head 810fd7aa, CI green) |
| deploy | mechanic | sonnet | - | a3d1774785a1ffd7f | v0.6.29 @1d34dc26 running |
| deploy | mechanic | sonnet | - | a3d1774785a1ffd7f | STOPPED 52k: staging runs Nash dev build v0.6.9-dev-cf6e9723 (02:52Z), goose 20261011150000; cf6e9723 is ancestor of 1d34dc26; waiting Kanat |
Kanat 2026-10-09: deploy v0.6.29 over Nash dev build (rollback -> v0.6.9-dev-cf6e9723); delete Jona DM threads; ngrok after he fixes the bot manifest.
| deploy | mechanic | sonnet | - | a3d1774785a1ffd7f | resumed, deploying |
| cleanup | mechanic | sonnet | - | a84f741511bb45a15 | deleting DM threads/cards (local DB + Stream) |
| cleanup | mechanic | sonnet | - | a84f741511bb45a15 | DONE 75k: DM threads, cards, Jona omni-channel + contact_map deleted; scans 0 (e2e/incident-cleanup.md) |
| deploy | mechanic | sonnet | - | a3d1774785a1ffd7f | DONE 59k: v0.6.29 @1d34dc26 15:07Z, goose 20261011180000, 0 restarts, 0 ERROR, no 5xx, health 200 |
E2E rev4 verified 15:36Z: e2e/slackbot-4.md

## Next (Kanat 2026-10-09: «1 - запускай», GitHub PAT added)
| 969 | author | opus | 1 | aa8c946f1575b26fb | running; branch connectors/openai-optional-tool-args, db model_router_test_e2e_969, slot 20261011190000 if needed |
| gh-e2e | e2e | opus | - | a7c683df6501f8b9b | running; e2e/github-1.md |
| 969 | author | opus | 1 | aa8c946f1575b26fb | PR #853 854c947e 111k 5.9m; root cause: no strict -> OpenAI auto-strict makes all props required; asked to scope strict:false to schemas with optional fields |
| gh-e2e | e2e | opus | - | a7c683df6501f8b9b | DONE 184k 8m: PASS via custom bearer connector; findings renumbered F35-F39 -> AI-990; PAT connection 249a40aa… kept locally |
| 969 | author | opus | 1b | aa8c946f1575b26fb | head 15500e37, conditional strict, 128k total |
| 969 | reviewer | opus | 1 | a6a9a4fb0755e4656 | NO-GO 95k 4.1m (R1.1 Should fix: anyOf test) |
| 969 | fixer+delta | sonnet/opus | 1-2 | a0459bb3/a6a9a4fb | GO; merged |
| 969 | e2e rerun | sonnet | - | a5539a54c98406fe7 | PASS 16:07:30Z (e2e/ai969-rerun.md); AI-969 Done; F40 -> AI-990; local config e2e-slack now on openai/gpt-5.6-sol; #853 not on staging |
| migrate-e2e | e2e | opus | - | aca7500fc6ca16d71 | running (local plugins migrate; Kanat: local DB needs no approval) |
