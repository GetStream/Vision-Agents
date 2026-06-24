# Telnyx Example Fix Plan

## Branch Check

- Project root: `/Users/nash/git_projects/stream/video_ai/vision-agents`
- Active branch: `codex/add-telnyx-phone-plugin`
- Main workspace status before implementation: clean
- Temporary worktree `/private/tmp/vision-agents-pr594` is detached and contains ad hoc live-test scripts only; do not use those untracked files as-is.

## Goal

Make the Telnyx plugin PR usable for developers by adding documented,
plugin-local inbound and outbound phone examples that can run with an explicit
Telnyx setup flag, explain the required Call Control setup, and fail early with
actionable setup errors.

## Plan

1. Confirm the editable PR branch/worktree and avoid mixing unrelated local changes.
2. Add minimal Telnyx inbound and outbound example scripts under `plugins/telnyx/examples`.
3. Document Call Control App setup clearly:
   - webhook URL
   - phone-number routing for inbound
   - `connection_id` for outbound
   - ngrok/local development URL
   - destination-number verification restrictions
4. Add an explicit `--setup-telnyx` development path:
   - create a temporary Call Control App
   - auto-detect local ngrok where possible
   - route the inbound phone number for inbound runs
   - restore routing and delete temporary app on normal shutdown
5. Add preflight checks so bad setup gives useful errors before a developer places a call:
   - missing env vars
   - invalid Call Control App ID
   - webhook URL mismatch
   - inbound phone number not routed to the Call Control App
   - unverified outbound destination where detectable
6. Add focused tests around helper/preflight logic without making live Telnyx API calls.
7. Run formatting, linting, relevant Telnyx/example tests, and live outbound/inbound smoke checks.

## Implementation Notes

- Do not silently create/delete Call Control Apps in the default example; require `--setup-telnyx`.
- Do not silently reroute a developer's Telnyx phone number; require `--setup-telnyx`.
- The README should show the `--setup-telnyx` quick start and manual setup alternative.
- Any account-mutating helper must be explicit, opt-in, and cleaned up on normal shutdown.
- Keep `examples/03_phone_and_rag_example` unchanged in this PR. It is a Twilio + RAG demo, while these examples should be a minimal Telnyx plugin starting point.
