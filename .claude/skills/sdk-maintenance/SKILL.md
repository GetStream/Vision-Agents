---
name: sdk-maintenance
description: Change, generate, test or release the Stream acceleration SDKs in this repository. Use for API parity, resource wrappers, language conventions and SDK update checkpoints. For applications that use the SDKs, use sdk.
---

# Maintain an SDK

## Scope the change

Read the target SDK's manifest, implementation, tests and `sdk-<language>` skill, if one
exists. Use [sdk](../sdk/SKILL.md) only for application guidance. Start a shared API change
in Go, then track each other SDK's required work in the feature PR or a linked issue.
Name the affected language, missing behavior and verification still needed. Do not claim
parity from generated types alone.

Use the current implementation and tests to resolve historical guidance. The
[legacy SDK updates](references/legacy-sdk-updates.md) preserve the former shared skill
byte for byte for checkpoint comparison. That file contains obsolete instructions and
contradictory examples. It is an archive, not a workflow to execute. Search only the
relevant feature or checkpoint diff; do not load the whole archive for ordinary work.
Do not append new notes there or to the application skill.

## Implement and verify

1. Read the affected Go API declaration and `acceleration/api/openapi.yaml`. The spec is
   generated from Go; change the declaration and regenerate it rather than editing YAML.
   See [repository API rules](../../../AGENTS.md#http-api).
2. Run the target SDK's generator as documented in its language guide or README. Review
   the generated diff for new fields, nullability, enums and client access restrictions.
3. Add or update the public resource method and typed options. Group by resource, nest
   subresources and use the language's normal naming. Application examples should use
   those methods rather than raw HTTP or generated operation functions. If a wrapper is
   missing, implement it before documenting it as available.
4. Check behavior beyond generation: folder sync, streaming events, tool dispatch,
   cancellation, session lifecycle and authentication where the change touches them.
   Keep omitted optional values distinct from explicit zero, false or empty values.
5. Add a regression test through the public API for the behavior being fixed. Run the
   language's required format, generation and test checks. Compile updated examples.
   Record live checks separately from offline checks and report skipped checks.
6. Update current usage documentation in place. Put user-facing changes in the changelog
   under the repository's rules. Put remaining cross-language work in the PR or linked
   issue, with enough detail for the next SDK maintainer to find and verify it.

Server-only operations include configuring agents and listening for dispatch. Client
generation must respect `x-client-accessible`; do not expose server credentials on a
device. JS serves both Node and browsers, so verify the authorization boundary as well
as its types. Preserve each SDK's dependency policy; JS has zero runtime dependencies.

Use the client to obtain router handles so endpoint and identity are inherited. Keep
model targets in router configuration and per-call overrides in the call options.
When changing endpoint resolution, test an ordinary resource call after construction;
testing the constructor alone missed the Go region regression.

## Update checkpoints

Each `sdks/<language>/.sdk_update_log` stores content IDs, not commit IDs:

```text
spec <blob ID of acceleration/api/openapi.yaml>
skill <blob ID of the maintenance guidance last reconciled>
```

Blob IDs survive squash merges. The two entries describe separate work: regenerated
types may match a new spec while handwritten behavior still needs an older checkpoint.
Do not advance either entry merely because the files were reorganized.

For an SDK with an existing checkpoint:

1. Read its two recorded IDs. Compare the spec ID with the current spec using
   `git diff <spec-blob> HEAD:acceleration/api/openapi.yaml`.
2. Inspect the recorded skill blob with `git show <skill-blob>`. If its frontmatter names
   `sdk`, it predates this split. Compare it with
   `HEAD:.claude/skills/sdk-maintenance/references/legacy-sdk-updates.md` to find historical
   work not yet reconciled. Then read the current maintenance guide. Do not compare that
   old checkpoint with the rewritten application skill.
3. If the recorded skill names `sdk-maintenance`, compare it with
   `HEAD:.claude/skills/sdk-maintenance/SKILL.md` instead.
4. Check feature PRs and linked issues since the checkpoint for cross-language work.
   The spec diff cannot reveal every handwritten SDK behavior change, and this guide is
   not a feature backlog. Resolve relevant archive notes against current code and tests.
5. Advance each checkpoint only after its applicable work is complete and verified.
   For the skill entry, use the new maintenance guide's blob ID. Preserve an older entry
   while work remains; describe the outstanding work in the PR or linked issue.

Obtain committed IDs with `git rev-parse HEAD:<path>`. For changes not yet committed,
use `git hash-object <path>`. After the migration, the skill path is
`.claude/skills/sdk-maintenance/SKILL.md`. Existing IDs remain valid Git objects; the
archive is kept unchanged so older SDKs can still calculate their outstanding diff.

## Before release

Read the target package's release workflow and verify the intended commit, version and
registry. Confirm that documented APIs are in that release and list compatibility gaps.
Changing a skill or checkpoint does not publish a package. Publish only when the task
authorizes a release.
