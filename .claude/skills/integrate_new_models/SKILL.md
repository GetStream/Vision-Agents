---
name: integrate_new_models
description: Find the models worth adding with the new_models skill, then build support for each one in its own branch and open a draft PR per model.
---

# From "we should have this" to a PR per model

[new_models](../new_models/SKILL.md) says what is missing. This does the work, one model at a
time, one branch at a time, so each lands as a PR that can be judged on its own.

## 1. Get the list

Run [new_models](../new_models/SKILL.md) and keep its report. It has already dropped anything
in [`router.yaml`](../../../acceleration/internal/routing/router.yaml), so what is left is the
work. Drop anything else the report cannot give an API id and a docs page for — an integration
cannot be written against a blog post.

## 2. Decide how big each one is

Two very different jobs wear the same words. Before dispatching anything, check whether the
vendor already has a package for that modality — `internal/tts/<vendor>`, `internal/stt/<vendor>`,
`internal/llm/<vendor>`, `internal/sts/<vendor>`:

- **The package exists.** A new model from a vendor we already speak is usually a `router.yaml`
  entry plus whatever the new model does that the old one could not. Small.
- **The package does not exist.** A new vendor, or a modality we have never asked that vendor
  for, is a provider package, its registry entry, three test files and the config. Gemini TTS is
  this even though we already talk to Gemini for STT and speech-to-speech.

Say which of the two each model is in the report at the end. It is the difference between an
afternoon and a week, and it is the first thing a reviewer wants to know.

## 3. One subagent per model

Dispatch a `best-of-n-runner` subagent per model, in parallel, each with the model
`claude-opus-5-5-medium`. That subagent type takes its own git worktree and branch, which is what
keeps the work separable — without it two models end up in one diff and neither can be merged.

Branch per model: `model/<vendor>-<model-id>`.

Each subagent gets: the model's name, API id, type, docs URL and launch date from the report;
which of the two jobs in step 2 it is; and the skill to follow.

| Type | Skill to follow |
| --- | --- |
| TTS | [tts](../tts/SKILL.md) to build it, [router-tts](../router-tts/SKILL.md) for what config may claim |
| STT | [stt](../stt/SKILL.md) to build it, [router-stt](../router-stt/SKILL.md) for what config may claim |
| LLM | [router-llm](../router-llm/SKILL.md) |
| STS | [router-sts](../router-sts/SKILL.md) |

Tell each subagent to finish with a draft PR per [pr](../pr/SKILL.md), and to commit per
[commit](../commit/SKILL.md). A subagent that cannot get the model working must still say so and
leave the branch unpushed rather than open a PR on a guess — a router entry for a model nobody
has heard answer is worse than no entry, because routing will send real traffic at it.

The integration tests need vendor credentials. A subagent that cannot run them says which test it
could not run and why, in the PR body.

## 4. Report

Per model: the name, the type, which of the two jobs it turned out to be, the PR link, and
whether the integration test passed. Then the models that were dropped and why, so the next run
does not re-litigate them.
